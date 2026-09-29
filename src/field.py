"""
Analytic task-space navigation field: constructs the *desired* SE(2) twist
field directly from the task specification (avoid each obstacle, reach the
target within an approach-orientation tolerance).

`stage1_2.py::build_exact_generators` evaluates `desired_velocity` (paired
with `desired_heading` for the orientation component) directly on a grid --
Stage 1's twist field is read from this formula, not learned from data.
See SIMPLIFICATIONS.md item 1 and Phase 11.
"""

import numpy as np
from problem import wrap_angle


def bow_signs(problem):
    """Consistent side to route around each obstacle, fixed once for the
    whole workspace rather than decided per point (one sign per obstacle,
    same per-obstacle rule as before this was generalized to multiple
    obstacles). A purely radial repulsive term creates a saddle point /
    local minimum directly behind an obstacle on the line from a start
    point to the goal whenever the obstacle sits between them; committing
    to one consistent circulation direction per obstacle (like a
    single-vortex panel method) avoids that instead of needing per-point
    disambiguation.

    Caveat (SIMPLIFICATIONS.md): this per-obstacle rule is not guaranteed
    robust once multiple obstacles' repulsion is summed -- two obstacles
    that land on the *same* side of the base-target line get the *same*
    circulation direction, which can create a new compound local minimum
    between them that no single obstacle's swirl fixes on its own
    (empirically confirmed during this feature's design: some same-side
    placements failed up to ~19% of streamline integrations with a
    consistent stall distance, others didn't -- it depends on the specific
    geometry, not just same-side-vs-opposite-side). The default two-obstacle
    problem was chosen to avoid this, not because the rule generalizes
    safely for arbitrary placements/obstacle counts.
    """
    base = np.array(problem.base, dtype=float)
    target = np.array(problem.target, dtype=float)
    d = target - base
    signs = []
    for cx, cy, _r in problem.obstacles:
        obs = np.array([cx, cy], dtype=float)
        cross = d[0] * (obs[1] - base[1]) - d[1] * (obs[0] - base[0])
        side = np.sign(cross) if cross != 0 else 1.0
        signs.append(-side)  # bow away from the obstacle
    return signs


def _smoothstep(w):
    w = np.clip(w, 0.0, 1.0)
    return w * w * (3 - 2 * w)


def attractive_velocity(p, target, d_star=3.0, k_att=1.0):
    """Velocity pulling toward `target`: proportional to distance close in
    (gradient of a quadratic potential, i.e. a spring), capped to constant
    speed beyond `d_star` (gradient of a conic potential) so it stays
    bounded far from the goal."""
    diff = np.asarray(target, dtype=float) - np.asarray(p, dtype=float)
    dist = np.linalg.norm(diff)
    if dist < 1e-9:
        return np.zeros(2)
    return (k_att * min(dist, d_star) / dist) * diff


def repulsive_velocity(p, obstacle_center, obstacle_radius, sign,
                        influence=2.5, k_rep=6.0, swirl_frac=0.85, mag_cap=40.0):
    """Velocity pushing away from the obstacle. Blends from purely radial
    repulsion at the edge of the influence region to a mostly tangential
    "swirl" close to the surface (direction fixed by `sign`), so the field
    routes consistently around one side instead of stalling at the saddle
    point a plain (non-rotational) artificial potential field produces
    directly behind an obstacle that sits between a start point and the
    goal."""
    diff = np.asarray(p, dtype=float) - np.asarray(obstacle_center, dtype=float)
    dist = np.linalg.norm(diff)
    clearance = dist - obstacle_radius
    if clearance <= 1e-6 or clearance >= influence:
        return np.zeros(2)
    r_hat = diff / (dist + 1e-12)
    tangent = np.array([-r_hat[1], r_hat[0]]) * sign
    w = 1.0 - clearance / influence  # 0 at the influence boundary, 1 at the surface
    direction = (1 - swirl_frac * w) * r_hat + (swirl_frac * w) * tangent
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    mag = k_rep * (1.0 / clearance - 1.0 / influence) / (clearance ** 2)
    return min(mag, mag_cap) * direction


def desired_velocity(p, problem, signs, **kwargs):
    att = attractive_velocity(p, problem.target,
                               d_star=kwargs.get("d_star", 3.0),
                               k_att=kwargs.get("k_att", 1.0))
    rep = np.zeros(2)
    for (cx, cy, r), sign in zip(problem.obstacles, signs):
        rep += repulsive_velocity(p, (cx, cy), r, sign,
                                   influence=kwargs.get("influence", 2.5),
                                   k_rep=kwargs.get("k_rep", 6.0),
                                   swirl_frac=kwargs.get("swirl_frac", 0.85))
    return att + rep


def desired_heading(p, v, problem, blend_radius=2.0):
    """Desired end-effector orientation field. Far from the target it just
    faces the direction of travel ("look where you're going"). Near the
    target it blends toward
    whichever *edge* of the [target_theta-tol, target_theta+tol] band is
    closer to the travel heading -- a dead-band correction: only correct as
    much as needed to enter the tolerance, not to hit the exact center. If
    the travel heading is already within tolerance, no correction is
    requested at all, so the field's rotational component genuinely
    vanishes near the goal when the approach is already acceptable (this is
    what gives Stage 2/3 a real place to detect a lower local intrinsic
    dimension, instead of always synthesizing a 3rd generator by
    construction).
    """
    vnorm = np.linalg.norm(v)
    travel_heading = np.arctan2(v[1], v[0]) if vnorm > 1e-9 else problem.target_theta
    err = wrap_angle(problem.target_theta - travel_heading)
    tol = problem.target_theta_tol
    if abs(err) <= tol:
        return travel_heading
    dist = np.linalg.norm(np.asarray(problem.target, dtype=float) - np.asarray(p, dtype=float))
    w = _smoothstep(1.0 - dist / blend_radius)
    band_edge = problem.target_theta - np.sign(err) * tol
    return travel_heading + w * wrap_angle(band_edge - travel_heading)


def conservative_potential(p, problem, d_star=3.0, k_att=1.0, influence=2.5, k_rep=6.0):
    """Scalar potential for the *non-swirled* (purely radial) part of the
    field only -- diagnostic/plotting use. The swirl term in
    `repulsive_velocity` has nonzero curl by design (that's what lets it
    route around the obstacle instead of stalling), so the actual field
    used for trajectory integration is not a pure gradient field and has no
    single exact scalar potential; this conservative approximation is only
    meant to give an intuitive cost-field heatmap."""
    target = np.asarray(problem.target, dtype=float)
    p = np.asarray(p, dtype=float)
    dist = np.linalg.norm(target - p)
    if dist <= d_star:
        u_att = 0.5 * k_att * dist ** 2
    else:
        u_att = k_att * d_star * dist - 0.5 * k_att * d_star ** 2

    u_rep = 0.0
    for cx, cy, r in problem.obstacles:
        diff = p - np.array([cx, cy], dtype=float)
        clearance = np.linalg.norm(diff) - r
        if clearance <= 1e-6:
            u_rep += 0.5 * k_rep * (1.0 / 1e-6 - 1.0 / influence) ** 2
        elif clearance < influence:
            u_rep += 0.5 * k_rep * (1.0 / clearance - 1.0 / influence) ** 2
    return u_att + u_rep
