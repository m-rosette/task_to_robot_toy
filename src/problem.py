"""
Toy problem definition for the "learn task-space twist field -> discover
minimal Lie algebra -> synthesize mechanism" pipeline.

Everything lives in SE(2): a planar end-effector pose is (x, y, theta).

The geodesic, reach-only pipeline (`run_geodesic_pipeline.py`) reads the
multi-target fields (`targets`, `targets_list`) and the fixed physical
parameters (`base_angle`, `probe_length`, `link_radius`, `clearance_margin`,
`grid_resolution`); the legacy pipeline (`legacy/`, `run_pipeline.py`) reads
only the single-target fields, which stay the default.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ToyProblem:
    # Workspace bounds: (xmin, ymin, xmax, ymax)
    bounds: tuple = (0.0, 0.0, 10.0, 10.0)

    # Base location of the future robot (world frame, fixed joint 0 origin)
    base: tuple = (1.0, 1.0)

    # Single target: position + desired end-effector orientation on arrival,
    # with a tolerance band -- any final heading within +-target_theta_tol
    # of target_theta counts as an acceptable approach pose.
    target: tuple = (8.5, 8.0)
    target_theta: float = np.deg2rad(60)
    target_theta_tol: float = np.deg2rad(15)

    # Circular obstacle regions: each (cx, cy, radius). The default is the
    # single-obstacle problem `run_pipeline.py` has been running; the
    # two-obstacle problem documented in SIMPLIFICATIONS.md item 21 is kept
    # as the `TWO_OBSTACLES` preset below.
    obstacles: tuple = ((5.0, 5.0, 1.0),)

    # Characteristic length used to make twists unit-consistent: a twist
    # (v, omega) mixes length/time with 1/time, so every norm, cosine,
    # orthogonalization, and threshold on twists uses the metric
    # ||xi||^2 = |v|^2 / L^2 + omega^2 (see `twist_metric`). None -> the
    # base-to-target distance, i.e. the reach the task demands.
    char_length: float = None

    # Multi-target task: tuple of (x, y, theta, theta_tol). None -> the single
    # target above, so existing presets and the legacy pipeline are unchanged.
    targets: tuple = None

    # Fixed reference direction at the mount (not a design variable). A chain
    # whose first joint is revolute absorbs it into that joint's angle; a
    # chain whose first joint is prismatic ignores it and uses its fitted
    # `slide_angle` instead.
    base_angle: float = 0.0

    # Fixed tool segment appended along the last link. The end-effector point
    # is the probe tip and its heading is the last link's direction. The
    # probe is also the segment used to prune each target's heading band.
    probe_length: float = 0.5

    # Physical half-width of every link (and the probe): a body segment
    # collides when it comes within `link_radius` of an obstacle.
    link_radius: float = 0.1

    # Extra inflation on top of `link_radius` for planning (geodesic field,
    # bounds, tolerance pruning, and the collision term while fitting).
    clearance_margin: float = 0.05

    # Cells per unit length for the geodesic field and the minimum-link grid.
    grid_resolution: float = 20.0

    @property
    def targets_list(self):
        """Targets as a list of (x, y, theta, theta_tol), whichever way the
        problem was specified."""
        if self.targets is None:
            return [(float(self.target[0]), float(self.target[1]),
                     float(self.target_theta), float(self.target_theta_tol))]
        return [tuple(float(v) for v in t) for t in self.targets]

    @property
    def inflation(self):
        """Obstacle inflation used for planning: link radius + margin."""
        return float(self.link_radius + self.clearance_margin)

    @property
    def length_scale(self):
        """Characteristic length: `char_length` if set, else the farthest
        base-to-target distance (the reach the task demands)."""
        if self.char_length is not None:
            return float(self.char_length)
        return float(max(np.hypot(x - self.base[0], y - self.base[1])
                         for x, y, _, _ in self.targets_list))

    def in_bounds(self, xy):
        xmin, ymin, xmax, ymax = self.bounds
        x, y = xy
        return xmin <= x <= xmax and ymin <= y <= ymax

    def obstacle_clearance(self, xy):
        """Signed distance to the nearest obstacle's surface; negative =
        inside some obstacle."""
        x, y = xy
        if not self.obstacles:
            return np.inf
        return min(np.hypot(x - cx, y - cy) - r for cx, cy, r in self.obstacles)

    def clearance_field(self, pts):
        """Vectorized `obstacle_clearance` over points (..., 2)."""
        pts = np.asarray(pts, dtype=float)
        out = np.full(pts.shape[:-1], np.inf)
        for cx, cy, r in self.obstacles:
            out = np.minimum(out, np.hypot(pts[..., 0] - cx, pts[..., 1] - cy) - r)
        return out

    def segment_clearance(self, a, b):
        """Exact signed clearance of straight segment(s) a -> b (arrays
        (..., 2), broadcast against each other) from the circular
        obstacles: distance from each obstacle center to the closest point
        on the segment, minus the radius, minimized over obstacles."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a, b = np.broadcast_arrays(a, b)
        d = b - a
        dd = np.maximum(np.sum(d * d, axis=-1), 1e-18)
        out = np.full(a.shape[:-1], np.inf)
        for cx, cy, r in self.obstacles:
            c = np.array([cx, cy])
            t = np.clip(np.sum((c - a) * d, axis=-1) / dd, 0.0, 1.0)
            closest = a + t[..., None] * d
            out = np.minimum(out, np.linalg.norm(closest - c, axis=-1) - r)
        return out


TWO_OBSTACLES = dict(obstacles=((5.0, 5.2, 1.4), (7.0, 3.0, 1.0)))

# Targets on an arc around the far (lower-right) side of the single
# obstacle, two of them behind it and one close beside it. Headings follow the direction in which a
# collision-free path around that side arrives. The arc stays on one side
# because transitions are straight lines in joint space: an arm anchored at
# the base cannot swing its links across the obstacle between targets on
# opposite sides without a retracting via-configuration, which the toy does
# not model.
_RING_TARGETS = (
    (8.5, 8.0, 60.0),
    (7.0, 7.0, 75.0),
    (6.5, 8.5, 90.0),
    (9.0, 6.5, 45.0),
    (8.5, 5.0, 30.0),
    (7.5, 3.8, 20.0),
    (8.0, 2.5, 5.0),
    (6.4, 5.6, 65.0),   # close to the obstacle: part of its band is pruned
)


def _ring(tol_deg, n=None):
    return tuple((x, y, np.deg2rad(th), np.deg2rad(tol_deg)) for x, y, th in _RING_TARGETS[:n])


MULTI_TARGET_WIDE = dict(targets=_ring(40.0))
# Exact headings on the first seven ring targets. The eighth exists to show
# partial band pruning, which an exact heading cannot (it is kept or not),
# and with its heading fixed no chain of up to three joints reaches all
# eight targets collision-free.
MULTI_TARGET_EXACT = dict(targets=_ring(0.0, n=7))

# Parallel-branch gantry analogue: every target is approached along the
# same heading (+y), and nothing obstructs the straight reach from the base.
PARALLEL_BRANCHES = dict(
    obstacles=(),
    targets=tuple((x, y, np.deg2rad(90.0), np.deg2rad(10.0))
                  for x, y in ((3.0, 6.0), (5.0, 7.5), (7.0, 5.5), (8.5, 8.0))),
)

PRESETS = {
    "one_obstacle": {},
    "multi_target_wide": MULTI_TARGET_WIDE,
    "multi_target_exact": MULTI_TARGET_EXACT,
    "parallel_branches": PARALLEL_BRANCHES,
    "two_obstacles": TWO_OBSTACLES,
}

# Presets the legacy single-target pipeline (`run_pipeline.py`) supports.
LEGACY_PRESETS = {k: PRESETS[k] for k in ("one_obstacle", "two_obstacles")}


# --------------------------------------------------------------------------
# Unit-consistent twist metric
# --------------------------------------------------------------------------
#
# Twists throughout this pipeline are *hybrid* (a.k.a. coordinate) twists
# (vx, vy, omega): the world-frame velocity of the end-effector point plus
# the heading rate, i.e. exactly (xdot, ydot, thetadot) in SE(2)
# coordinates. Their Euclidean norm depends on the length unit (meters vs.
# millimeters changes which component dominates), so every comparison
# between twists goes through this diagonal metric instead.

def twist_metric(L):
    """Diagonal weights W with ||xi||_W^2 = xi^T W xi = |v|^2/L^2 + omega^2."""
    return np.array([1.0 / L ** 2, 1.0 / L ** 2, 1.0])


def to_dimensionless(xi, L):
    """Map hybrid twist(s) (..., 3) to dimensionless form (v/L, omega), in
    which the plain Euclidean inner product equals the W-metric."""
    xi = np.asarray(xi, dtype=float)
    return xi * np.array([1.0 / L, 1.0 / L, 1.0])


def twist_norm(xi, L):
    return float(np.linalg.norm(to_dimensionless(xi, L)))


def twist_normalize(xi, L):
    """Unit twist under the W-metric (returned in physical units)."""
    return np.asarray(xi, dtype=float) / (twist_norm(xi, L) + 1e-12)


def twist_cos(a, b, L):
    da, db = to_dimensionless(a, L), to_dimensionless(b, L)
    return float(da @ db / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-12))


def span_residual(basis, xi, L):
    """Relative W-metric residual of `xi` after least-squares projection onto
    span(basis columns): 0 = in the span, 1 = orthogonal to it. `basis` is
    (3, k) in physical units (e.g. a Jacobian, or stacked generators)."""
    Bd = to_dimensionless(np.asarray(basis, dtype=float).T, L).T
    xd = to_dimensionless(xi, L)
    nx = np.linalg.norm(xd)
    if nx < 1e-12:
        return 0.0
    if Bd.size == 0:
        return 1.0
    c, *_ = np.linalg.lstsq(Bd, xd, rcond=None)
    return float(np.linalg.norm(xd - Bd @ c) / nx)


# --------------------------------------------------------------------------
# SE(2) helpers
# --------------------------------------------------------------------------

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def se2_log(dtheta, dx, dy):
    """Closed-form log map of SE(2): relative transform (dtheta,dx,dy)
    expressed in the *body* frame -> body twist (vx, vy, omega)."""
    if abs(dtheta) < 1e-8:
        vx, vy = dx, dy
    else:
        A = np.sin(dtheta) / dtheta
        B = (1 - np.cos(dtheta)) / dtheta
        det = A * A + B * B
        vx = (A * dx + B * dy) / det
        vy = (-B * dx + A * dy) / det
    return np.array([vx, vy, dtheta])


def relative_body_twist(pose1, pose2, dt):
    """Body-frame twist that carries pose1 -> pose2 in time dt, computed via
    the SE(2) logarithm of T1^{-1} T2 (this is the 'Lie group logarithm'
    finite-difference twist estimator referenced in Stage 1)."""
    x1, y1, th1 = pose1
    x2, y2, th2 = pose2
    dx_w, dy_w = x2 - x1, y2 - y1
    c, s = np.cos(th1), np.sin(th1)
    dx = c * dx_w + s * dy_w
    dy = -s * dx_w + c * dy_w
    dtheta = wrap_angle(th2 - th1)
    twist = se2_log(dtheta, dx, dy)
    return twist / dt


def se2_bracket(xi1, xi2):
    """Lie bracket in se(2) of *spatial* (or body) twists, i.e. of the
    corresponding right- (left-) invariant vector fields. Elements are
    (vx, vy, omega).

    NOT valid for the hybrid twists used elsewhere in this pipeline (EE-point
    velocity + omega): those are coordinate velocities, and two constant
    hybrid fields -- e.g. "translate along x" and "spin about the EE point"
    -- have commuting flows although this formula returns a nonzero value
    for them. Stage 3 therefore uses the coordinate bracket
    (`stage3_4._vector_field_bracket`); this function is kept as a correct
    utility for spatial twists only.

    se(2) = so(2) (+) R^2 (semidirect product), so(2) acts on R^2 by the
    90-degree rotation J = [[0,-1],[1,0]]. Result:
        [xi1, xi2] = (omega1 * J v2 - omega2 * J v1, 0)
    i.e. translations commute with each other; rotation and translation
    generally do not commute.
    """
    v1 = np.array(xi1[:2]); w1 = xi1[2]
    v2 = np.array(xi2[:2]); w2 = xi2[2]
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    v_out = w1 * (J @ v2) - w2 * (J @ v1)
    return np.array([v_out[0], v_out[1], 0.0])
