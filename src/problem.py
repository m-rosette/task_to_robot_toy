"""
Toy problem definition for the "learn task-space twist field -> discover
minimal Lie algebra -> synthesize mechanism" pipeline.

Everything lives in SE(2): a planar end-effector pose is (x, y, theta).
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

    @property
    def length_scale(self):
        if self.char_length is not None:
            return float(self.char_length)
        return float(np.hypot(self.target[0] - self.base[0], self.target[1] - self.base[1]))

    def in_bounds(self, xy):
        xmin, ymin, xmax, ymax = self.bounds
        x, y = xy
        return xmin <= x <= xmax and ymin <= y <= ymax

    def obstacle_clearance(self, xy):
        """Signed distance to the nearest obstacle's surface; negative =
        inside some obstacle."""
        x, y = xy
        return min(np.hypot(x - cx, y - cy) - r for cx, cy, r in self.obstacles)


TWO_OBSTACLES = dict(obstacles=((5.0, 5.2, 1.4), (7.0, 3.0, 1.0)))
PRESETS = {"one_obstacle": {}, "two_obstacles": TWO_OBSTACLES}


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
