"""
Toy problem definition for the "learn task-space twist field -> discover
minimal Lie algebra -> synthesize mechanism" pipeline.

Everything lives in SE(2): a planar end-effector pose is (x, y, theta).

`run_geodesic_pipeline.py` reads the multi-target fields (`targets`,
`targets_list`) and the physical parameters (`base_angle`, `probe_length`,
`link_radius`, `clearance_margin`, `grid_resolution`). `run_pipeline.py`
(`legacy/`) reads only the single-target fields. The single-target fields
are the default.
"""

import numpy as np
from dataclasses import dataclass


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
    # ||xi||^2 = |v|^2 / L^2 + omega^2 (see `to_dimensionless`). None -> the
    # base-to-target distance, i.e. the reach the task demands.
    char_length: float = None

    # Multi-target task: tuple of (x, y, theta, theta_tol). None -> the
    # single target above.
    targets: tuple = None

    # Fixed reference heading at the mount (not a design variable). With a
    # revolute first joint, the first link heading is base_angle + q[0].
    # A prismatic first joint uses its fitted `slide_angle` instead.
    base_angle: float = 0.0

    # Fixed tool segment along the last link. The end-effector point is the
    # probe tip. The end-effector heading is the last link direction. Stage 1
    # uses the probe to prune the heading band of each target.
    probe_length: float = 0.5

    # Half-width of each link and of the probe. A body segment collides when
    # its clearance from an obstacle is less than `link_radius`.
    link_radius: float = 0.1

    # Inflation added to `link_radius` for planning: the geodesic field,
    # tolerance pruning, approach cones, and the collision residual of the fit.
    clearance_margin: float = 0.05

    # Cells per unit length of the geodesic field grid. The minimum-link grid
    # uses half this value.
    grid_resolution: float = 20.0

    @property
    def targets_list(self):
        """Targets as a list of (x, y, theta, theta_tol), from `targets` or
        from the single-target fields."""
        if self.targets is None:
            return [(float(self.target[0]), float(self.target[1]),
                     float(self.target_theta), float(self.target_theta_tol))]
        return [tuple(float(v) for v in t) for t in self.targets]

    @property
    def inflation(self):
        """Planning obstacle inflation: `link_radius` + `clearance_margin`."""
        return float(self.link_radius + self.clearance_margin)

    @property
    def length_scale(self):
        """Characteristic length: `char_length` if set, else the largest
        base-to-target distance."""
        if self.char_length is not None:
            return float(self.char_length)
        return float(max(np.hypot(x - self.base[0], y - self.base[1])
                         for x, y, _, _ in self.targets_list))

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
        """Signed clearance of straight segments a -> b from the circular
        obstacles. `a` and `b` are arrays (..., 2) that broadcast. The
        clearance is the minimum, over the obstacles, of the
        center-to-segment distance minus the radius."""
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

# Targets on an arc on the lower-right side of the single obstacle. Two
# targets are behind the obstacle and one is near it. Each heading is the
# arrival direction of a collision-free path on that side. All targets are on
# one side because transitions are straight lines in joint space. The toy has
# no via-configurations, so an arm cannot move its links across the obstacle.
_RING_TARGETS = (
    (8.5, 8.0, 60.0),
    (7.0, 7.0, 75.0),
    (6.5, 8.5, 90.0),
    (9.0, 6.5, 45.0),
    (8.5, 5.0, 30.0),
    (7.5, 3.8, 20.0),
    (8.0, 2.5, 5.0),
    (6.4, 5.6, 65.0),   # near the obstacle: Stage 1 removes part of its band
)


def _ring(tol_deg, n=None):
    return tuple((x, y, np.deg2rad(th), np.deg2rad(tol_deg)) for x, y, th in _RING_TARGETS[:n])


MULTI_TARGET_WIDE = dict(targets=_ring(40.0))
# Exact headings on the first seven ring targets. With an exact heading on
# all eight targets, no chain of up to three joints is collision-free.
MULTI_TARGET_EXACT = dict(targets=_ring(0.0, n=7))

# Parallel-branch gantry analogue: all targets have the same heading (+y),
# and no obstacle blocks the straight line from the base.
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

# Presets that `run_pipeline.py` (single target) supports.
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
# Angles
# --------------------------------------------------------------------------

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
