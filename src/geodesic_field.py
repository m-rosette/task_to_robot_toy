"""
Stage 2: Geodesic twist field.

For each target k, the translational part of the field is the negative
gradient of the geodesic distance d_k to the target. d_k is the length of
the shortest collision-free path. Stage 2 calculates d_k on an occupancy
grid with obstacles inflated by `link_radius + clearance_margin`. The field
has no local minima. Its streamline from the base mount is the reach path,
which is the shortest collision-free path to the target.

The orientation part is a dead-band rule on the geodesic distance. The
heading follows the direction of travel. Within `blend_radius` of the target
(along the free path), the heading turns toward the nearest edge of the
pruned heading band. A smoothstep weight on the distance sets the turn. At
the target, the heading is on the band edge. If the direction of travel is
in the band, the heading does not turn.

Fast marching (`scikit-fmm`) calculates the distances. The obstacles are
circles, so the occupancy comes from the exact clearance at each cell
center.

`field_mode="potential"` keeps the distance field and `geodesic_length`. It
replaces the field direction, the heading rule, and the reach paths with the
legacy potential field.
"""

import dataclasses
import itertools

import numpy as np
import skfmm
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

from problem import wrap_angle
from targets import heading_residual


DISTANCE_BACKEND = "scikit-fmm"


ROUTE_TIE_TOL = 0.05   # keep a tied route if its length is within this fraction of the shortest
ROUTE_DISTINCT = 0.5   # two reach paths are distinct if their Hausdorff distance is more than this


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------

def make_grid(problem, resolution=None):
    """Cell-centered grid over the workspace. Arrays use [iy, ix] indices."""
    res = float(resolution or problem.grid_resolution)
    xmin, ymin, xmax, ymax = problem.bounds
    nx = int(round((xmax - xmin) * res))
    ny = int(round((ymax - ymin) * res))
    h = (xmax - xmin) / nx
    xs = xmin + (np.arange(nx) + 0.5) * h
    ys = ymin + (np.arange(ny) + 0.5) * h
    XX, YY = np.meshgrid(xs, ys)
    centers = np.stack([XX, YY], axis=-1)
    clearance = problem.clearance_field(centers)
    return {"xs": xs, "ys": ys, "XX": XX, "YY": YY, "h": h, "centers": centers,
            "clearance": clearance, "occupied": clearance <= problem.inflation,
            "bounds": problem.bounds}


def cell_of(grid, xy):
    ix = int(np.clip(np.floor((xy[0] - grid["bounds"][0]) / grid["h"]), 0, len(grid["xs"]) - 1))
    iy = int(np.clip(np.floor((xy[1] - grid["bounds"][1]) / grid["h"]), 0, len(grid["ys"]) - 1))
    return iy, ix


def _segment_mask(grid, a, b, half_width):
    """Cells with centers within `half_width` of segment a -> b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    P = grid["centers"]
    t = np.clip(np.sum((P - a) * d, axis=-1) / max(d @ d, 1e-18), 0.0, 1.0)
    closest = a + t[..., None] * d
    return np.linalg.norm(P - closest, axis=-1) <= half_width


def _ray_to_boundary(bounds, p, u):
    """Point where the ray p + s u (s >= 0) leaves the workspace box."""
    xmin, ymin, xmax, ymax = bounds
    s = np.inf
    for lo, hi, pi, ui in ((xmin, xmax, p[0], u[0]), (ymin, ymax, p[1], u[1])):
        if ui > 1e-12:
            s = min(s, (hi - pi) / ui)
        elif ui < -1e-12:
            s = min(s, (lo - pi) / ui)
    return p + s * u


def make_interpolator(field, grid):
    """Bilinear interpolator of `field` that ignores non-finite cells.
    `field` is [iy, ix], or [iy, ix, m] for m components with one shared
    mask. The interpolator divides the interpolated zero-filled field by
    the interpolated finite mask. Thus the weight of a non-finite corner
    goes to the finite corners. Points outside the cell centers are
    clamped. The returned function takes xy (2,) or (N, 2) and returns
    nan where no corner is finite. `ndimage.map_coordinates` keeps the
    call overhead low for the per-point streamline loop."""
    field = np.asarray(field, dtype=float)
    comps = field[..., None] if field.ndim == 2 else field
    ok = np.isfinite(comps).all(axis=-1)
    vals = [np.where(ok, comps[..., i], 0.0) for i in range(comps.shape[-1])]
    weight = ok.astype(float)
    origin = np.array([grid["xs"][0], grid["ys"][0]])
    top = np.array([len(grid["xs"]) - 1, len(grid["ys"]) - 1])

    def f(xy):
        xy = np.asarray(xy, dtype=float)
        idx = np.clip((np.atleast_2d(xy) - origin) / grid["h"], 0, top)[:, ::-1].T   # (2, N) as (iy, ix)
        w = ndimage.map_coordinates(weight, idx, order=1)[:, None]
        num = np.stack([ndimage.map_coordinates(v, idx, order=1) for v in vals], axis=-1)
        out = np.where(w > 1e-12, num / np.maximum(w, 1e-12), np.nan)
        out = out[:, 0] if field.ndim == 2 else out
        return out[0] if xy.ndim == 1 else out

    return f


def interp(field, grid, xy):
    """Bilinear interpolation of a scalar grid at `xy` (see
    `make_interpolator`). Returns a float for one point and an array for
    (N, 2)."""
    v = make_interpolator(field, grid)(xy)
    return float(v) if np.ndim(xy) == 1 else v


# --------------------------------------------------------------------------
# Geodesic distance and its gradient field
# --------------------------------------------------------------------------

def geodesic_distance(grid, target_xy, blocked=None):
    """Geodesic distance from each free cell to `target_xy`. The distance is
    inf in occupied or disconnected cells. `blocked` adds occupied cells,
    for example the walls that force a route around one side of an
    obstacle."""
    occ = grid["occupied"] if blocked is None else (grid["occupied"] | blocked)
    h = grid["h"]
    tx, ty = target_xy
    rho = h
    phi = np.hypot(grid["XX"] - tx, grid["YY"] - ty) - rho
    d = skfmm.distance(np.ma.MaskedArray(phi, occ), dx=h)
    d = np.ma.filled(d.astype(float), np.inf) + rho
    d[occ] = np.inf
    d[~np.isfinite(d) | (d > 1e8)] = np.inf
    # Fast marching gives values to a disconnected free pocket only if the
    # pocket touches the zero contour. Other cells stay inf.
    return d


def _diff(f, h, axis):
    """Derivative along `axis` with central differences where both
    neighbors are finite and one-sided differences next to obstacles."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (1, 1)
    fp = np.pad(f, pad, constant_values=np.inf)
    sl_next = [slice(None), slice(None)]
    sl_prev = [slice(None), slice(None)]
    sl_next[axis] = slice(2, None)
    sl_prev[axis] = slice(None, -2)
    nxt, prv = fp[tuple(sl_next)], fp[tuple(sl_prev)]
    ok_n, ok_p, ok_0 = np.isfinite(nxt), np.isfinite(prv), np.isfinite(f)
    with np.errstate(invalid="ignore"):
        out = np.where(ok_n & ok_p, (nxt - prv) / (2 * h),
              np.where(ok_n, (nxt - f) / h, np.where(ok_p, (f - prv) / h, 0.0)))
    out[~ok_0] = 0.0
    return out


def _wrapped_diff(H, h, axis):
    """Derivative of an angle field, from wrapped angle differences."""
    Hf = np.where(np.isfinite(H), H, np.nan)
    pad = [(0, 0), (0, 0)]
    pad[axis] = (1, 1)
    Hp = np.pad(Hf, pad, constant_values=np.nan)
    sl_next = [slice(None), slice(None)]
    sl_prev = [slice(None), slice(None)]
    sl_next[axis] = slice(2, None)
    sl_prev[axis] = slice(None, -2)
    nxt, prv = Hp[tuple(sl_next)], Hp[tuple(sl_prev)]
    ok_n, ok_p = np.isfinite(nxt), np.isfinite(prv)
    dn = wrap_angle(np.nan_to_num(nxt) - np.nan_to_num(Hf))
    dp = wrap_angle(np.nan_to_num(Hf) - np.nan_to_num(prv))
    out = np.where(ok_n & ok_p, 0.5 * (dn + dp) / h,
          np.where(ok_n, dn / h, np.where(ok_p, dp / h, 0.0)))
    out[~np.isfinite(Hf)] = 0.0
    return out


def _smoothstep(w):
    w = np.clip(w, 0.0, 1.0)
    return w * w * (3 - 2 * w)


def heading_field(vx, vy, dist, intervals, blend_radius):
    """Dead-band orientation rule on the grid (see the module docstring)."""
    travel = np.arctan2(vy, vx)
    out = np.array(travel, dtype=float)
    w = _smoothstep(1.0 - np.where(np.isfinite(dist), dist, np.inf) / blend_radius)
    for idx in np.ndindex(travel.shape):
        if not np.isfinite(dist[idx]):
            out[idx] = np.nan
            continue
        r = heading_residual(travel[idx], intervals)
        if r != 0.0:
            # the nearest band edge is travel - r. Turn toward it by the weight w.
            out[idx] = travel[idx] - w[idx] * r
    return out


def _potential_field_arrays(problem, grid, target):
    """Legacy attractor/repeller/swirl field to one target, on the grid."""
    from legacy.field import bow_signs, desired_heading, desired_velocity
    x, y, th, tol = target
    sub = dataclasses.replace(problem, target=(x, y), target_theta=th, target_theta_tol=tol, targets=None)
    signs = bow_signs(sub)
    vx = np.zeros(grid["XX"].shape)
    vy = np.zeros(grid["XX"].shape)
    H = np.full(grid["XX"].shape, np.nan)
    for idx in np.ndindex(vx.shape):
        if grid["occupied"][idx]:
            continue
        p = grid["centers"][idx]
        v = desired_velocity(p, sub, signs)
        n = np.linalg.norm(v)
        if n > 1e-9:
            vx[idx], vy[idx] = v / n
        H[idx] = desired_heading(p, v, sub)
    return vx, vy, H


# --------------------------------------------------------------------------
# Reach paths
# --------------------------------------------------------------------------

def integrate_reach_path(grid, vx, vy, start, target, step=None, max_len_factor=4.0):
    """RK4 streamline of the unit field (vx, vy) from `start`. It stops
    within one cell of `target`, where the field is zero or undefined, or
    after the maximum step count. Returns (points (N, 2), arc length,
    reached)."""
    h = grid["h"]
    step = step or 0.5 * h
    target = np.asarray(target, float)
    xmin, ymin, xmax, ymax = grid["bounds"]
    max_steps = int(max_len_factor * np.hypot(xmax - xmin, ymax - ymin) / step)

    field = make_interpolator(np.stack([vx, vy], axis=-1), grid)

    def vel(p):
        v = field(p)
        n = np.linalg.norm(v)
        return v / n if np.isfinite(n) and n > 1e-9 else None

    p = np.asarray(start, float).copy()
    pts = [p.copy()]
    reached = False
    for _ in range(max_steps):
        if np.linalg.norm(p - target) <= h:
            reached = True
            break
        k1 = vel(p)
        if k1 is None:
            break
        k2 = vel(p + 0.5 * step * k1)
        k3 = vel(p + 0.5 * step * k2) if k2 is not None else None
        k4 = vel(p + step * k3) if k3 is not None else None
        if k4 is None:
            p = p + step * k1
        else:
            p = p + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        pts.append(p.copy())
    if reached:
        pts.append(target.copy())
    pts = np.array(pts)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return pts, length, reached


def _hausdorff(a, b):
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])


def alternative_routes(problem, grid, target_xy, d_default, path_default, tie_tol=ROUTE_TIE_TOL):
    """Reach paths of the other homotopy classes with a geodesic length
    within `tie_tol` of the shortest. Each of the 2^N_obstacles wall sets
    blocks the left or right side of each obstacle with a thin wall to
    the boundary. A route must be distinct (`ROUTE_DISTINCT`) from the
    default path and from the routes before it."""
    base = np.asarray(problem.base, float)
    target_xy = np.asarray(target_xy, float)
    L0 = interp(d_default, grid, base)
    routes = []
    if not problem.obstacles or not np.isfinite(L0):
        return routes
    line = target_xy - base
    normal = np.array([-line[1], line[0]]) / (np.linalg.norm(line) + 1e-12)
    for signs in itertools.product((1, -1), repeat=len(problem.obstacles)):
        blocked = np.zeros(grid["XX"].shape, dtype=bool)
        for (cx, cy, _r), s in zip(problem.obstacles, signs):
            c = np.array([cx, cy])
            blocked |= _segment_mask(grid, c, _ray_to_boundary(problem.bounds, c, s * normal), grid["h"])
        if blocked[cell_of(grid, base)] or blocked[cell_of(grid, target_xy)]:
            continue
        d = geodesic_distance(grid, target_xy, blocked)
        L = interp(d, grid, base)
        if not np.isfinite(L) or L > (1.0 + tie_tol) * L0:
            continue
        gy, gx = _diff(d, grid["h"], 0), _diff(d, grid["h"], 1)
        n = np.hypot(gx, gy) + 1e-12
        pts, length, reached = integrate_reach_path(grid, -gx / n, -gy / n, base, target_xy)
        if not reached:
            continue
        others = [path_default] + [r["points"] for r in routes]
        if all(_hausdorff(pts, o) > ROUTE_DISTINCT for o in others):
            routes.append({"points": pts, "length": length, "geodesic_length": float(L),
                           "reached": True, "label": "walls" + "".join("+" if s > 0 else "-" for s in signs)})
    return routes


# --------------------------------------------------------------------------
# Stage 2 entry point
# --------------------------------------------------------------------------

def build_geodesic_field(problem, target_sets, field_mode="geodesic", blend_radius=2.0,
                         route_ties=True, tie_tol=ROUTE_TIE_TOL, resolution=None):
    """Stage 2: one field for each feasible target.

    Returns {"grid", "mode", "backend", "blend_radius", "targets"}.
    "targets" has one dict for each target. The dict has the distance
    grid, the unit field (vx, vy), and the heading and heading-rate
    (omega) grids. It also has `geodesic_length` = d_k(base),
    `euclidean_length`, and `reach_paths` (the default route first, then
    the tied routes). An infeasible target (empty pruned band) has None.
    """
    grid = make_grid(problem, resolution)
    base = np.asarray(problem.base, float)
    if grid["occupied"][cell_of(grid, base)]:
        raise ValueError("base mount lies inside an inflated obstacle")
    out = []
    for ts in target_sets:
        if not ts["feasible"]:
            out.append(None)
            continue
        txy = ts["xy"]
        d = geodesic_distance(grid, txy)
        if field_mode == "geodesic":
            gy, gx = _diff(d, grid["h"], 0), _diff(d, grid["h"], 1)
            n = np.hypot(gx, gy)
            with np.errstate(invalid="ignore", divide="ignore"):
                vx, vy = np.where(n > 1e-12, -gx / n, 0.0), np.where(n > 1e-12, -gy / n, 0.0)
            H = heading_field(vx, vy, d, ts["intervals"], blend_radius)
        elif field_mode == "potential":
            vx, vy, H = _potential_field_arrays(problem, grid, (txy[0], txy[1], ts["theta"], ts["tol"]))
        else:
            raise ValueError(f"unknown field mode {field_mode!r}")
        vx[grid["occupied"]] = 0.0
        vy[grid["occupied"]] = 0.0
        omega = vx * _wrapped_diff(H, grid["h"], 1) + vy * _wrapped_diff(H, grid["h"], 0)

        pts, length, reached = integrate_reach_path(grid, vx, vy, base, txy)
        L_geo = interp(d, grid, base)
        paths = [{"points": pts, "length": length, "geodesic_length": float(L_geo),
                  "reached": reached, "label": "default"}]
        if field_mode == "geodesic" and (route_ties or not reached):
            alts = alternative_routes(problem, grid, txy, d, pts, tie_tol)
            if not reached and alts:
                # The base is on the cut locus (an exact tie), so the
                # unconstrained streamline goes into the obstacle. The walled
                # routes resolve the tie. The stalled path stays in the record.
                alts.sort(key=lambda r: r["geodesic_length"])
                paths = alts if route_ties else alts[:1]
                paths[0]["stalled_default"] = {"points": pts, "length": length}
            else:
                paths += alts if route_ties else []
        out.append({"index": ts["index"], "distance": d, "vx": vx, "vy": vy,
                    "heading": H, "omega": omega, "geodesic_length": float(L_geo),
                    "euclidean_length": float(np.linalg.norm(txy - base)),
                    "reach_paths": paths})
    return {"grid": grid, "mode": field_mode, "backend": DISTANCE_BACKEND,
            "blend_radius": blend_radius, "targets": out}
