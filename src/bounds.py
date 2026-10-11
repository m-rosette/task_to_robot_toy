"""
Stage 3: Lower bounds on the mechanism.

Stage 3 reports three separate bounds:

1. Counting bound (total joints). A chain of n joints has n + 1 or fewer
   shared design values. These are one length or stroke for each joint and
   the slide direction of a prismatic first joint. The chain has n joint
   values for each target. K targets that fix c_k coordinates each need
   (n + 1) + n*K >= sum_k c_k. The bound uses the maximum design-value
   count, so it is valid for all combinations of n joints.

2. Minimum-link bound (revolute joints). Links are straight, and a prismatic
   joint makes a segment longer without a bend. Thus at a target
   configuration, the arm is a polyline with one segment for each revolute
   joint after the first, plus one. Let m_k be the minimum number of
   collision-free straight segments from the base to target k, and
   m = max_k m_k. Then the arm needs at least m - 1 revolute joints after
   its first joint. A revolute joint at the base sets the direction of the
   first segment and does not bend the chain. `minimum_link_counts` finds
   m_k with visibility levels on a grid.

3. Length bound. The arm, with the probe, is a curve in free space from the
   base to each target. Thus its total length at full extension is at least
   max_k L_min,k, where L_min,k is the geodesic distance from the base to
   target k.

The bounds inflate the obstacles by `link_radius` only (the hard collision
constraint), not by `clearance_margin`. Thus they stay lower bounds. The
minimum-link count is approximate: bend points are cell centers, and the
frontier cells are subsampled. Both approximations can only make m_k larger.
"""

import dataclasses

import numpy as np
from scipy import ndimage

from geodesic_field import geodesic_distance, interp, make_grid


MAX_LINK_LEVELS = 6
FRONTIER_SAMPLES = 300
FOUR_NEIGHBORS = ndimage.generate_binary_structure(2, 1)


def counting_bound(c_values, n_max=10):
    """Smallest n with (n + 1) + n*K >= sum(c_k) over the feasible targets."""
    c_values = [c for c in c_values if c is not None]
    K = len(c_values)
    total = int(sum(c_values))
    for n in range(1, n_max + 1):
        if (n + 1) + n * K >= total:
            return n
    return None


def _visible_from(problem, sources, cells, inflation, chunk=20):
    """(len(sources), len(cells)) boolean: True where the straight segment
    source -> cell has a clearance more than `inflation`."""
    sources = np.atleast_2d(sources)
    out = np.zeros((len(sources), len(cells)), dtype=bool)
    for s in range(0, len(sources), chunk):
        src = sources[s:s + chunk, None, :]
        out[s:s + chunk] = problem.segment_clearance(src, cells[None, :, :]) > inflation
    return out


def _frontier(level_mask_grid, free_grid):
    """Cells in the current level with a free 4-neighbor outside it."""
    m = level_mask_grid
    return m & ndimage.binary_dilation(free_grid & ~m, structure=FOUR_NEIGHBORS)


def minimum_link_counts(problem, target_xys, resolution=None, inflation=None,
                        max_levels=MAX_LINK_LEVELS, frontier_samples=FRONTIER_SAMPLES):
    """Minimum number of collision-free straight segments from the base to
    each target.

    Visibility levels grow from the base. V1 is the set of free cells
    visible from the base. V(i+1) is V(i) plus the cells visible from a
    deterministic sample of the frontier of V(i). Each target also has W,
    the set of free cells visible from the target (meet in the middle).
    m_k = 1 if the base sees the target. Otherwise, m_k is the first
    i + 1 where V(i) & W is not empty.

    Returns (m_k list, info). info has the level map (first level of each
    free cell, 0 = more than `max_levels`) and the grid.
    """
    infl = problem.link_radius if inflation is None else inflation
    grid = make_grid(problem, resolution or problem.grid_resolution / 2)
    free = grid["clearance"] > infl
    cells = grid["centers"][free]
    base = np.asarray(problem.base, float)

    levels = [_visible_from(problem, base, cells, infl)[0]]
    level_map = np.zeros(free.shape, dtype=int)
    level_map[free] = np.where(levels[0], 1, 0)

    def grow():
        cur = levels[-1]
        cur_grid = np.zeros(free.shape, dtype=bool)
        cur_grid[free] = cur
        fr = _frontier(cur_grid, free)[free]
        idx = np.flatnonzero(fr)
        if len(idx) == 0:
            return False
        if len(idx) > frontier_samples:
            idx = idx[np.linspace(0, len(idx) - 1, frontier_samples).round().astype(int)]
        nxt = cur | _visible_from(problem, cells[idx], cells, infl).any(axis=0)
        newly = nxt & ~cur
        tmp = level_map[free]
        tmp[newly] = len(levels) + 1
        level_map[free] = tmp
        levels.append(nxt)
        return bool(newly.any())

    m_k = []
    for txy in target_xys:
        txy = np.asarray(txy, float)
        if problem.segment_clearance(base, txy) > infl:
            m_k.append(1)
            continue
        W = _visible_from(problem, txy, cells, infl)[0]
        m = None
        i = 0
        while i < max_levels - 1:
            if i >= len(levels) and not grow():
                break
            if np.any(levels[i] & W):
                m = i + 2
                break
            i += 1
        m_k.append(m)
    # complete the level map for the plot
    while len(levels) < max_levels and grow():
        pass
    return m_k, {"level_map": level_map, "free": free, "grid": grid,
                 "inflation": float(infl), "resolution": float(1.0 / grid["h"])}


def compute_bounds(problem, target_sets, link_resolution=None):
    """Stage 3: all three bounds for the feasible targets."""
    feas = [ts for ts in target_sets if ts["feasible"]]
    c_values = [ts["c_k"] for ts in feas]
    n_count = counting_bound(c_values)

    m_k, link_info = minimum_link_counts(problem, [ts["xy"] for ts in feas], resolution=link_resolution)
    known = [m for m in m_k if m is not None]
    m = max(known) if known else None
    rotating_floor = None if m is None else max(m - 1, 0)

    # The length bound uses a grid inflated by link_radius (the hard
    # constraint) minus half a cell diagonal. A cell is free if any point in
    # it is free. Thus the grid free space contains the true free space, and
    # the grid geodesic distance is not more than the true shortest length.
    h = 1.0 / problem.grid_resolution
    hard = dataclasses.replace(problem, clearance_margin=-np.sqrt(0.5) * h)
    grid = make_grid(hard)
    base = np.asarray(problem.base, float)
    L_min = [float(interp(geodesic_distance(grid, ts["xy"]), grid, base)) for ts in feas]
    euclid = [float(np.linalg.norm(ts["xy"] - base)) for ts in feas]

    return {
        "counting": {"bound": n_count, "K": len(feas), "c_k": c_values, "sum_c": int(sum(c_values)),
                     "max_shared_values": None if n_count is None else n_count + 1},
        "min_link": {"m_k": m_k, "m": m, "rotating_floor": rotating_floor,
                     "resolution": link_info["resolution"], "inflation": link_info["inflation"],
                     "unresolved_targets": [ts["index"] for ts, mk in zip(feas, m_k) if mk is None]},
        "length": {"L_min_k": L_min, "bound": max(L_min) if L_min else None,
                   "euclidean_k": euclid, "inflation": float(hard.inflation)},
        "target_indices": [ts["index"] for ts in feas],
        "_link_info": link_info,
    }


def serialize_bounds(b):
    return {k: v for k, v in b.items() if not k.startswith("_")}
