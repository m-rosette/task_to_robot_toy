"""
Stage 3: Lower bounds on the mechanism.

Three bounds that count different things and are reported separately:

1. Counting bound (total joints). A chain of n joints has at most n + 1
   shared design values (one length or stroke per joint, plus the slide
   direction of a leading prismatic joint) and n joint values per target.
   Reaching K targets that fix c_k coordinates each needs
   (n + 1) + n*K >= sum_k c_k. Taking the largest d = n + 1 keeps the bound
   valid for every combination of that size.

2. Minimum-link bound (rotating joints). Links are straight and a prismatic
   joint lengthens a segment without bending it, so the arm at a target
   configuration is a polyline with one segment per rotating joint after
   the first, plus one. If the fewest collision-free straight segments from
   the base to target k is m_k, the arm needs at least m - 1 revolute
   joints after its first joint, m = max_k m_k. (A revolute joint at the
   base aims the first segment; it does not bend the chain.) m_k is found
   with visibility levels on a grid (see `minimum_link_counts`).

3. Length bound. The arm, probe included, is a curve from the base to each
   target in free space, so its total length at full extension is at least
   max_k L_min,k with L_min,k the geodesic distance from the base to target
   k.

The bounds use obstacles inflated by `link_radius` only (the hard collision
constraint), not by the extra planning margin, so they stay lower bounds.
The minimum-link count is approximate: intermediate bend points are limited
to cell centers and frontier cells are subsampled, both of which can only
overstate m_k. Its sensitivity to grid resolution is an ablation (E5).
"""

import dataclasses

import numpy as np

from geodesic_field import geodesic_distance, interp, make_grid


MAX_LINK_LEVELS = 6
FRONTIER_SAMPLES = 300


def counting_bound(c_values, n_max=10):
    """Smallest n with (n + 1) + n*K >= sum(c_k), over feasible targets."""
    c_values = [c for c in c_values if c is not None]
    K = len(c_values)
    total = int(sum(c_values))
    for n in range(1, n_max + 1):
        if (n + 1) + n * K >= total:
            return n
    return None


def _visible_from(problem, sources, cells, inflation, chunk=20):
    """(len(sources), len(cells)) boolean: straight segment source -> cell
    clears every obstacle by more than `inflation`."""
    sources = np.atleast_2d(sources)
    out = np.zeros((len(sources), len(cells)), dtype=bool)
    for s in range(0, len(sources), chunk):
        src = sources[s:s + chunk, None, :]
        out[s:s + chunk] = problem.segment_clearance(src, cells[None, :, :]) > inflation
    return out


def _frontier(level_mask_grid, free_grid):
    """Cells in the current level with a free 4-neighbor outside it."""
    m = level_mask_grid
    outside = free_grid & ~m
    nb = np.zeros_like(m)
    nb[1:, :] |= outside[:-1, :]
    nb[:-1, :] |= outside[1:, :]
    nb[:, 1:] |= outside[:, :-1]
    nb[:, :-1] |= outside[:, 1:]
    return m & nb


def minimum_link_counts(problem, target_xys, resolution=None, inflation=None,
                        max_levels=MAX_LINK_LEVELS, frontier_samples=FRONTIER_SAMPLES):
    """Fewest collision-free straight segments from the base to each target.

    Visibility levels grow from the base: V1 = free cells visible from the
    base, V(i+1) = V(i) plus the cells visible from (a deterministic sample
    of) V(i)'s frontier. Each target is matched from its own side (meet in
    the middle): with W = free cells visible from the target, m_k = 1 if the
    target is visible from the base, else the first i + 1 with
    V(i) & W nonempty.

    Returns (m_k list, info) where info holds the level map (first level of
    every free cell, 0 = beyond `max_levels`) and the grid used.
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
    # finish the level map for plotting
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

    # Length bound on a grid inflated by link_radius only (the hard
    # constraint), less half a cell diagonal: a cell then counts as free
    # whenever any point of it is truly free, so the grid's free space
    # contains the true one and its geodesic distance does not overstate
    # the true shortest length.
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
