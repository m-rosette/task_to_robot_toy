"""
Stage 4b: Priors from reach paths.

Each reach path (Stage 2) is simplified into a *segment path*: a polyline
of straight segments that stays in free space. Its bend points are
joint-location priors and its segment lengths are link-length priors.

Simplification is a constrained Douglas-Peucker pass (a shortcut is
accepted only if it stays within `eps` of the reach path *and* its straight
segment clears every obstacle by `link_radius`), followed by removing any
bend whose two neighbors already see each other. The result is a valid
collision-free polyline, so its segment count is an upper estimate of the
minimum-link count m_k from Stage 3.

Bend points are pooled across targets (and tied routes) and clustered with
k-means++ and a silhouette test (moved here from `legacy/stage3_4.py`);
cluster dispersion says how much the targets agree on where to bend.

`design_prior` turns segment paths into a starting design and per-target
starting configurations for any joint combination, and `approach_cones`
records, for each target, the headings inside its pruned band from which
the last link plus probe can come in without collision.
"""

import numpy as np

from problem import wrap_angle


DP_EPS = 0.25          # Douglas-Peucker deviation tolerance (length units)
BEND_CLUSTER_RADIUS_FRAC = 0.05  # max bend-cluster dispersion, as a fraction of length_scale
MIN_LENGTH = 0.05


# --------------------------------------------------------------------------
# Segment paths
# --------------------------------------------------------------------------

def _clear(problem, a, b, infl):
    return bool(problem.segment_clearance(a, b) > infl)


def _point_seg_dist(P, a, b):
    d = b - a
    t = np.clip(((P - a) @ d) / max(d @ d, 1e-18), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * d), axis=1)


def constrained_douglas_peucker(points, problem, eps=DP_EPS, infl=None):
    """Indices of the kept vertices of `points`."""
    infl = problem.link_radius if infl is None else infl

    def rec(i, j):
        if j <= i + 1:
            return [i, j]
        dev = _point_seg_dist(points[i + 1:j], points[i], points[j])
        k = int(np.argmax(dev)) + i + 1
        if dev.max() <= eps and _clear(problem, points[i], points[j], infl):
            return [i, j]
        if dev.max() < 1e-9:
            k = (i + j) // 2
        left = rec(i, k)
        return left[:-1] + rec(k, j)

    return rec(0, len(points) - 1)


def remove_redundant_bends(vertices, problem, infl=None):
    infl = problem.link_radius if infl is None else infl
    V = [np.asarray(v, float) for v in vertices]
    changed = True
    while changed and len(V) > 2:
        changed = False
        # drop the bend with the smallest turn whose neighbors see each other
        order = sorted(range(1, len(V) - 1), key=lambda i: abs(_turn(V[i - 1], V[i], V[i + 1])))
        for i in order:
            if _clear(problem, V[i - 1], V[i + 1], infl):
                del V[i]
                changed = True
                break
    return np.array(V)


def _turn(a, b, c):
    return wrap_angle(np.arctan2(*(c - b)[::-1]) - np.arctan2(*(b - a)[::-1]))


def segment_path(points, problem, eps=DP_EPS):
    idx = constrained_douglas_peucker(points, problem, eps)
    return remove_redundant_bends(points[idx], problem)


def _approximate(V, g):
    """Polyline with exactly g segments: drop the gentlest bends, or split
    the longest segment, until the count matches."""
    V = [np.asarray(v, float) for v in V]
    while len(V) - 1 > g:
        i = min(range(1, len(V) - 1), key=lambda i: abs(_turn(V[i - 1], V[i], V[i + 1])))
        del V[i]
    while len(V) - 1 < g:
        lens = [np.linalg.norm(V[i + 1] - V[i]) for i in range(len(V) - 1)]
        i = int(np.argmax(lens))
        V.insert(i + 1, 0.5 * (V[i] + V[i + 1]))
    return np.array(V)


# --------------------------------------------------------------------------
# Clustering (from legacy/stage3_4.py)
# --------------------------------------------------------------------------

def _kmeans_pp(points, k, n_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    n = len(points)
    centers = np.zeros((k, points.shape[1]))
    centers[0] = points[rng.integers(n)]
    for c in range(1, k):
        d2 = np.min(np.sum((points[:, None, :] - centers[None, :c, :]) ** 2, axis=2), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        centers[c] = points[rng.choice(n, p=probs)]
    assign = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_assign = np.argmin(d, axis=1)
        new_centers = np.array([
            points[new_assign == c].mean(axis=0) if np.any(new_assign == c) else centers[c]
            for c in range(k)
        ])
        if np.array_equal(new_assign, assign) and np.allclose(new_centers, centers):
            assign = new_assign
            centers = new_centers
            break
        assign, centers = new_assign, new_centers
    return centers, assign


def _silhouette_score(points, assign):
    """Mean silhouette coefficient (numpy-only, no sklearn dependency).
    Members of singleton clusters score 0 (the standard convention; the
    legacy copy scored them 1, which always favored k = number of points)."""
    n = len(points)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    labels = np.unique(assign)
    scores = np.zeros(n)
    for i in range(n):
        own = assign[i]
        same = (assign == own)
        same[i] = False
        if not same.any():
            continue
        a = d[i, same].mean()
        b = np.inf
        for lab in labels:
            if lab == own:
                continue
            other = assign == lab
            if other.any():
                b = min(b, d[i, other].mean())
        scores[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(scores.mean())


def cluster_bends(points, radius, k_max=6, seed=0):
    """Cluster 2D bend points with k-means++. The cluster count is the
    smallest k whose clusters all have dispersion <= `radius` (bends on
    opposite sides of an obstacle must not be averaged into its center);
    if no k up to `k_max` achieves that, the k with the best silhouette.
    Returns a list of {center, dispersion, n_members, members}."""
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) == 0:
        return []
    chosen, best = None, (-np.inf, None, None)
    for k in range(1, min(k_max, len(points)) + 1):
        c, a = _kmeans_pp(points, k, seed=seed)
        disp = max(np.mean(np.linalg.norm(points[a == j] - c[j], axis=1)) for j in set(a.tolist()))
        if disp <= radius:
            chosen = (c, a)
            break
        if k >= 2 and len(set(a.tolist())) >= 2:
            s = _silhouette_score(points, a)
            if s > best[0]:
                best = (s, c, a)
    if chosen is None:
        chosen = (best[1], best[2]) if best[1] is not None else (points.mean(axis=0, keepdims=True),
                                                                 np.zeros(len(points), dtype=int))
    centers, assign = chosen
    out = []
    for c in range(len(centers)):
        m = points[assign == c]
        if len(m) == 0:
            continue
        ctr = m.mean(axis=0)
        out.append({"center": ctr, "dispersion": float(np.mean(np.linalg.norm(m - ctr, axis=1))),
                    "n_members": int(len(m)), "members": m})
    return out


# --------------------------------------------------------------------------
# Stage 4b entry point
# --------------------------------------------------------------------------

def approach_cones(problem, target_sets, nominal_last=None):
    """Headings in each pruned band from which the last link (nominal length
    `nominal_last[k]`) plus probe clears every obstacle by the planning
    inflation. Returns per target a list of (lo, hi) intervals."""
    out = []
    for i, ts in enumerate(target_sets):
        if not ts["feasible"]:
            out.append([])
            continue
        Ln = problem.probe_length + (1.0 if nominal_last is None else nominal_last[i])
        hs = ts["samples"][ts["keep"]]
        u = np.stack([np.cos(hs), np.sin(hs)], axis=-1)
        ok = problem.segment_clearance(ts["xy"][None, :] - Ln * u, ts["xy"][None, :]) > problem.inflation
        runs, start = [], None
        for j, k in enumerate(ok):
            if k and start is None:
                start = j
            if (not k or j == len(ok) - 1) and start is not None:
                end = j if k else j - 1
                runs.append((float(hs[start]), float(hs[end])))
                start = None
        out.append(runs)
    return out


def build_priors(problem, target_sets, field, seed=0):
    """Stage 4b: segment paths for every reach path, pooled bend clusters,
    and approach cones."""
    per_target = []
    bends = []
    for ts, tf in zip(target_sets, field["targets"]):
        if tf is None:
            per_target.append(None)
            continue
        routes = []
        for rp in tf["reach_paths"]:
            if not rp["reached"]:
                continue
            V = segment_path(rp["points"], problem)
            lens = np.linalg.norm(np.diff(V, axis=0), axis=1)
            routes.append({"label": rp["label"], "vertices": V, "segment_lengths": lens,
                           "n_segments": len(V) - 1, "bends": V[1:-1]})
            bends.extend(V[1:-1].tolist())
        per_target.append({"index": ts["index"], "routes": routes})
    clusters = cluster_bends(bends, radius=BEND_CLUSTER_RADIUS_FRAC * problem.length_scale, seed=seed)
    nominal_last = []
    for pt in per_target:
        if pt is None or not pt["routes"]:
            nominal_last.append(1.0)
        else:
            last = pt["routes"][0]["segment_lengths"][-1] - problem.probe_length
            nominal_last.append(float(np.clip(last, MIN_LENGTH, 2.0)))
    cones = approach_cones(problem, target_sets, nominal_last)
    return {"per_target": per_target, "bend_points": np.array(bends).reshape(-1, 2),
            "bend_clusters": clusters, "approach_cones": cones}


def chain_groups(combo):
    """Split a joint combination into straight chain segments: a new segment
    starts at every revolute joint after the first joint."""
    groups = [[0]]
    for i in range(1, len(combo)):
        if combo[i] == "R":
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def design_prior(problem, combo, priors, target_sets):
    """Starting design and per-target starting configurations for `combo`
    from the segment paths. Returns (design dict, list of dof0 per target,
    with P joints as stroke fractions in [0, 1])."""
    groups = chain_groups(combo)
    g = len(groups)
    approx = []
    for pt in priors["per_target"]:
        if pt is None or not pt["routes"]:
            approx.append(None)
            continue
        approx.append(_approximate(pt["routes"][0]["vertices"], g))
    seg_lens = np.array([np.linalg.norm(np.diff(V, axis=0), axis=1) for V in approx if V is not None])
    if len(seg_lens) == 0:
        reach = problem.length_scale
        seg_lens = np.full((1, g), reach / g)
    seg_lens = seg_lens.copy()
    seg_lens[:, -1] = np.maximum(seg_lens[:, -1] - problem.probe_length, MIN_LENGTH)
    share = np.median(seg_lens, axis=0)

    n = len(combo)
    lengths, strokes = np.zeros(n), np.zeros(n)
    for gi, grp in enumerate(groups):
        each = share[gi] / len(grp)
        for j in grp:
            if combo[j] == "R":
                lengths[j] = max(each, MIN_LENGTH)
            else:
                lengths[j] = max(0.5 * each, MIN_LENGTH)   # housing
                strokes[j] = max(each, MIN_LENGTH)          # at mid-stroke: h + s/2 = each
    slide = None
    if combo[0] == "P":
        firsts = [np.arctan2(*(V[1] - V[0])[::-1]) for V in approx if V is not None]
        slide = float(np.angle(np.mean(np.exp(1j * np.array(firsts))))) if firsts else 0.0
    design = {"joint_types": combo, "lengths": lengths, "strokes": strokes, "slide_angle": slide}

    dofs = []
    for V in approx:
        if V is None:
            dofs.append(None)
            continue
        phis = [np.arctan2(*(V[i + 1] - V[i])[::-1]) for i in range(g)]
        q = np.zeros(n)
        prev = slide if combo[0] == "P" else problem.base_angle
        for gi, grp in enumerate(groups):
            for j in grp:
                if combo[j] == "R":
                    q[j] = wrap_angle(phis[gi] - prev)
                    prev = phis[gi]
                else:
                    q[j] = 0.5
        dofs.append(q)
    return design, dofs


def serialize_priors(pr):
    return {
        "per_target": [None if pt is None else {
            "index": pt["index"],
            "routes": [{"label": r["label"], "n_segments": r["n_segments"],
                        "vertices": r["vertices"].tolist(),
                        "segment_lengths": r["segment_lengths"].tolist()} for r in pt["routes"]],
        } for pt in pr["per_target"]],
        "bend_clusters": [{"center": c["center"].tolist(), "dispersion": c["dispersion"],
                           "n_members": c["n_members"]} for c in pr["bend_clusters"]],
        "approach_cones_deg": [[[float(np.rad2deg(lo)), float(np.rad2deg(hi))] for lo, hi in cone]
                               for cone in pr["approach_cones"]],
    }
