"""
Stage 4b: Priors from reach paths.

Stage 4b simplifies each reach path (Stage 2) into a segment path: a
polyline of straight segments in free space. Its bend points are
joint-location priors. Its segment lengths are link-length priors.

The simplification is a constrained Douglas-Peucker pass, then the removal
of each bend whose two neighbors see each other. The pass accepts a shortcut
only if the shortcut stays within `eps` of the reach path and clears all
obstacles by `link_radius`. The result is a collision-free polyline. Thus
its segment count is an upper estimate of the minimum-link count m_k from
Stage 3.

Stage 4b pools the bend points of all targets and tied routes. It clusters
them with scikit-learn k-means++ and the silhouette score. The cluster
dispersion shows how much the targets agree on the bend locations.

`design_prior` makes a start design and per-target start configurations for
a joint combination from the segment paths. `approach_cones` gives, for each
target, the headings in its pruned band at which the last link and the probe
are clear of obstacles.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from problem import wrap_angle
from targets import intervals_of


DP_EPS = 0.25          # Douglas-Peucker deviation tolerance (length units)
BEND_CLUSTER_RADIUS_FRAC = 0.05  # maximum bend-cluster dispersion, as a fraction of length_scale
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
        # remove the bend with the smallest turn whose neighbors see each other
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
    """Polyline with exactly g segments. It removes the smallest bends, or
    divides the longest segment, until the count is g."""
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
# Clustering
# --------------------------------------------------------------------------

def cluster_bends(points, radius, k_max=6, seed=0):
    """Clusters 2D bend points with k-means++. The cluster count is the
    smallest k at which the dispersion of each cluster is at most
    `radius`. Thus bends on opposite sides of an obstacle do not merge
    into its center. If no k up to `k_max` meets this condition, the
    cluster count is the k with the best silhouette. Members of singleton
    clusters score 0. Returns a list of {center, dispersion, n_members,
    members}."""
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) == 0:
        return []
    chosen, best = None, (-np.inf, None, None)
    for k in range(1, min(k_max, len(np.unique(points, axis=0))) + 1):
        km = KMeans(n_clusters=k, init="k-means++", random_state=seed).fit(points)
        c, a = km.cluster_centers_, km.labels_
        disp = max(np.mean(np.linalg.norm(points[a == j] - c[j], axis=1)) for j in set(a.tolist()))
        if disp <= radius:
            chosen = (c, a)
            break
        n_labels = len(set(a.tolist()))
        if k >= 2 and n_labels >= 2:
            # silhouette_score needs a non-singleton cluster. All singletons score 0.
            s = silhouette_score(points, a) if n_labels < len(points) else 0.0
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
    """Headings in each pruned band at which the last link (nominal length
    `nominal_last[k]`) and the probe clear all obstacles by the planning
    inflation. Returns a list of (lo, hi) intervals for each target."""
    out = []
    for i, ts in enumerate(target_sets):
        if not ts["feasible"]:
            out.append([])
            continue
        Ln = problem.probe_length + (1.0 if nominal_last is None else nominal_last[i])
        hs = ts["samples"][ts["keep"]]
        u = np.stack([np.cos(hs), np.sin(hs)], axis=-1)
        ok = problem.segment_clearance(ts["xy"][None, :] - Ln * u, ts["xy"][None, :]) > problem.inflation
        out.append(intervals_of(hs, ok))
    return out


def build_priors(problem, target_sets, field, seed=0):
    """Stage 4b: segment paths for each reach path, pooled bend clusters,
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
    """Divides a joint combination into straight chain segments. A new
    segment starts at each revolute joint after the first joint."""
    groups = [[0]]
    for i in range(1, len(combo)):
        if combo[i] == "R":
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def design_prior(problem, combo, priors, target_sets):
    """Start design and per-target start configurations for `combo` from the
    segment paths. Returns (design dict, list of dof0 for each target). P
    joints use stroke fractions in [0, 1]."""
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
