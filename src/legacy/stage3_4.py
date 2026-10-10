"""
Stage 3: Infer the Underlying Lie Algebra
Stage 4: Identify Kinematic Motion Primitives

Twist convention (SIMPLIFICATIONS.md Phase 13): every twist here is a
*hybrid* twist (vx, vy, omega) -- the world-frame velocity of the
end-effector point plus the heading rate -- which is exactly the coordinate
velocity (xdot, ydot, thetadot) on SE(2). Generator fields are therefore
plain vector fields in (x, y, theta) coordinates, and their Lie bracket is
the coordinate bracket, with no se(2) commutator term (see
`_vector_field_bracket`). All comparisons between twists use the
unit-consistent metric from `problem.twist_metric`.
"""

import numpy as np
from problem import twist_cos, twist_normalize, span_residual, to_dimensionless


# A generator counts as prismatic when its instantaneous center of rotation
# is farther than this many characteristic lengths away (radius of
# curvature |v|/|omega| > factor * L). Replaces the old unit-dependent
# threshold on the omega component of a Euclidean-normalized twist.
PRISMATIC_RADIUS_FACTOR = 2.0

# A bracket is "closes_in_span" if its relative residual outside the local
# generator span (W-metric) is below this.
SPAN_RESIDUAL_TOL = 0.1


def _match_generator(g, other_generators, L, min_cos=0.3):
    """Match generator `g` to the closest-direction generator in a
    neighboring cell (mod sign, since generators have no canonical sign or
    ordering across cells), using the unit-consistent twist cosine. Returns
    None if nothing lines up well enough to treat as "the same generator"
    there."""
    if other_generators is None or len(other_generators) == 0:
        return None
    sims = np.array([twist_cos(o, g, L) for o in other_generators])
    idx = int(np.argmax(np.abs(sims)))
    if abs(sims[idx]) < min_cos:
        return None
    vec = other_generators[idx]
    return vec if sims[idx] >= 0 else -vec


def _spatial_jacobian(key, gi, stage2_results, cell_w, cell_h, L):
    """Finite-difference spatial Jacobian d(generator)/d(x,y) (a 3x2 matrix)
    of generator index `gi`'s vector field at `key`, using whichever
    neighboring grid points have data (central difference if both sides are
    present, one-sided otherwise). Returns None if no neighbor has usable
    data in either axis. Returns (J, central): `central` is True only if
    both axes used central differences (O(h^2) error instead of O(h)).

    The fields are defined as functions of position only, i.e. extended to
    SE(2) independently of theta, so d/dtheta = 0 and this 3x2 matrix is the
    full coordinate derivative."""
    i, j = key
    g0 = stage2_results[key]["generators"][gi]

    def gen_at(nk):
        nb = stage2_results.get(nk)
        return _match_generator(g0, nb["generators"], L) if nb is not None else None

    gxp, gxm = gen_at((i + 1, j)), gen_at((i - 1, j))
    gyp, gym = gen_at((i, j + 1)), gen_at((i, j - 1))

    if gxp is not None and gxm is not None:
        dgdx = (gxp - gxm) / (2 * cell_w)
    elif gxp is not None:
        dgdx = (gxp - g0) / cell_w
    elif gxm is not None:
        dgdx = (g0 - gxm) / cell_w
    else:
        dgdx = None

    if gyp is not None and gym is not None:
        dgdy = (gyp - gym) / (2 * cell_h)
    elif gyp is not None:
        dgdy = (gyp - g0) / cell_h
    elif gym is not None:
        dgdy = (g0 - gym) / cell_h
    else:
        dgdy = None

    if dgdx is None and dgdy is None:
        return None, False
    central = all(g is not None for g in (gxp, gxm, gyp, gym))
    dgdx = np.zeros(3) if dgdx is None else dgdx
    dgdy = np.zeros(3) if dgdy is None else dgdy
    return np.stack([dgdx, dgdy], axis=1), central


def _vector_field_bracket(key, gi, gj, stage2_results, cell_w, cell_h, L=1.0):
    """Lie bracket of generator fields gi, gj at `key`, as vector fields in
    SE(2) coordinates (x, y, theta):

        [X, Y] = DY . X - DX . Y

    where DY is the spatial Jacobian of Y's field (theta-derivative zero,
    see `_spatial_jacobian`) contracted with X's (vx, vy) components.

    There is deliberately NO se(2) commutator term. Hybrid twists are
    coordinate velocities, and for coordinate vector fields the bracket is
    the transport part alone; the se(2) commutator is the bracket of
    right-/left-invariant fields, i.e. of spatial/body twists. Adding it to
    hybrid twists mixed two conventions (Phase 13): e.g. "translate along
    x" and "spin about the EE point" have commuting flows, yet the old
    formula returned (0, -1, 0) for them, and it flagged the (commuting)
    joint fields of a 2R arm as non-commuting in 177/186 cells. See
    tests/test_stage3_bracket.py.

    Returns (bracket, has_transport, central). With no neighbor data on
    either field the bracket cannot be estimated; it is returned as zeros
    with has_transport=False, and callers must not read that as
    "commuting". `central` is True only if both fields' derivatives used
    central differences on both axes; one-sided estimates at the edge of
    the data are much noisier (all 7 false positives on the 2R test were
    such cells) and are reported as low-confidence.
    """
    gens = stage2_results[key]["generators"]
    X, Y = gens[gi], gens[gj]
    Jx, cx = _spatial_jacobian(key, gi, stage2_results, cell_w, cell_h, L)
    Jy, cy = _spatial_jacobian(key, gj, stage2_results, cell_w, cell_h, L)
    bracket = np.zeros(3)
    has_transport = False
    if Jy is not None:
        bracket = bracket + Jy @ X[:2]
        has_transport = True
    if Jx is not None:
        bracket = bracket - Jx @ Y[:2]
        has_transport = True
    return bracket, has_transport, (cx and cy)


def _local_rank(gens, L, tol=1e-6):
    if len(gens) == 0:
        return 0
    s = np.linalg.svd(to_dimensionless(np.asarray(gens), L), compute_uv=False)
    return int(np.sum(s > tol * max(s.max(), 1e-12)))


def analyze_lie_algebra(stage2_results, meta):
    """Stage 3. For every evaluation point with >=2 local generators,
    compute the full pairwise Lie bracket table as vector fields over SE(2)
    (see `_vector_field_bracket`), and check whether each bracket lies in
    the span of the generators already found there (involutive / closes) or
    points in a genuinely new direction (non-involutive).

    What the output means (the lower bound it supports): a holonomic serial
    chain's available EE twists at a configuration are the column space of
    its Jacobian, and its EE image is an immersed submanifold, so that family
    of spaces is involutive. If the task *requires* every twist in a local
    generator set D at some pose, any holonomic chain realizing the task
    needs at least rank(Lie closure of D) joints. `dof_lower_bound` is the
    maximum of that closed local rank over all cells. It is only as good as
    the requirement it is computed from: a single-valued field is rank 1
    everywhere (always involutive), so for the current exact field the bound
    is 1 by construction -- higher DOF has to come from Stage 5's minimality
    search against the task's pose/collision requirements (Phase 13). The
    bound also assumes a holonomic chain; nonholonomic transmissions
    (Nakamura et al. 2001) can beat it.

    Local closure: a "new_direction" bracket is appended to that cell's
    generator set (`closed_stage2_results`). In se(2) (dim 3) this is a
    single pass: at 1 generator there are no pairs, at 2 a new independent
    direction makes 3 which spans everything, and 3 independent generators
    already span everything. A future SE(3) extension needs real iteration.

    `algebra_dim_estimate` is kept as an alias of `dof_lower_bound` for
    existing report consumers.
    """
    cell_w, cell_h = meta["cell_w"], meta["cell_h"]
    L = float(meta.get("length_scale", 1.0))
    per_cell = {}
    n_multi = 0
    n_commuting = 0
    n_new_direction = 0
    n_with_spatial_data = 0
    n_no_estimate = 0
    n_low_confidence = 0
    for key, res in stage2_results.items():
        gens = res["generators"]
        k = gens.shape[0]
        if k < 2:
            continue
        n_multi += 1
        pairs = {}
        cell_has_transport = False
        for a in range(k):
            for b in range(a + 1, k):
                bracket, has_transport, central = _vector_field_bracket(
                    key, a, b, stage2_results, cell_w, cell_h, L)
                cell_has_transport = cell_has_transport or has_transport
                bnorm = float(np.linalg.norm(to_dimensionless(bracket, L)))
                resid = span_residual(gens.T, bracket, L)
                if not has_transport:
                    status = "no_estimate"
                elif bnorm < 1e-6:
                    status = "commuting"
                else:
                    status = "closes_in_span" if resid < SPAN_RESIDUAL_TOL else "new_direction"
                    if status == "new_direction" and not central:
                        status = "new_direction_low_confidence"
                Gd = to_dimensionless(gens, L).T
                bd = to_dimensionless(bracket, L)
                c, *_ = np.linalg.lstsq(Gd, bd, rcond=None)
                residual_vector = (bd - Gd @ c) * np.array([L, L, 1.0])  # back to physical units
                pairs[(a, b)] = {"bracket": bracket, "bracket_norm": bnorm,
                                 "status": status, "has_transport": has_transport,
                                 "central_difference": central,
                                 "residual_vector": residual_vector}

        statuses = [p["status"] for p in pairs.values()]
        if all(s == "commuting" for s in statuses):
            n_commuting += 1
        if any(s == "new_direction" for s in statuses):
            n_new_direction += 1
        elif any(s == "new_direction_low_confidence" for s in statuses):
            n_low_confidence += 1
        if all(s == "no_estimate" for s in statuses):
            n_no_estimate += 1
        if cell_has_transport:
            n_with_spatial_data += 1
        per_cell[key] = {"pairs": pairs, "has_spatial_data": cell_has_transport}

    # Closure: separate downstream pass over the same first-pass data.
    MIN_BRACKET_NORM_FOR_CLOSURE = 1e-3
    closed_stage2_results = {}
    n_closure_promoted = 0
    local_ranks = {}
    for key, res in stage2_results.items():
        gens = res["generators"]
        new_gens = gens
        if gens.shape[0] == 2:
            pair = per_cell.get(key, {}).get("pairs", {}).get((0, 1))
            if (pair is not None and pair["status"] == "new_direction"
                    and pair["bracket_norm"] > MIN_BRACKET_NORM_FOR_CLOSURE):
                new_gen = twist_normalize(pair["residual_vector"], L)
                new_gens = np.vstack([gens, new_gen[None, :]])
                n_closure_promoted += 1
        closed_stage2_results[key] = {**res, "generators": new_gens,
                                      "intrinsic_dim": int(new_gens.shape[0])}
        local_ranks[key] = _local_rank(new_gens, L)

    rank_values = list(local_ranks.values())
    dof_lower_bound = max(rank_values) if rank_values else 0
    rank_hist = {int(r): int(sum(1 for x in rank_values if x == r)) for r in sorted(set(rank_values))}

    if n_multi == 0:
        summary = ("Every cell carries a single required twist (rank-1 field). A "
                   "rank-1 distribution is always involutive, so the field alone "
                   "implies a local DOF lower bound of 1; any further DOF must come "
                   "from the task's pose/collision requirements (Stage 5's "
                   "minimality search).")
    else:
        summary = (f"{n_multi} cells have >=2 required generators; "
                   f"{n_new_direction} have a bracket outside the local span "
                   f"(+{n_low_confidence} more only by one-sided differences, not "
                   f"counted), "
                   f"{n_commuting} commute, {n_no_estimate} had no neighbor data to "
                   f"estimate a bracket. Max closed local rank (DOF lower bound for a "
                   f"holonomic chain): {dof_lower_bound}.")

    return {
        "per_cell": per_cell,
        "n_multi_generator_cells": n_multi,
        "n_commuting": n_commuting,
        "n_new_direction": n_new_direction,
        "n_no_estimate": n_no_estimate,
        "n_new_direction_low_confidence": n_low_confidence,
        "n_cells_with_spatial_derivative": n_with_spatial_data,
        "local_rank_histogram": rank_hist,
        "dof_lower_bound": dof_lower_bound,
        "algebra_dim_estimate": dof_lower_bound,
        "summary": summary,
        "closed_stage2_results": closed_stage2_results,
        "n_closure_promoted": n_closure_promoted,
    }


def _instantaneous_center(point, v, omega):
    """For a planar twist (v, omega) observed at world point `point`,
    return the instantaneous center of rotation (ICR), i.e. the screw axis
    location (pitch = 0 always in 2D -- no helical motion is possible)."""
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    # v = omega * J @ (point - center)  =>  center = point - (1/omega) * Jinv @ v
    # Jinv = J.T = -J
    center = np.array(point) - (1.0 / omega) * (-J @ np.array(v))
    return center


def classify_primitives(stage2_results, stage3_result=None, length_scale=1.0,
                        prismatic_radius_factor=PRISMATIC_RADIUS_FACTOR):
    """Stage 4: classify *every* local generator at every cell via screw
    theory (revolute vs prismatic; a helical/coupled classification would
    require an out-of-plane pitch, impossible in strict 2D -- flagged as a
    simplification), and cluster revolute generators' ICRs to propose a
    small number of candidate physical joint axes.

    Classifies all of a cell's generators, not just the dominant one
    (`generators[0]`) -- a cell can have >=2 generators when Stage 2 found
    statistically-structured residual motion beyond the significant mean
    direction (e.g. exactly the cells Stage 3 flags as "new_direction",
    non-commuting rotation+translation). Previously those secondary
    generators were silently dropped here even though Stage 2 had already
    discovered them, so `icr_clusters` below was clustering from an
    incomplete population of the field's actual generators.

    If `stage3_result` is given (item 22), each record also gets a
    `"couplings"` list -- which other co-located generators it was bracketed
    against in Stage 3, and whether they commute -- using bracket data Stage
    3 already computed but previously discarded after collapsing it into one
    global DOF count. Closure-derived generators (Stage 3's third generator
    at a promoted cell, if `stage2_results` is the closed dict) get
    `couplings: []` by construction: Stage 3's bracket pass only ever
    indexes the original, pre-closure generator count, so there's no pair
    entry for an index that didn't exist yet when brackets were computed.
    Intentional, not a gap to fix.
    """
    records = []
    for key, res in stage2_results.items():
        pair_info = {}
        if stage3_result is not None:
            pair_info = stage3_result["per_cell"].get(key, {}).get("pairs", {})
        for gi, g in enumerate(res["generators"]):
            v, omega = g[:2], g[2]
            vnorm = np.linalg.norm(v)
            # prismatic <=> ICR farther than factor * L (radius of curvature
            # |v|/|omega|); unit-consistent, unlike a threshold on omega.
            if abs(omega) * prismatic_radius_factor * length_scale <= vnorm:
                kind = "prismatic"
                params = {"direction": (v / (vnorm + 1e-12)).tolist()}
            else:
                kind = "revolute"
                center = _instantaneous_center(res["center"], v, omega)
                params = {"icr": center.tolist(), "omega_sign": float(np.sign(omega))}
            couplings = [
                {"with": (b if gi == a else a), "status": info["status"],
                 "bracket_norm": float(info["bracket_norm"])}
                for (a, b), info in pair_info.items() if gi in (a, b)
            ]
            records.append({
                "center": res["center"], "kind": kind, "params": params,
                "generator": g, "cell": key, "generator_index": gi,
                "is_dominant": gi == 0, "couplings": couplings,
            })

    n_prismatic = sum(1 for r in records if r["kind"] == "prismatic")
    n_revolute = len(records) - n_prismatic

    icr_clusters = _cluster_revolute_axes(
        [r for r in records if r["kind"] == "revolute"], k_max=4, length_scale=length_scale,
    )

    return {
        "records": records,
        "n_prismatic": n_prismatic,
        "n_revolute": n_revolute,
        "icr_clusters": icr_clusters,
    }


def _cluster_revolute_axes(revolute_records, k_max=4, n_iter=50, seed=0, length_scale=1.0):
    """Cluster revolute generators and pick a cluster count via silhouette
    score, k-means++-initialized (SIMPLIFICATIONS.md item 9).

    Clustering happens in the bounded, unit-consistent twist space: each
    generator is mapped to dimensionless form (v/L, omega), sign-fixed so
    omega > 0, and normalized. That stays bounded as a generator approaches
    pure translation, unlike raw ICR coordinates.

    Each cluster's `center` is the coordinate-wise median of its members'
    own ICRs (robust to the far-away ICRs of near-prismatic members), and
    `icr_spread` is the median distance of member ICRs from that center.
    Read `icr_spread` relative to the workspace before treating a cluster as
    "one axis": the ICRs of the end-effector's motion trace its fixed
    centrode, and coincide with a physical joint axis only for motion a
    single joint produces (critique section 3.5).
    """
    if len(revolute_records) == 0:
        return []

    axes = []
    for r in revolute_records:
        g = to_dimensionless(np.array(r["generator"], dtype=float), length_scale)
        if g[2] < 0:
            g = -g
        axes.append(g / (np.linalg.norm(g) + 1e-12))
    axes = np.array(axes)

    k_max = min(k_max, len(axes))
    if len(axes) < 2 or k_max < 2:
        assign = np.zeros(len(axes), dtype=int)
        centers_axes = axes.mean(axis=0, keepdims=True)
    else:
        best_k, best_score, best_assign, best_centers = 1, -np.inf, np.zeros(len(axes), dtype=int), axes.mean(axis=0, keepdims=True)
        for k in range(2, k_max + 1):
            centers, assign = _kmeans_pp(axes, k, n_iter=n_iter, seed=seed)
            if len(set(assign.tolist())) < 2:
                continue
            score = _silhouette_score(axes, assign)
            if score > best_score:
                best_score, best_k, best_assign, best_centers = score, k, assign, centers
        # only accept multi-cluster structure if it's a clearly better fit
        # than treating all axes as one joint
        if best_score < 0.5:
            best_k, best_assign, best_centers = 1, np.zeros(len(axes), dtype=int), axes.mean(axis=0, keepdims=True)
        assign, centers_axes = best_assign, best_centers

    clusters = []
    for c in range(centers_axes.shape[0]):
        members = [revolute_records[i] for i in range(len(axes)) if assign[i] == c]
        if not members:
            continue
        icrs = np.array([m["params"]["icr"] for m in members], dtype=float)
        center = np.median(icrs, axis=0)
        spread = float(np.median(np.linalg.norm(icrs - center, axis=1)))
        axis_phys = centers_axes[c] * np.array([length_scale, length_scale, 1.0])
        clusters.append({"center": center.tolist(), "icr_spread": spread,
                         "axis": axis_phys.tolist(), "n_members": len(members)})
    return clusters


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
    """Mean silhouette coefficient (numpy-only, no sklearn dependency)."""
    n = len(points)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    labels = np.unique(assign)
    scores = np.zeros(n)
    for i in range(n):
        own = assign[i]
        same = (assign == own)
        same[i] = False
        a = d[i, same].mean() if same.any() else 0.0
        b = np.inf
        for lab in labels:
            if lab == own:
                continue
            other = assign == lab
            if other.any():
                b = min(b, d[i, other].mean())
        scores[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(scores.mean())
