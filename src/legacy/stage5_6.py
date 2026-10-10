"""
Stage 5: Assemble the Minimal Kinematic Graph
Stage 6: Validation and Embodiment Analysis

Stage 5 searches every revolute/prismatic joint string for open serial
chains of 1..n_max joints anchored at `base`, fits each candidate's design
variables (link lengths or prismatic strokes, base orientation, tool heading
offset), and selects the *smallest* joint count with a candidate that meets
the task's requirements: reach the target within its position/heading
tolerance without colliding at that configuration. Among feasible
candidates at that joint count, the lowest fitted score wins; the score
includes how well the mechanism can realize the navigation field's twists
(`_field_realizability`), which acts as a preference, not a requirement
(critique section 3.2; SIMPLIFICATIONS.md Phase 13).

Branching topologies and closed loops are still out of scope.

Twists are hybrid (EE-point velocity, omega); `jacobian()` returns them in
the same convention, and all comparisons go through the unit-consistent
metric in `problem.py`.
"""

import itertools

import numpy as np
from scipy.optimize import minimize, least_squares

from problem import span_residual, to_dimensionless, wrap_angle


MIN_LENGTH = 0.05   # floor on link lengths / prismatic strokes during fitting
POS_TOL_FRAC = 0.01  # feasibility: target position error <= this * length_scale
HEADING_SLACK = 1e-3  # feasibility: heading may exceed the tolerance band by this (rad)
REACH_COST_TOL = 1e-4  # IK sum-of-squares below this counts as "pose reached"


# --------------------------------------------------------------------------
# General planar serial-chain kinematics (mixed revolute/prismatic joints)
# --------------------------------------------------------------------------

def fk(base, joint_types, dof, lengths, base_angle=0.0, tool_offset=0.0):
    """Forward kinematics of a planar serial chain of revolute ('R') and
    prismatic ('P') joints anchored at `base`.

    - The chain's initial heading is `base_angle` (a design variable; it used
      to be fixed at 0, so a leading P joint always slid along world +x).
    - Revolute joint i: dof[i] is the joint angle; lengths[i] is the fixed
      link length traversed along the new heading after the rotation.
    - Prismatic joint i: a rail of length lengths[i] (the stroke) along the
      current heading; dof[i] in [0, lengths[i]] is the carriage position,
      and the rest of the chain rides on the carriage.
    - The end-effector heading is the accumulated chain heading plus
      `tool_offset` (a fixed tool angle; a design variable).

    Returns end-effector (x, y, theta) and the list of joint positions
    (base, then the end of each joint's link / the carriage of each rail).
    """
    x, y = float(base[0]), float(base[1])
    phi = float(base_angle)
    joints = [np.array([x, y])]
    for t, val, L in zip(joint_types, dof, lengths):
        if t == "R":
            phi += val
            x += L * np.cos(phi)
            y += L * np.sin(phi)
        else:
            x += val * np.cos(phi)
            y += val * np.sin(phi)
        joints.append(np.array([x, y]))
    return np.array([x, y, phi + tool_offset]), joints


def body_segments(base, joint_types, dof, lengths, base_angle=0.0):
    """Physical line segments occupied by the mechanism at configuration
    `dof`, for collision checks: each revolute link, and each prismatic
    joint's *whole rail* (it occupies its full stroke regardless of where
    the carriage is)."""
    _, joints = fk(base, joint_types, dof, lengths, base_angle)
    segs = []
    phi = float(base_angle)
    for i, (t, val, L) in enumerate(zip(joint_types, dof, lengths)):
        if t == "R":
            phi += val
            segs.append((joints[i], joints[i + 1]))
        else:
            d = np.array([np.cos(phi), np.sin(phi)])
            segs.append((joints[i], joints[i] + max(L, MIN_LENGTH) * d))
    return segs


def jacobian(base, joint_types, dof, lengths, base_angle=0.0):
    """Geometric Jacobian mapping joint velocities to the end-effector's
    hybrid twist (vx, vy, omega): world-frame velocity of the EE point plus
    heading rate. Revolute columns use z x (p_ee - p_i); prismatic columns
    are the rail direction, with no contribution to omega. The tool offset
    is a constant heading shift and does not enter the Jacobian."""
    n = len(joint_types)
    ee, joints = fk(base, joint_types, dof, lengths, base_angle)
    p_ee = ee[:2]
    J = np.zeros((3, n))
    phi = float(base_angle)
    for i, (t, val) in enumerate(zip(joint_types, dof)):
        p_i = joints[i]
        if t == "R":
            J[0, i] = -(p_ee[1] - p_i[1])
            J[1, i] = (p_ee[0] - p_i[0])
            J[2, i] = 1.0
            phi += val
        else:
            J[0, i] = np.cos(phi)
            J[1, i] = np.sin(phi)
            J[2, i] = 0.0
    return J


def _joint_bounds(joint_types, lengths):
    lo = np.array([-np.inf if t == "R" else 0.0 for t in joint_types])
    hi = np.array([np.inf if t == "R" else max(L, MIN_LENGTH) for t, L in zip(joint_types, lengths)])
    return lo, hi


def solve_ik(base, joint_types, lengths, target_xytheta, dof0=None, n_restarts=8, seed=0,
             base_angle=0.0, tool_offset=0.0, heading_tol=0.0, position_only=False):
    """Bounded least-squares IK with the analytic Jacobian.

    Residual: EE position error, plus the heading error beyond
    `heading_tol` (a dead-band, matching the task's orientation tolerance),
    unless `position_only`. Prismatic joints are bounded to [0, stroke].
    Returns (dof, cost) with cost = sum of squared residuals.
    """
    target = np.asarray(target_xytheta, dtype=float)
    lo, hi = _joint_bounds(joint_types, lengths)
    rng = np.random.default_rng(seed)

    def fun_jac(d):
        ee, _ = fk(base, joint_types, d, lengths, base_angle, tool_offset)
        J = jacobian(base, joint_types, d, lengths, base_angle)
        r = ee[:2] - target[:2]
        if position_only:
            return r, J[:2]
        e = wrap_angle(ee[2] - target[2])
        if abs(e) > heading_tol:
            return np.append(r, e - np.sign(e) * heading_tol), J
        return np.append(r, 0.0), np.vstack([J[:2], np.zeros(len(joint_types))])

    inits = []
    if dof0 is not None:
        inits.append(np.clip(np.asarray(dof0, dtype=float), lo, hi))
    for _ in range(n_restarts):
        inits.append(np.array([rng.uniform(-np.pi, np.pi) if t == "R" else rng.uniform(0.05, 0.95) * h
                               for t, h in zip(joint_types, hi)]))

    best, best_cost = None, np.inf
    for d0 in inits:
        res = least_squares(lambda d: fun_jac(d)[0], d0, jac=lambda d: fun_jac(d)[1],
                            bounds=(lo, hi), method="trf", xtol=1e-10, ftol=1e-10, max_nfev=100)
        cost = float(np.sum(res.fun ** 2))
        if cost < best_cost:
            best_cost, best = cost, res.x
        if best_cost < 1e-14:
            break
    best = np.array([wrap_angle(v) if t == "R" else v for t, v in zip(joint_types, best)])
    return best, best_cost


# --------------------------------------------------------------------------
# CHOMP-style obstacle cost (Zucker et al. 2013), borrowed as a *cost model*
# only, evaluated on the mechanism's physical body segments at already-solved
# configurations.
# --------------------------------------------------------------------------

def _link_obstacle_cost(problem, segments, n_body_points=6, epsilon=0.3):
    """Softened obstacle cost averaged over points sampled along each body
    segment: zero once clear by `epsilon`, a quadratic ramp near the
    boundary, and a linear penetration penalty inside."""
    total = 0.0
    n_points = 0
    for a, b in segments:
        for t in np.linspace(0, 1, n_body_points):
            d = problem.obstacle_clearance((1 - t) * a + t * b)
            if d < 0:
                c = -d + 0.5 * epsilon
            elif d < epsilon:
                c = (d - epsilon) ** 2 / (2 * epsilon)
            else:
                c = 0.0
            total += c
            n_points += 1
    return total / max(n_points, 1)


def _min_clearance(problem, segments, n_body_points=25):
    return float(min(problem.obstacle_clearance((1 - t) * a + t * b)
                     for a, b in segments for t in np.linspace(0, 1, n_body_points)))


# --------------------------------------------------------------------------
# Field realizability, shared by Stage 5 (objective) and Stage 6 (validation)
# --------------------------------------------------------------------------

def _field_realizability(problem, design, stage2_results, keys, n_restarts=1, seed=0,
                         warm=None, with_collision=False):
    """For each cell: put the mechanism at the *pose the field requires
    there* -- (center, field heading) -- and measure how much of the
    required twist its Jacobian can produce:

        realizability = 1 - span_residual(J, required twist)   (W-metric)

    so 1 means the twist lies in the Jacobian's column space and 0 means it
    is orthogonal to it. A cell whose pose the mechanism cannot reach scores
    0 (it cannot produce the twist there at all) rather than being skipped,
    which used to inflate the average for chains that reach few cells.

    Replaces `_direction_alignment` (Phase 13), which (a) solved IK for
    heading 0 at every cell instead of the field's heading -- the source of
    the PRP "always singular" artifact -- and (b) scored the best single
    Jacobian column's |cosine| with the twist instead of the column space.

    `warm` is an optional {key: dof} cache of previous solutions, used as IK
    warm starts across calls with nearby designs (Stage 5's inner loop).
    """
    L = problem.length_scale
    base = np.asarray(problem.base, dtype=float)
    jt, lengths = design["joint_types"], design["lengths"]
    ba, to = design["base_angle"], design["tool_offset"]
    out = {"scores": [], "reached": [], "collision_costs": [], "configs": {}}
    for key in keys:
        res = stage2_results[key]
        p = res["center"]
        h = res.get("heading")
        target = np.array([p[0], p[1], 0.0 if h is None else h])
        dof, cost = solve_ik(base, jt, lengths, target, dof0=None if warm is None else warm.get(key),
                             n_restarts=n_restarts, seed=seed, base_angle=ba, tool_offset=to,
                             position_only=h is None)
        if warm is not None:
            warm[key] = dof
        reached = cost < REACH_COST_TOL
        out["reached"].append(bool(reached))
        if not reached:
            out["scores"].append(0.0)
            continue
        out["configs"][key] = dof
        J = jacobian(base, jt, dof, lengths, ba)
        out["scores"].append(1.0 - span_residual(J, res["generators"][0], L))
        if with_collision:
            out["collision_costs"].append(
                _link_obstacle_cost(problem, body_segments(base, jt, dof, lengths, ba),
                                    epsilon=0.03 * L))
    return out


def _sigma_ratio(J, L):
    s = np.linalg.svd(to_dimensionless(J.T, L).T, compute_uv=False)
    return float(s.min() / (s.max() + 1e-12))


# --------------------------------------------------------------------------
# Stage 5: topology search
# --------------------------------------------------------------------------

def _sign_invariant_mean_direction(vecs):
    """Mean direction of unit vectors with arbitrary/inconsistent sign (e.g.
    prismatic generators, which have no canonical sign) -- a naive vector
    mean would partially cancel opposite-sign recordings of the same
    physical axis. Uses the dominant eigenvector of the mean outer product,
    the same sign-invariant approach `_cluster_revolute_axes` already uses
    (there via forcing omega > 0; direction has no such natural sign here,
    so this is the general version). Returns None if `vecs` is empty."""
    if len(vecs) == 0:
        return None
    vecs = np.asarray(vecs, dtype=float)
    M = np.einsum("ni,nj->ij", vecs, vecs) / len(vecs)
    eigvals, eigvecs = np.linalg.eigh(M)
    return eigvecs[:, np.argmax(eigvals)]


def _coupling_strength(records):
    """Mean non-commuting-coupling strength (item 22) over `records`: for
    each record, its strongest `"new_direction"` coupling's `bracket_norm`
    (from Stage 4's `couplings`, see stage3_4.py), or 0 if it has none.
    Deliberately coarse -- a single global signal per kind (revolute or
    prismatic), not per-specific-cluster, since there's no per-cluster
    member tracking to aggregate over without adding that plumbing (a finer
    version is possible but not built here). Returns None for an empty
    input, so callers can distinguish "no data" from "real zero signal"."""
    if not records:
        return None
    vals = [max([c["bracket_norm"] for c in r.get("couplings", []) if c["status"] == "new_direction"],
                default=0.0) for r in records]
    return float(np.mean(vals))


def prefilter_type_counts(n, stage4_result, stage2_results, length_scale=1.0):
    """Cheap, IK-free pre-check run once before Stage 5's real combinatorial
    search: for each joint-type-*count* class (e.g. for n=3: 3R, 2R1P, 1R2P,
    3P), build a candidate Jacobian directly from Stage 4's already-
    discovered axis geometry (`icr_clusters` for revolute axes, a
    sign-invariant mean direction over prismatic records for prismatic) --
    independently-placed, fixed axes, NOT threaded through a real chain's
    cumulative link lengths -- and check its column-space span against the
    field's actual required generators via least-squares residual (same
    style as `stage3_4.py`'s bracket-closure check).

    A fixed-axis Jacobian's span never depends on column order, so this can
    only discriminate between joint-type *counts*, never orderings of the
    same count (RRP vs RPR vs PRR are indistinguishable to it) -- ordering
    stays exclusively the real search's job below. This is deliberately
    soft/diagnostic: callers should attach these scores to the report and
    may use them to reorder which candidates get evaluated first, but must
    never use them to exclude a candidate -- the independent-axis
    idealization can't see chain-ordering/reachability effects a real serial
    chain has, so a poor score here is not proof a real chain can't do
    better.

    Important caveat on interpreting `"coverage"`: at n=3 (this problem's
    actual DOF count), a generic 3x3 `J_hyp` with at least one revolute
    column is full-rank almost everywhere, and a full-rank 3x3 system has an
    *exact* least-squares solution for any target twist -- so `"coverage"`
    saturates near 1 for essentially every class with `n_R >= 1`, and mostly
    only distinguishes "has a revolute axis" from the `n_R == 0` (all-
    prismatic) edge, which provably can't reproduce a nonzero-omega
    generator. The real discriminating signal at n=3 is in `"min_rank"`/
    `"frac_full_rank"`, not the coverage number. At n=2 (or a hypothetical
    higher n), where full rank isn't a given, `"coverage"` carries more real
    directional information.

    Also: unlike revolute (which draws distinct axes from `icr_clusters` up
    to however many real clusters exist), prismatic classes with `n_P >= 2`
    reuse the *same* single representative direction for every prismatic
    column *by construction* -- there's no per-direction clustering step
    for prismatic axes analogous to `_cluster_revolute_axes`. So
    multi-prismatic classes are systematically rank-limited in their P
    sub-block regardless of how directionally diverse the demonstrated
    prismatic motion actually was. A stated bias, not a bug to fix here.

    Returns `{(n_R, n_P): {...} | None, ...}` for every `n_R + n_P == n`,
    `n_R, n_P >= 0`. A class scores `None` if the data needed to build it
    doesn't exist at all (zero revolute clusters but n_R>0 needed, or zero
    prismatic records but n_P>0 needed) -- guarded explicitly so an empty
    array's `.mean()` can't silently produce NaN that poisons the residual
    without raising. Note Stage 5 fits every joint string, so every class
    enumerated here (e.g. no all-prismatic entry at any n) -- expected, not
    a bug: some returned keys won't have a matching candidate string.

    Each class also gets `"coupling_signal"` (item 22, descoped from an
    originally-planned ordering/adjacency scorer -- see
    notes/open_research_quesitons/lie_algebra_pipeline_plan.md for why):
    a weighted blend of two global, kind-level non-commuting-coupling
    strengths (`_coupling_strength`), by how many revolute/prismatic slots
    the class uses. Purely informational, same soft/diagnostic status as
    `coverage`/`min_rank` -- never used to exclude a candidate.
    """
    revolute_clusters = sorted(
        (cl for cl in stage4_result["icr_clusters"] if cl["center"] is not None),
        key=lambda cl: cl["n_members"], reverse=True,
    )
    revolute_positions = [np.array(cl["center"], dtype=float) for cl in revolute_clusters]

    prismatic_dirs = [r["params"]["direction"] for r in stage4_result["records"]
                       if r["kind"] == "prismatic"]
    prismatic_dir = _sign_invariant_mean_direction(prismatic_dirs)

    revolute_coupling_signal = _coupling_strength(
        [r for r in stage4_result["records"] if r["kind"] == "revolute"])
    prismatic_coupling_signal = _coupling_strength(
        [r for r in stage4_result["records"] if r["kind"] == "prismatic"])

    expected_max_rank = min(3, n)
    scores = {}
    for n_r in range(0, n + 1):
        n_p = n - n_r
        if n_r > 0 and not revolute_positions:
            scores[(n_r, n_p)] = None
            continue
        if n_p > 0 and prismatic_dir is None:
            scores[(n_r, n_p)] = None
            continue

        axes = [revolute_positions[i] if i < len(revolute_positions) else revolute_positions[-1]
                for i in range(n_r)]

        coverages = []
        ranks = []
        for res in stage2_results.values():
            p = np.asarray(res["center"], dtype=float)
            cols = []
            for p_i in axes:
                d = p - p_i
                cols.append([-d[1], d[0], 1.0])
            for _ in range(n_p):
                cols.append([prismatic_dir[0], prismatic_dir[1], 0.0])
            J_hyp = np.array(cols).T if cols else np.zeros((3, 0))
            ranks.append(int(np.linalg.matrix_rank(to_dimensionless(J_hyp.T, length_scale).T)) if cols else 0)
            for gen in res["generators"]:
                resid = span_residual(J_hyp, gen, length_scale) if cols else 1.0
                coverages.append(1.0 - resid)

        parts = []
        if n_r > 0 and revolute_coupling_signal is not None:
            parts.append((n_r, revolute_coupling_signal))
        if n_p > 0 and prismatic_coupling_signal is not None:
            parts.append((n_p, prismatic_coupling_signal))
        coupling_signal = (sum(w * v for w, v in parts) / sum(w for w, v in parts)
                            if parts else None)

        scores[(n_r, n_p)] = {
            "coverage": float(np.mean(coverages)) if coverages else None,
            "min_rank": min(ranks) if ranks else 0,
            "frac_full_rank": float(np.mean([r >= expected_max_rank for r in ranks])) if ranks else 0.0,
            "revolute_axes_distinct": min(n_r, len(revolute_positions)),
            "revolute_axes_used": n_r,
            "prismatic_axes_used": n_p,
            "coupling_signal": coupling_signal,
        }
    return scores


def _seed_lengths(joint_types, icr_clusters, base, reach_needed, n):
    """Initial per-joint length guess for the Nelder-Mead search below,
    informed by Stage 4's clustered ICR positions instead of blindly
    splitting `reach_needed` evenly across joints (the previous behavior).
    Stage 4 already estimated where in the workspace each revolute axis
    sits (`icr_clusters`, from the field's own generator directions); this
    reuses that geometry as a starting point rather than rediscovering axis
    placement from scratch inside every candidate's optimization.

    Heuristic only (Nelder-Mead still refines it): walk `joint_types` in
    order, and for each 'R' joint assign the next not-yet-used revolute
    cluster's distance-from-base (minus the cumulative distance already
    "spent" by earlier joints) as that joint's initial length. Joints with
    no matching cluster (prismatic joints, or more R joints than clusters)
    fall back to the old uniform split.
    """
    base = np.array(base, dtype=float)
    revolute_dists = sorted(
        float(np.linalg.norm(np.array(cl["center"]) - base))
        for cl in icr_clusters if cl["center"] is not None
    )
    fallback = max(reach_needed / n, 0.1)
    lengths = np.full(n, fallback)
    cumulative = 0.0
    dist_iter = iter(revolute_dists)
    for i, t in enumerate(joint_types):
        d = next(dist_iter, None) if t == "R" else None
        if d is not None:
            lengths[i] = float(np.clip(d - cumulative, 0.1, 2.0 * reach_needed))
            cumulative = max(cumulative + lengths[i], d)
        else:
            cumulative += fallback
    return lengths


def _seed_design(joint_types, icr_clusters, problem, seed):
    """Initial design vector [lengths..., base_angle, tool_offset]. Lengths
    come from `_seed_lengths` (ICR-informed for revolute joints); the base
    angle points at the target. Seeds > 0 perturb the start so a multi-seed
    run is a genuine multi-start rather than repeated identical fits."""
    n = len(joint_types)
    base = np.asarray(problem.base, dtype=float)
    target = np.asarray(problem.target, dtype=float)
    reach = float(np.linalg.norm(target - base))
    lengths = _seed_lengths(joint_types, icr_clusters, base, reach, n)
    base_angle = float(np.arctan2(*(target - base)[::-1]))
    tool_offset = 0.0
    if seed > 0:
        rng = np.random.default_rng(seed)
        lengths = lengths * rng.uniform(0.7, 1.3, n)
        base_angle += rng.uniform(-0.5, 0.5)
        tool_offset += rng.uniform(-0.5, 0.5)
    return np.concatenate([lengths, [base_angle, tool_offset]])


def _unpack(params, joint_types):
    n = len(joint_types)
    return {"joint_types": list(joint_types),
            "lengths": np.maximum(np.abs(params[:n]), MIN_LENGTH),
            "base_angle": float(params[n]),
            "tool_offset": float(params[n + 1])}


def _evaluate_at_target(problem, design, n_restarts, seed, dof0=None):
    base = np.asarray(problem.base, dtype=float)
    target_pose = np.array([*problem.target, problem.target_theta])
    dof, _ = solve_ik(base, design["joint_types"], design["lengths"], target_pose, dof0=dof0,
                      n_restarts=n_restarts, seed=seed, base_angle=design["base_angle"],
                      tool_offset=design["tool_offset"], heading_tol=problem.target_theta_tol)
    ee, joints = fk(base, design["joint_types"], dof, design["lengths"],
                    design["base_angle"], design["tool_offset"])
    pos_err = float(np.linalg.norm(ee[:2] - target_pose[:2]))
    orient_err = float(abs(wrap_angle(ee[2] - target_pose[2])))
    heading_violation = max(0.0, orient_err - problem.target_theta_tol)
    segs = body_segments(base, design["joint_types"], dof, design["lengths"], design["base_angle"])
    return dof, ee, joints, segs, pos_err, orient_err, heading_violation


DEFAULT_WEIGHTS = {"pose": 100.0, "align": 0.5, "collision": 3.0, "reg": 0.01}


def _fit_candidate(problem, joint_types, stage2_results, align_keys, seed, icr_clusters,
                   weights=None):
    """Fit one joint string's design variables with Nelder-Mead.

    Objective = pose * [(pos_err/L)^2 + heading_violation^2]
              + align * (1 - mean field realizability on `align_keys`)
              + collision * mean CHOMP-style cost (target + alignment configs)
              + reg * std(lengths)/L

    The Stage-4 type-fraction mismatch term is gone (Phase 13): the fractions
    it compared against were dominated by constructed generators and have no
    theoretical link to which joint types a chain needs.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    L = problem.length_scale
    n = len(joint_types)
    warm = {}
    target_warm = [None]

    def cost(params):
        design = _unpack(params, joint_types)
        dof, _, _, segs, pos_err, _, hv = _evaluate_at_target(problem, design, 2, seed, target_warm[0])
        target_warm[0] = dof
        fr = _field_realizability(problem, design, stage2_results, align_keys, n_restarts=1,
                                  seed=seed, warm=warm, with_collision=True)
        align_pen = 1.0 - float(np.mean(fr["scores"])) if fr["scores"] else 1.0
        col = float(np.mean(fr["collision_costs"] + [_link_obstacle_cost(problem, segs, epsilon=0.03 * L)]))
        return (w["pose"] * ((pos_err / L) ** 2 + hv ** 2) + w["align"] * align_pen
                + w["collision"] * col + w["reg"] * float(np.std(design["lengths"])) / L)

    x0 = _seed_design(joint_types, icr_clusters, problem, seed)
    res = minimize(cost, x0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400, "maxfev": 60 * (n + 2)})
    design = _unpack(res.x, joint_types)

    dof, ee, joints, segs, pos_err, orient_err, hv = _evaluate_at_target(problem, design, 8, seed)
    min_clear = _min_clearance(problem, segs)
    fr = _field_realizability(problem, design, stage2_results, align_keys, n_restarts=3,
                              seed=seed, with_collision=True)
    feasible = bool(pos_err <= POS_TOL_FRAC * L and hv <= HEADING_SLACK and min_clear > 0.0)

    return {
        **design, "n_links": n,
        "home_dof": dof, "achieved_ee": ee, "joints": joints, "segments": segs,
        "score": float(res.fun), "assembly_cost": float(res.fun),
        "pos_err": pos_err, "orient_err": orient_err, "heading_violation": float(hv),
        "min_clearance_at_target": min_clear, "feasible": feasible,
        "mean_alignment": float(np.mean(fr["scores"])) if fr["scores"] else float("nan"),
        "align_reach_fraction": float(np.mean(fr["reached"])) if fr["reached"] else 0.0,
        "mean_collision_cost": float(np.mean(fr["collision_costs"] + [_link_obstacle_cost(problem, segs, epsilon=0.03 * L)])),
        "seed": seed,
    }


def _summarize(c):
    return {"joint_types": "".join(c["joint_types"]), "n_links": c["n_links"],
            "feasible": c["feasible"], "score": c["score"], "pos_err": c["pos_err"],
            "orient_err": c["orient_err"], "heading_violation": c["heading_violation"],
            "min_clearance_at_target": c["min_clearance_at_target"],
            "mean_alignment": c["mean_alignment"], "align_reach_fraction": c["align_reach_fraction"],
            "mean_collision_cost": c["mean_collision_cost"], "prefilter": c.get("prefilter")}


def assemble_mechanism(problem, stage3_result, stage4_result, stage2_results,
                       n_align_samples=8, seed=0, top_k=3, n_max=3, weights=None):
    """Stage 5: fit every R/P joint string with 1..n_max joints and select
    the smallest joint count that has a feasible candidate (reaches the
    target within tolerance, collision-free there). Among feasible
    candidates at that count, the lowest score wins; if nothing is feasible
    at any count, the lowest score overall is returned with
    `selected_n = None`.

    Searching from 1 joint regardless of Stage 3 is deliberate: it tests
    Stage 3's `dof_lower_bound` instead of trusting it
    (`lower_bound_consistent` is False if a design with fewer joints than
    the bound turned out feasible).

    Alignment cells are a fixed, evenly spaced subset of the grid (not a
    random draw), so seeds differ only through the fit's starting point and
    IK restarts.
    """
    L = problem.length_scale
    lower_bound = int(stage3_result.get("dof_lower_bound", stage3_result.get("algebra_dim_estimate", 1)))
    keys = sorted(stage2_results.keys())
    if len(keys) > n_align_samples:
        idx = np.unique(np.linspace(0, len(keys) - 1, n_align_samples).round().astype(int))
        align_keys = [keys[i] for i in idx]
    else:
        align_keys = keys

    all_cands, per_n, prefilter_scores = [], {}, {}
    for n in range(1, n_max + 1):
        pf = prefilter_type_counts(n, stage4_result, stage2_results, L)
        prefilter_scores.update({f"{n_r}R{n_p}P": v for (n_r, n_p), v in pf.items()})
        cands = []
        for jt in itertools.product("RP", repeat=n):
            c = _fit_candidate(problem, jt, stage2_results, align_keys, seed,
                               stage4_result["icr_clusters"], weights)
            c["prefilter"] = pf.get((jt.count("R"), jt.count("P")))
            cands.append(c)
        feas = sorted((c for c in cands if c["feasible"]), key=lambda c: c["score"])
        per_n[n] = {"n_candidates": len(cands), "n_feasible": len(feas),
                    "feasible": ["".join(c["joint_types"]) for c in feas],
                    "best_feasible": "".join(feas[0]["joint_types"]) if feas else None}
        all_cands += cands

    feasible_ns = [n for n in per_n if per_n[n]["n_feasible"] > 0]
    selected_n = min(feasible_ns) if feasible_ns else None
    pool = [c for c in all_cands if c["n_links"] == selected_n and c["feasible"]] if selected_n else all_cands
    ranked = sorted(pool, key=lambda c: c["score"])
    best = ranked[0]
    best["selected_n"] = selected_n
    best["dof_lower_bound"] = lower_bound
    best["lower_bound_consistent"] = all(per_n[n]["n_feasible"] == 0 for n in per_n if n < lower_bound)
    best["per_n"] = per_n
    best["candidates_tried"] = [_summarize(c) for c in sorted(all_cands, key=lambda c: (c["n_links"], c["score"]))]
    best["top_k_mechanisms"] = ranked[:top_k]
    best["prefilter_scores"] = prefilter_scores
    return best


# --------------------------------------------------------------------------
# Stage 6: validation
# --------------------------------------------------------------------------

def validate_mechanism(problem, mechanism, coverage_locs, stage2_results,
                       stage3_result=None, n_config_samples=400, seed=0,
                       singularity_ratio_eps=1e-2):
    """Stage 6: independent re-check of a fitted mechanism.

    - Coverage: crude distance-from-base annulus test over `coverage_locs`
      (ignores orientation and occlusion; SIMPLIFICATIONS.md item 13).
    - Field realizability over *every* grid cell, at the field's pose there
      (see `_field_realizability`): fraction of cells whose pose is
      reachable, and mean realizability (unreachable cells count as 0).
    - Singularities: sigma_min/sigma_max of the dimensionless Jacobian at
      every reached cell and at the target configuration. A relative,
      unit-consistent threshold (Phase 13; the old check solved IK for
      heading 0, which forced PRP onto its singular angle).
    - Collision: minimum clearance of the physical body (rails included) at
      the target configuration. Still a static check, not swept motion
      (item 12).

    `stage3_result` is accepted for interface compatibility; singularities
    are now checked at every reached cell rather than only Stage 3's
    "new_direction" cells, which a single-valued field no longer produces.
    """
    rng = np.random.default_rng(seed)
    L = problem.length_scale
    jt = mechanism["joint_types"]
    n = mechanism["n_links"]
    lengths = mechanism["lengths"]
    ba, to = mechanism.get("base_angle", 0.0), mechanism.get("tool_offset", 0.0)
    base = np.asarray(problem.base, dtype=float)
    design = {"joint_types": jt, "lengths": lengths, "base_angle": ba, "tool_offset": to}

    dof_samples = np.zeros((n_config_samples, n))
    for i, t in enumerate(jt):
        if t == "R":
            dof_samples[:, i] = rng.uniform(-np.pi, np.pi, n_config_samples)
        else:
            dof_samples[:, i] = rng.uniform(0.0, max(lengths[i], MIN_LENGTH), n_config_samples)
    ee_positions = np.array([fk(base, jt, d, lengths, ba, to)[0][:2] for d in dof_samples])

    dists_from_base = np.linalg.norm(ee_positions - base, axis=1)
    reach_min, reach_max = float(dists_from_base.min()), float(dists_from_base.max())
    dists = np.linalg.norm(coverage_locs - base, axis=1)
    coverage_fraction = float(np.mean((dists >= reach_min - 1e-6) & (dists <= reach_max + 1e-6)))

    keys = list(stage2_results.keys())
    fr = _field_realizability(problem, design, stage2_results, keys, n_restarts=3, seed=seed)
    reached_scores = [s for s, r in zip(fr["scores"], fr["reached"]) if r]

    ratios = [_sigma_ratio(jacobian(base, jt, d, lengths, ba), L) for d in fr["configs"].values()]
    target_ratio = _sigma_ratio(jacobian(base, jt, mechanism["home_dof"], lengths, ba), L)
    min_clear = _min_clearance(problem, body_segments(base, jt, mechanism["home_dof"], lengths, ba))

    return {
        "reach_min": reach_min,
        "reach_max": reach_max,
        "coverage_fraction": coverage_fraction,
        "mean_generator_alignment": float(np.mean(fr["scores"])) if fr["scores"] else float("nan"),
        "field_reach_fraction": float(np.mean(fr["reached"])) if fr["reached"] else 0.0,
        "mean_realizability_where_reached": float(np.mean(reached_scores)) if reached_scores else None,
        "n_aligned_cells": len(reached_scores),
        "home_config_hits_obstacle": bool(min_clear < 0.0),
        "min_clearance_at_target": min_clear,
        "ee_position_samples": ee_positions,
        "singularity_check": {
            "n_reached": len(ratios),
            "near_singular_fraction": float(np.mean([r < singularity_ratio_eps for r in ratios])) if ratios else None,
            "median_sigma_ratio": float(np.median(ratios)) if ratios else None,
            "target_sigma_ratio": target_ratio,
            "target_near_singular": bool(target_ratio < singularity_ratio_eps),
        },
    }
