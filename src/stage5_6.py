"""
Stage 5: Reduced search
Stage 6: Validation

Stage 5 fits each admissible joint combination (Stage 4), smallest joint
count first. It stops at the first joint count that has a feasible design.
Each fit solves for one shared design and one configuration for each target:

- Design variables: a link length for each revolute joint, a housing length
  and a stroke for each prismatic joint, and `slide_angle` when the first
  joint is prismatic. The problem fixes the base angle and the probe length.
- Residuals: the probe-tip position error and heading-band violation at each
  target, a hinge on body clearance below the planning inflation, a hinge on
  body points outside the workspace box, and a small regularizer on the
  total length. The hinges apply at each target configuration and at samples
  on the joint-space transitions between consecutive targets.

A design is feasible when these conditions are true at each target
configuration and along each transition (dense check):

- The probe tip is within `POS_TOL_FRAC` * length scale of the target.
- The heading is in the pruned band.
- The clearance of each body segment is at least `link_radius`.
- The body is in the workspace box.

The score of a feasible design is (1 - smallest sigma ratio over the target
configurations) plus a small length regularizer. A lower score is better.

If no admissible combination of the selected class is feasible, the search
continues with the next covering class (see `classes.py`).

Kinematics (`fk`): straight links, no fixed bends. A revolute joint rotates
the chain, and a link follows it. A prismatic joint is a telescopic segment:
housing length h plus extension q in [0, stroke], along the current heading.
A fixed probe of `probe_length` follows the last joint. The end-effector is
the probe tip, and its heading is the last link direction. The initial
heading is `base_angle`, or `slide_angle` when the first joint is prismatic.

Twists are hybrid (EE-point velocity, omega). `jacobian()` returns them in
this convention. Conditioning uses the unit-consistent metric in
`problem.py`.
"""

import time

import numpy as np
from scipy.optimize import least_squares

from classes import ALL_COMBINATIONS, class_combinations, rotating_after_first
from priors import design_prior
from problem import to_dimensionless
from targets import heading_residual


MIN_LENGTH = 0.05      # minimum link length, housing length, and stroke
POS_TOL_FRAC = 0.01    # feasibility: position error <= POS_TOL_FRAC * length_scale
HEADING_SLACK = 1e-3   # feasibility: maximum heading error outside the pruned band (rad)
SINGULAR_RATIO = 1e-2  # sigma_min / sigma_max below this value is near-singular

W_POS, W_HEAD, W_COL, W_REG = 10.0, 3.0, 10.0, 0.03
FIT_TRANSITION_SAMPLES = 6
CHECK_TRANSITION_SAMPLES = 25
FIT_MAX_NFEV = 200     # max_nfev per start (scipy does not count Jacobian estimates)
N_RESTARTS = 3         # start points per combination and seed
LENGTH_BOUND_FACTOR = 1.2  # maximum design length = LENGTH_BOUND_FACTOR * length bound

START_MODES = ("priors", "random", "legacy_icr")


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------

def initial_heading(joint_types, base_angle=0.0, slide_angle=None):
    if len(joint_types) and joint_types[0] == "P" and slide_angle is not None:
        return float(slide_angle)
    return float(base_angle)


def fk(base, joint_types, dof, lengths, base_angle=0.0, probe_length=0.0, slide_angle=None):
    """Forward kinematics of a planar serial chain of revolute ('R') and
    prismatic ('P') joints. The chain starts at `base` and ends in a
    fixed probe.

    - Revolute joint i: dof[i] is the joint angle. lengths[i] is the link
      length along the new heading.
    - Prismatic joint i: lengths[i] is the housing length. dof[i] >= 0 is
      the extension, so the segment length is lengths[i] + dof[i].
    - The probe extends `probe_length` along the heading after the last
      joint.

    Returns the probe-tip pose (x, y, theta) and the joint positions: the
    base, then the end of each joint segment. The last position is the
    probe mount.
    """
    x, y = float(base[0]), float(base[1])
    phi = initial_heading(joint_types, base_angle, slide_angle)
    joints = [np.array([x, y])]
    for t, val, L in zip(joint_types, dof, lengths):
        if t == "R":
            phi += val
            seg = L
        else:
            seg = L + val
        x += seg * np.cos(phi)
        y += seg * np.sin(phi)
        joints.append(np.array([x, y]))
    tip = np.array([x + probe_length * np.cos(phi), y + probe_length * np.sin(phi), phi])
    return tip, joints


def body_segments(base, joint_types, dof, lengths, base_angle=0.0, probe_length=0.0, slide_angle=None):
    """Body segments at configuration `dof`: one for each joint (revolute
    link, or prismatic housing plus extension), then the probe."""
    tip, joints = fk(base, joint_types, dof, lengths, base_angle, probe_length, slide_angle)
    segs = [(joints[i], joints[i + 1]) for i in range(len(joint_types))]
    segs.append((joints[-1], tip[:2]))
    return segs


def jacobian(base, joint_types, dof, lengths, base_angle=0.0, probe_length=0.0, slide_angle=None):
    """Geometric Jacobian from joint velocities to the hybrid twist
    (vx, vy, omega) of the probe tip. A revolute column is
    (z x (p_tip - p_i), 1). A prismatic column is (segment direction, 0)."""
    n = len(joint_types)
    tip, joints = fk(base, joint_types, dof, lengths, base_angle, probe_length, slide_angle)
    p = tip[:2]
    J = np.zeros((3, n))
    phi = initial_heading(joint_types, base_angle, slide_angle)
    for i, (t, val) in enumerate(zip(joint_types, dof)):
        if t == "R":
            phi += val
            J[:, i] = [-(p[1] - joints[i][1]), p[0] - joints[i][0], 1.0]
        else:
            J[:, i] = [np.cos(phi), np.sin(phi), 0.0]
    return J


def fk_batch(base, joint_types, Q, lengths, base_angle=0.0, probe_length=0.0, slide_angle=None):
    """Vectorized `fk` over configurations Q (M, n). Returns points
    (M, n + 2, 2) (base, segment ends, probe tip) and headings (M,)."""
    Q = np.atleast_2d(Q)
    M = Q.shape[0]
    phi = np.full(M, initial_heading(joint_types, base_angle, slide_angle))
    pts = np.zeros((M, len(joint_types) + 2, 2))
    pts[:, 0] = base
    for i, (t, L) in enumerate(zip(joint_types, lengths)):
        if t == "R":
            phi = phi + Q[:, i]
            seg = np.full(M, L)
        else:
            seg = L + Q[:, i]
        pts[:, i + 1, 0] = pts[:, i, 0] + seg * np.cos(phi)
        pts[:, i + 1, 1] = pts[:, i, 1] + seg * np.sin(phi)
    pts[:, -1, 0] = pts[:, -2, 0] + probe_length * np.cos(phi)
    pts[:, -1, 1] = pts[:, -2, 1] + probe_length * np.sin(phi)
    return pts, phi


# Problem-level wrappers -----------------------------------------------------

def design_fk(problem, design, dof):
    return fk(problem.base, design["joint_types"], dof, design["lengths"], problem.base_angle,
              problem.probe_length, design.get("slide_angle"))


def design_segments(problem, design, dof):
    return body_segments(problem.base, design["joint_types"], dof, design["lengths"], problem.base_angle,
                         problem.probe_length, design.get("slide_angle"))


def design_jacobian(problem, design, dof):
    return jacobian(problem.base, design["joint_types"], dof, design["lengths"], problem.base_angle,
                    problem.probe_length, design.get("slide_angle"))


def total_length(problem, design):
    """Arm length at full extension, with the probe."""
    jt = design["joint_types"]
    return float(sum(design["lengths"]) + sum(s for t, s in zip(jt, design["strokes"]) if t == "P")
                 + problem.probe_length)


def _sigma_ratio(J, L):
    s = np.linalg.svd(to_dimensionless(J.T, L).T, compute_uv=False)
    return float(s.min() / (s.max() + 1e-12))


# --------------------------------------------------------------------------
# Design vector packing. In the fit, a revolute joint value is an angle and
# a prismatic joint value is a stroke fraction t in [0, 1] (q = t * stroke).
# Thus the stroke bound is a box bound, and the stroke is a design variable.
# --------------------------------------------------------------------------

def _layout(combo):
    lay = []
    for j, t in enumerate(combo):
        lay.append(("len", j))
        if t == "P":
            lay.append(("stroke", j))
    if combo[0] == "P":
        lay.append(("slide", 0))
    return lay


def pack_design(design):
    combo = design["joint_types"]
    z = []
    for kind, j in _layout(combo):
        z.append(design["lengths"][j] if kind == "len" else
                 design["strokes"][j] if kind == "stroke" else design["slide_angle"])
    return np.array(z, dtype=float)


def unpack_design(z, combo):
    n = len(combo)
    d = {"joint_types": combo, "lengths": np.zeros(n), "strokes": np.zeros(n), "slide_angle": None}
    for v, (kind, j) in zip(z, _layout(combo)):
        if kind == "len":
            d["lengths"][j] = v
        elif kind == "stroke":
            d["strokes"][j] = v
        else:
            d["slide_angle"] = float(v)
    return d


def _design_box(combo, lmax):
    lo, hi = [], []
    for kind, _ in _layout(combo):
        if kind == "slide":
            lo.append(-np.inf)
            hi.append(np.inf)
        else:
            lo.append(MIN_LENGTH)
            hi.append(lmax)
    return np.array(lo), np.array(hi)


def _joint_box(combo):
    lo = np.array([-np.inf if t == "R" else 0.0 for t in combo])
    hi = np.array([np.inf if t == "R" else 1.0 for t in combo])
    return lo, hi


def _t_to_q(T, design):
    """Stroke fractions -> prismatic extensions. Revolute values do not
    change."""
    Q = np.array(T, dtype=float, copy=True)
    for j, t in enumerate(design["joint_types"]):
        if t == "P":
            Q[..., j] = Q[..., j] * design["strokes"][j]
    return Q


class EvalCounter:
    """Counts residual-function evaluations, with Jacobian estimates."""

    def __init__(self):
        self.n = 0

    def wrap(self, f):
        def g(x):
            self.n += 1
            return f(x)
        return g


# --------------------------------------------------------------------------
# IK (initial per-target configurations of a start design)
# --------------------------------------------------------------------------

def solve_ik(problem, design, target_set, dof0=None, n_restarts=4, rng=None, counter=None):
    """Bounded least-squares IK for one target pose set. The residuals are
    the probe-tip position error and the heading distance from the pruned
    band. `dof0` and the result use stroke fractions for prismatic
    joints. Returns (t-form dof, cost)."""
    combo = design["joint_types"]
    rng = rng or np.random.default_rng(0)
    L = problem.length_scale
    lo, hi = _joint_box(combo)
    txy = target_set["xy"]
    iv = target_set["intervals"]

    def fun(T):
        tip, _ = design_fk(problem, design, _t_to_q(T, design))
        return np.array([(tip[0] - txy[0]) / L, (tip[1] - txy[1]) / L, heading_residual(tip[2], iv)])

    if counter is not None:
        fun = counter.wrap(fun)
    inits = [] if dof0 is None else [np.clip(np.asarray(dof0, float), lo, hi)]
    for _ in range(n_restarts):
        inits.append(np.array([rng.uniform(-np.pi, np.pi) if t == "R" else rng.uniform(0.1, 0.9) for t in combo]))
    best, best_cost = None, np.inf
    for T0 in inits:
        res = least_squares(fun, T0, bounds=(lo, hi), method="trf", xtol=1e-10, ftol=1e-10, max_nfev=50)
        c = float(np.sum(res.fun ** 2))
        if c < best_cost:
            best, best_cost = res.x, c
        if best_cost < 1e-12:
            break
    return best, best_cost


# --------------------------------------------------------------------------
# Evaluation of a fitted design (fit and Stage 6)
# --------------------------------------------------------------------------

def _transition_configs(Q, order, n_samples):
    """Linear joint-space interpolation between consecutive targets in
    `order`. Revolute angles are not wrapped, so the fit sets the
    direction of rotation. Returns (configs (M, n), pair index of each
    config)."""
    out, pair = [], []
    s = np.arange(1, n_samples + 1) / (n_samples + 1)
    for p, (a, b) in enumerate(zip(order[:-1], order[1:])):
        out.append(Q[a][None, :] + s[:, None] * (Q[b] - Q[a])[None, :])
        pair += [p] * n_samples
    if not out:
        return np.zeros((0, Q.shape[1])), []
    return np.concatenate(out), pair


def _body_points(problem, design, Q):
    pts, _ = fk_batch(problem.base, design["joint_types"], Q, design["lengths"], problem.base_angle,
                      problem.probe_length, design.get("slide_angle"))
    return pts


def _clearances(problem, design, Q, pts=None):
    pts = _body_points(problem, design, Q) if pts is None else pts
    return problem.segment_clearance(pts[:, :-1], pts[:, 1:])  # (M, n + 1)


def _outside(problem, pts):
    """Distance of each body point outside the workspace box (0 inside). The
    box is convex, so a segment between two inside points is inside."""
    xmin, ymin, xmax, ymax = problem.bounds
    return np.maximum.reduce([xmin - pts[..., 0], pts[..., 0] - xmax, ymin - pts[..., 1], pts[..., 1] - ymax,
                              np.zeros(pts.shape[:-1])])


def evaluate_design(problem, design, Q, target_sets, order, n_transition=CHECK_TRANSITION_SAMPLES):
    """Feasibility and quality of `design` at the per-target configurations
    Q (physical units, rows in `target_sets` order)."""
    L = problem.length_scale
    per = []
    for k, ts in enumerate(target_sets):
        tip, joints = design_fk(problem, design, Q[k])
        J = design_jacobian(problem, design, Q[k])
        clr = float(np.min(_clearances(problem, design, Q[k][None, :])))
        hr = heading_residual(tip[2], ts["intervals"])
        Jxy = J[:2]
        _, sv, Vt = np.linalg.svd(Jxy)
        rank = int(np.sum(sv > 1e-9 * max(sv.max(), 1e-12)))
        null = Vt[rank:]
        rot_null = bool(null.shape[0] > 0 and np.max(np.abs(J[2] @ null.T)) > 1e-6)
        band_width = max((hi - lo for lo, hi in ts["intervals"]), default=0.0)
        per.append({
            "index": ts["index"], "tip": tip, "joints": joints, "dof": Q[k],
            "pos_err": float(np.linalg.norm(tip[:2] - ts["xy"])),
            "heading": float(tip[2]), "heading_residual": float(hr),
            "heading_in_band": bool(abs(hr) <= HEADING_SLACK),
            "min_clearance": clr,
            "sigma_ratio": _sigma_ratio(J, L),
            "heading_adjustable": bool(rot_null and band_width > 0),
        })
    Qt, pair = _transition_configs(Q, order, n_transition)
    trans = []
    out_t = 0.0
    if len(Qt):
        pts_t = _body_points(problem, design, Qt)
        c = np.min(_clearances(problem, design, Qt, pts_t), axis=1)
        o = np.max(_outside(problem, pts_t), axis=1)
        out_t = float(o.max())
        for p, (a, b) in enumerate(zip(order[:-1], order[1:])):
            sel = np.array(pair) == p
            trans.append({"from": target_sets[a]["index"], "to": target_sets[b]["index"],
                          "min_clearance": float(c[sel].min()), "max_outside_workspace": float(o[sel].max())})
    out_k = float(np.max(_outside(problem, _body_points(problem, design, Q))))
    pos_ok = all(p["pos_err"] <= POS_TOL_FRAC * L for p in per)
    head_ok = all(p["heading_in_band"] for p in per)
    clr_targets = min(p["min_clearance"] for p in per)
    clr_trans = min((t["min_clearance"] for t in trans), default=np.inf)
    col_ok = clr_targets >= problem.link_radius and clr_trans >= problem.link_radius
    in_ws = max(out_k, out_t) <= 1e-6
    min_sr = min(p["sigma_ratio"] for p in per)
    return {
        "per_target": per, "transitions": trans,
        "reached_all": bool(pos_ok), "headings_ok": bool(head_ok), "collision_free": bool(col_ok),
        "inside_workspace": bool(in_ws), "max_outside_workspace": max(out_k, out_t),
        "feasible": bool(pos_ok and head_ok and col_ok and in_ws),
        "min_clearance_targets": float(clr_targets), "min_clearance_transitions": float(clr_trans),
        "min_sigma_ratio": float(min_sr),
        "score": float((1.0 - min_sr) + 0.01 * total_length(problem, design) / L),
    }


# --------------------------------------------------------------------------
# Stage 5: fit of one combination
# --------------------------------------------------------------------------

def target_order(problem, target_sets):
    """Nearest-neighbor tour. It starts at the target nearest to the base."""
    P = np.array([ts["xy"] for ts in target_sets])
    cur = np.asarray(problem.base, float)
    left = list(range(len(P)))
    order = []
    while left:
        i = min(left, key=lambda j: np.linalg.norm(P[j] - cur))
        order.append(i)
        left.remove(i)
        cur = P[i]
    return order


def _start_point(problem, combo, mode, rng, ctx, attempt):
    """Start design (and optional t-form configurations) for one fit."""
    n = len(combo)
    lmax = ctx["lmax"]
    if mode == "priors":
        design, dofs = design_prior(problem, combo, ctx["priors"], ctx["target_sets_all"])
        dofs = [dofs[ts["index"]] for ts in ctx["target_sets"]]
        if attempt > 0 or ctx["seed"] > 0:
            design["lengths"] = design["lengths"] * rng.uniform(0.8, 1.2, n)
            design["strokes"] = design["strokes"] * rng.uniform(0.8, 1.2, n)
            if design["slide_angle"] is not None:
                design["slide_angle"] += rng.uniform(-0.2, 0.2)
            dofs = [None if d is None else d + np.array([rng.normal(0, 0.2) if t == "R" else 0.0 for t in combo])
                    for d in dofs]
    elif mode == "random":
        design = {"joint_types": combo,
                  "lengths": rng.uniform(MIN_LENGTH, lmax, n),
                  "strokes": np.array([rng.uniform(MIN_LENGTH, lmax) if t == "P" else 0.0 for t in combo]),
                  "slide_angle": float(rng.uniform(-np.pi, np.pi)) if combo[0] == "P" else None}
        dofs = [None] * len(ctx["target_sets"])
    elif mode == "legacy_icr":
        from legacy.stage5_6 import _seed_lengths
        base = np.asarray(problem.base, float)
        t0 = ctx["target_sets"][0]["xy"]
        reach = float(np.linalg.norm(t0 - base))
        Ls = _seed_lengths(combo, ctx["icr_clusters"], base, reach, n)
        if ctx["seed"] > 0 or attempt > 0:
            Ls = Ls * rng.uniform(0.7, 1.3, n)
        design = {"joint_types": combo,
                  "lengths": np.array([L if t == "R" else 0.5 * L for t, L in zip(combo, Ls)]),
                  "strokes": np.array([L if t == "P" else 0.0 for t, L in zip(combo, Ls)]),
                  "slide_angle": (float(np.arctan2(*(t0 - base)[::-1]) + rng.uniform(-0.5, 0.5))
                                  if combo[0] == "P" else None)}
        dofs = [None] * len(ctx["target_sets"])
    else:
        raise ValueError(f"unknown start mode {mode!r}")
    design["lengths"] = np.clip(design["lengths"], MIN_LENGTH, lmax)
    design["strokes"] = np.array([np.clip(s, MIN_LENGTH, lmax) if t == "P" else 0.0
                                  for t, s in zip(combo, design["strokes"])])
    return design, dofs


def fit_combination(problem, combo, ctx, seed=0, mode="priors", n_restarts=N_RESTARTS):
    """Fits one joint combination to all targets. It tries up to
    `n_restarts` start points of the given mode and stops at the first
    feasible fit. Returns the best fit (feasible first, then lowest score
    or residual), with evaluation counts and wall-clock time."""
    t_start = time.perf_counter()
    rng = np.random.default_rng([seed, ALL_COMBINATIONS.index(combo), START_MODES.index(mode)])
    tsets, order = ctx["target_sets"], ctx["order"]
    K, n = len(tsets), len(combo)
    L = problem.length_scale
    infl = problem.inflation
    txy = np.array([ts["xy"] for ts in tsets])
    pairs = list(zip(order[:-1], order[1:]))
    s_fit = np.arange(1, FIT_TRANSITION_SAMPLES + 1) / (FIT_TRANSITION_SAMPLES + 1)
    dlo, dhi = _design_box(combo, ctx["lmax"])
    jlo, jhi = _joint_box(combo)
    nz = len(dlo)
    lo = np.concatenate([dlo, np.tile(jlo, K)])
    hi = np.concatenate([dhi, np.tile(jhi, K)])
    counter = EvalCounter()

    def residuals(x):
        design = unpack_design(x[:nz], combo)
        T = x[nz:].reshape(K, n)
        Q = _t_to_q(T, design)
        pts, phi = fk_batch(problem.base, combo, Q, design["lengths"], problem.base_angle,
                            problem.probe_length, design["slide_angle"])
        r_pos = (W_POS / L) * (pts[:, -1] - txy).ravel()
        r_head = W_HEAD * np.array([heading_residual(phi[k], tsets[k]["intervals"]) for k in range(K)])
        if pairs:
            Qt = np.concatenate([Q[a][None, :] + s_fit[:, None] * (Q[b] - Q[a])[None, :] for a, b in pairs])
            Qall = np.concatenate([Q, Qt])
        else:
            Qall = Q
        pts_all = _body_points(problem, design, Qall)
        clr = _clearances(problem, design, Qall, pts_all)
        r_col = (W_COL / L) * np.maximum(0.0, infl - clr).ravel()
        r_ws = (W_COL / L) * _outside(problem, pts_all).ravel()
        r_reg = np.array([W_REG * total_length(problem, design) / L])
        return np.concatenate([r_pos, r_head, r_col, r_ws, r_reg])

    f = counter.wrap(residuals)
    attempts = []
    best = None
    evals_to_feasible = None
    for attempt in range(n_restarts):
        design0, dofs0 = _start_point(problem, combo, mode, rng, {**ctx, "seed": seed}, attempt)
        T0 = np.zeros((K, n))
        for k, ts in enumerate(tsets):
            T0[k], _ = solve_ik(problem, design0, ts, dof0=dofs0[k], n_restarts=2 if dofs0[k] is not None else 4,
                                rng=rng, counter=counter)
        x0 = np.clip(np.concatenate([pack_design(design0), T0.ravel()]), lo, hi)
        res = least_squares(f, x0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=FIT_MAX_NFEV)
        design = unpack_design(res.x[:nz], combo)
        Q = _t_to_q(res.x[nz:].reshape(K, n), design)
        ev = evaluate_design(problem, design, Q, tsets, order)
        cand = {"combo": combo, "design": design, "Q": Q, "evaluation": ev, "feasible": ev["feasible"],
                "residual_cost": float(np.sum(res.fun ** 2)), "attempt": attempt}
        attempts.append({"attempt": attempt, "feasible": ev["feasible"], "residual_cost": cand["residual_cost"],
                         "evals_so_far": counter.n})
        key = (not cand["feasible"], ev["score"] if cand["feasible"] else cand["residual_cost"])
        if best is None or key < (not best["feasible"], best["evaluation"]["score"] if best["feasible"]
                                  else best["residual_cost"]):
            best = cand
        if ev["feasible"]:
            evals_to_feasible = counter.n
            break
    best.update({"n_evals": counter.n, "evals_to_feasible": evals_to_feasible,
                 "wall_time_s": time.perf_counter() - t_start, "attempts": attempts,
                 "seed": seed, "start_mode": mode,
                 "score": best["evaluation"]["score"] if best["feasible"] else np.inf})
    return best


# --------------------------------------------------------------------------
# Stage 5: ordered search
# --------------------------------------------------------------------------

def build_context(problem, target_sets, bounds, classes, priors, icr_clusters=None):
    feas = [ts for ts in target_sets if ts["feasible"]]
    lb = bounds["length"]["bound"] or problem.length_scale
    return {"target_sets": feas, "target_sets_all": target_sets, "order": target_order(problem, feas),
            "bounds": bounds, "classes": classes, "priors": priors,
            "lmax": LENGTH_BOUND_FACTOR * lb, "icr_clusters": icr_clusters or []}


def _exclusion_reason(combo, classes, final_class=None):
    """Reason why `combo` is not admissible in `final_class` (the class
    where the search stopped), or None if it is admissible."""
    sel = final_class or classes["selected_class"]
    if sel is not None and combo not in class_combinations(sel):
        return "class"
    if len(combo) < (classes["counting_bound"] or 1):
        return "counting_bound"
    if rotating_after_first(combo) < (classes["rotating_floor"] or 0):
        return "rotating_floor"
    return None


def search_mechanism(problem, ctx, seed=0, starts="priors", n_restarts=N_RESTARTS, minimality=True):
    """Stage 5: fits the admissible combinations, smallest joint count
    first, and stops at the first count with a feasible design. If the
    selected class has no feasible design, the search continues with the
    next covering class. Then it fits all combinations with one joint
    fewer, with the same budget, as minimality evidence. Returns the
    selected mechanism and the full search record."""
    t0 = time.perf_counter()
    fitted = {}
    fit_order = []
    cumulative = 0
    first_feasible_evals = None
    selected_n, selected_class = None, None
    for stage in ctx["classes"]["escalation"]:
        combos = stage["admissible"]
        for n in sorted({len(c) for c in combos}):
            for c in [c for c in combos if len(c) == n and c not in fitted]:
                r = fit_combination(problem, c, ctx, seed, starts, n_restarts)
                if r["feasible"] and first_feasible_evals is None:
                    first_feasible_evals = cumulative + r["evals_to_feasible"]
                cumulative += r["n_evals"]
                fitted[c] = r
                fit_order.append(c)
            if any(fitted[c]["feasible"] for c in combos if len(c) == n):
                selected_n, selected_class = n, stage["class"]
                break
        if selected_n is not None:
            break

    minimality_record = []
    if minimality and selected_n is not None and selected_n >= 2:
        for c in [c for c in ALL_COMBINATIONS if len(c) == selected_n - 1]:
            if c not in fitted:
                fitted[c] = fit_combination(problem, c, ctx, seed, starts, n_restarts)
            minimality_record.append({"combo": c, "feasible": fitted[c]["feasible"],
                                      "excluded_by": _exclusion_reason(c, ctx["classes"], selected_class),
                                      "fitted_in_search": c in fit_order})

    if selected_n is not None:
        pool = [fitted[c] for c in fit_order if len(c) == selected_n and fitted[c]["feasible"]]
        best = min(pool, key=lambda r: r["score"])
    else:
        best = min((fitted[c] for c in fit_order), key=lambda r: r["residual_cost"])
    nb = ctx["classes"]["counting_bound"] or 1
    return {
        "mechanism": best, "selected_n": selected_n, "selected_class": selected_class,
        "escalated": selected_class is not None and selected_class != ctx["classes"]["selected_class"],
        "fitted": fitted, "fit_order": fit_order,
        "first_feasible_evals": first_feasible_evals, "total_evals_search": cumulative,
        "wall_time_s": time.perf_counter() - t0, "seed": seed, "starts": starts,
        "minimality": minimality_record,
        "minimality_holds": (all(not m["feasible"] for m in minimality_record) if minimality_record else None),
        "counting_bound_respected": bool(selected_n is None or
                                         all(not fitted[c]["feasible"] for c in fitted if len(c) < nb)),
    }


# --------------------------------------------------------------------------
# Stage 6: validation
# --------------------------------------------------------------------------

def validate_mechanism(problem, mechanism, ctx):
    """Stage 6: independent check of a fitted mechanism.

    - At each target: reach, heading in the pruned band, clearance (with
      the probe), sigma ratio of the dimensionless Jacobian, and
      `heading_adjustable`. `heading_adjustable` is true when J has a
      null direction for the tip position that rotates the tip, and the
      pruned band has a width.
    - Clearance along each transition, with twice the sample density of
      the search check.
    - Bend and elbow: for each revolute joint after the first, at each
      target configuration, the distance to the nearest bend cluster.
      Also the distance to the nearest bend of the segment paths of that
      target.
    - Bound consistency with Stage 3.
    """
    design, Q = mechanism["design"], mechanism["Q"]
    combo = design["joint_types"]
    ev = evaluate_design(problem, design, Q, ctx["target_sets"], ctx["order"],
                         n_transition=2 * CHECK_TRANSITION_SAMPLES)
    clusters = [np.asarray(c["center"]) for c in ctx["priors"]["bend_clusters"]]
    elbows = []
    for k, p in enumerate(ev["per_target"]):
        pt = ctx["priors"]["per_target"][p["index"]]
        own = np.array([b for r in (pt["routes"] if pt else []) for b in r["bends"]]).reshape(-1, 2)
        for i in range(1, len(combo)):
            if combo[i] != "R":
                continue
            e = np.asarray(p["joints"][i])
            elbows.append({
                "target": p["index"], "joint": i, "position": e.tolist(),
                "to_nearest_cluster": float(min(np.linalg.norm(e - c) for c in clusters)) if clusters else None,
                "to_nearest_own_bend": float(np.min(np.linalg.norm(own - e, axis=1))) if len(own) else None,
            })
    b = ctx["bounds"]
    tl = total_length(problem, design)
    n_rot = rotating_after_first(combo)
    own_d = [e["to_nearest_own_bend"] for e in elbows if e["to_nearest_own_bend"] is not None]
    cl_d = [e["to_nearest_cluster"] for e in elbows if e["to_nearest_cluster"] is not None]
    return {
        **{k: v for k, v in ev.items() if k != "per_target"},
        "per_target": [{k: (v.tolist() if isinstance(v, np.ndarray) else
                            [np.asarray(j).tolist() for j in v] if k == "joints" else v)
                        for k, v in p.items()} for p in ev["per_target"]],
        "near_singular_targets": [p["index"] for p in ev["per_target"] if p["sigma_ratio"] < SINGULAR_RATIO],
        "elbows": elbows,
        "mean_elbow_to_own_bend": float(np.mean(own_d)) if own_d else None,
        "mean_elbow_to_cluster": float(np.mean(cl_d)) if cl_d else None,
        "bound_consistency": {
            "total_length_with_probe": tl,
            "length_bound": b["length"]["bound"],
            # the grid bound has fast-marching error (approximately 0.2% above
            # the tangent-arc-tangent length on one_obstacle), so allow two cells
            "length_grid_tol": 2.0 / problem.grid_resolution,
            "length_margin": None if b["length"]["bound"] is None else tl - b["length"]["bound"],
            "length_ok": bool(b["length"]["bound"] is None
                              or tl >= b["length"]["bound"] - 2.0 / problem.grid_resolution),
            "n_joints": len(combo),
            "counting_bound": b["counting"]["bound"],
            "counting_ok": bool(b["counting"]["bound"] is None or len(combo) >= b["counting"]["bound"]),
            "rotating_after_first": n_rot,
            "rotating_floor": b["min_link"]["rotating_floor"],
            "rotating_ok": bool(b["min_link"]["rotating_floor"] is None or n_rot >= b["min_link"]["rotating_floor"]),
        },
    }


def serialize_design(problem, design):
    jt = design["joint_types"]
    return {"joint_types": jt, "lengths": np.asarray(design["lengths"]).tolist(),
            "strokes": [float(s) if t == "P" else None for t, s in zip(jt, design["strokes"])],
            "slide_angle_deg": None if design.get("slide_angle") is None else float(np.rad2deg(design["slide_angle"])),
            "base_angle_deg": float(np.rad2deg(problem.base_angle)),
            "probe_length": problem.probe_length,
            "total_length_with_probe": total_length(problem, design)}


def summarize_fit(problem, r):
    ev = r["evaluation"]
    return {"combo": r["combo"], "feasible": r["feasible"], "score": None if not np.isfinite(r["score"]) else r["score"],
            "residual_cost": r["residual_cost"], "reached_all": ev["reached_all"], "headings_ok": ev["headings_ok"],
            "collision_free": ev["collision_free"], "inside_workspace": ev["inside_workspace"],
            "min_clearance_targets": ev["min_clearance_targets"],
            "min_clearance_transitions": ev["min_clearance_transitions"], "min_sigma_ratio": ev["min_sigma_ratio"],
            "max_pos_err": max(p["pos_err"] for p in ev["per_target"]),
            "n_evals": r["n_evals"], "evals_to_feasible": r["evals_to_feasible"],
            "wall_time_s": r["wall_time_s"], "attempts": r["attempts"],
            "design": serialize_design(problem, r["design"])}
