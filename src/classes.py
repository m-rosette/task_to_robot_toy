"""
Stage 4a: Motion-class restriction.

Motion classes in SE(2), anchored at the fixed base (`fk` places the first
joint there), from smallest to largest:

| Class              | Coset test over all targets                                   | Joint combinations |
|--------------------|---------------------------------------------------------------|--------------------|
| line_translation   | positions collinear with the base, one common heading in every band | P            |
| rotation           | positions equidistant from the base, heading - bearing common | R                  |
| planar_translation | one common heading inside every pruned band                  | P, PP, PPP         |
| se2                | none of the above                                             | any with an R      |

A joint combination is admissible when it belongs to the chosen class, has
at least the counting bound's number of joints, and has at least the
rotating-joint floor's number of revolute joints after its first joint.
The *smallest covering class* is the smallest class whose coset test covers
every target; the *selected class* is the first class, from the smallest
covering one upward, whose admissible list is nonempty. (For a single
target every class covers it, and the obstacle's rotating-joint floor then
empties the three smaller classes, which is how `one_obstacle` lands in
SE(2) with 8 of 14 combinations.)

Note on realizability: with straight links and no fixed bends, every
prismatic joint after the first slides along the same line as the first,
so a prismatic-only chain reaches a ray from the base with its heading
along that ray. Prismatic-only combinations therefore realize
line_translation with heading = bearing, never planar translation. The
class test reports that planar translation *covers* a common-heading task;
the search (`stage5_6.search_mechanism`) then escalates to the next class
when no admissible combination of the selected class is feasible.
"""

import itertools

import numpy as np

from problem import wrap_angle
from targets import SAMPLE_STEP, heading_residual


N_MAX = 3
ALL_COMBINATIONS = ["".join(p) for n in range(1, N_MAX + 1) for p in itertools.product("RP", repeat=n)]
CLASS_ORDER = ("line_translation", "rotation", "planar_translation", "se2")
CLASS_DIM = {"line_translation": 1, "rotation": 1, "planar_translation": 2, "se2": 3}
COSET_POS_TOL_FRAC = 0.01  # position tolerance of the coset tests, as a fraction of length_scale


def class_combinations(name):
    if name == "line_translation":
        return ["P"]
    if name == "rotation":
        return ["R"]
    if name == "planar_translation":
        return [c for c in ALL_COMBINATIONS if set(c) == {"P"}]
    if name == "se2":
        return [c for c in ALL_COMBINATIONS if "R" in c]
    raise ValueError(name)


def rotating_after_first(combo):
    return combo[1:].count("R")


def admissible(name, counting_bound, rotating_floor):
    nb = counting_bound or 1
    fl = rotating_floor or 0
    return [c for c in class_combinations(name) if len(c) >= nb and rotating_after_first(c) >= fl]


def _heading_grid():
    return np.arange(-np.pi, np.pi, SAMPLE_STEP)


def _max_common(interval_lists, offsets=None):
    """Largest number of interval lists containing one common angle, and
    that angle. `offsets` shifts each list (used for heading - bearing)."""
    grid = _heading_grid()
    best, best_h = 0, None
    for h in grid:
        n = 0
        for i, iv in enumerate(interval_lists):
            shift = 0.0 if offsets is None else offsets[i]
            if heading_residual(h + shift, iv) == 0.0:
                n += 1
        if n > best:
            best, best_h = n, float(h)
    return best, best_h


def _coverage_planar(feas):
    n, h = _max_common([ts["intervals"] for ts in feas])
    return {"covered": n, "common_heading_deg": None if h is None else float(np.rad2deg(h))}


def _coverage_line(feas, base, tol):
    """Greedy over the line direction: try each target's bearing (and the
    principal direction of all of them) as the line through the base."""
    rel = np.array([ts["xy"] - base for ts in feas])
    dirs = [np.arctan2(r[1], r[0]) for r in rel]
    M = rel.T @ rel
    w, V = np.linalg.eigh(M)
    dirs.append(float(np.arctan2(V[1, -1], V[0, -1])))
    best = {"covered": 0, "direction_deg": None, "common_heading_deg": None}
    for phi in dirs:
        u = np.array([np.cos(phi), np.sin(phi)])
        perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        members = [ts for ts, d in zip(feas, perp) if d <= tol]
        if len(members) <= best["covered"]:
            continue
        n, h = _max_common([ts["intervals"] for ts in members])
        if n > best["covered"]:
            best = {"covered": n, "direction_deg": float(np.rad2deg(phi)),
                    "common_heading_deg": None if h is None else float(np.rad2deg(h))}
    return best


def _coverage_rotation(feas, base, tol):
    """Greedy over the common offset delta = heading - bearing: for each
    delta, the targets whose band contains bearing + delta, then the largest
    group of those with a common radius (within `tol`)."""
    rel = np.array([ts["xy"] - base for ts in feas])
    bearings = np.arctan2(rel[:, 1], rel[:, 0])
    radii = np.linalg.norm(rel, axis=1)
    best = {"covered": 0, "radius": None, "offset_deg": None}
    for delta in _heading_grid():
        idx = [i for i, ts in enumerate(feas) if heading_residual(bearings[i] + delta, ts["intervals"]) == 0.0]
        if len(idx) <= best["covered"]:
            continue
        r = np.sort(radii[idx])
        j = 0
        top, top_r = 0, None
        for i in range(len(r)):
            while r[i] - r[j] > 2 * tol:
                j += 1
            if i - j + 1 > top:
                top, top_r = i - j + 1, float(np.median(r[j:i + 1]))
        if top > best["covered"]:
            best = {"covered": top, "radius": top_r, "offset_deg": float(np.rad2deg(wrap_angle(delta)))}
    return best


def classify_task(problem, target_sets, bounds):
    """Stage 4a: class coverage, the smallest covering class, the selected
    class, its admissible combinations, rho, and the escalation order the
    search follows if the selected class has no feasible design."""
    feas = [ts for ts in target_sets if ts["feasible"]]
    K = len(feas)
    base = np.asarray(problem.base, float)
    tol = COSET_POS_TOL_FRAC * problem.length_scale
    nb = bounds["counting"]["bound"]
    fl = bounds["min_link"]["rotating_floor"]

    coverage = {
        "line_translation": _coverage_line(feas, base, tol),
        "rotation": _coverage_rotation(feas, base, tol),
        "planar_translation": _coverage_planar(feas),
        "se2": {"covered": K},
    }
    covers = [c for c in CLASS_ORDER if coverage[c]["covered"] == K]
    smallest = covers[0] if covers else "se2"
    adm = {c: admissible(c, nb, fl) for c in CLASS_ORDER}

    escalation = []
    seen = set()
    for c in covers:
        new = [x for x in adm[c] if x not in seen]
        if new:
            escalation.append({"class": c, "admissible": new})
            seen.update(new)
    selected = escalation[0]["class"] if escalation else None
    sel_adm = adm[selected] if selected else []
    return {
        "K": K,
        "coverage": coverage,
        "covering_classes": covers,
        "smallest_covering_class": smallest,
        "selected_class": selected,
        "admissible": sel_adm,
        "excluded": [c for c in ALL_COMBINATIONS if c not in sel_adm],
        "n_admissible": len(sel_adm),
        "n_total": len(ALL_COMBINATIONS),
        "rho": len(sel_adm) / len(ALL_COMBINATIONS),
        "admissible_by_class": adm,
        "escalation": escalation,
        "counting_bound": nb,
        "rotating_floor": fl,
    }
