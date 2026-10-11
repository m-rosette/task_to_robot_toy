"""
E5: Ablations.

Changes one setting at a time around a preset. For each value, records the
bounds, the admissible set, and the Stage 5 selection over a few seeds:

- grid resolution (geodesic field and minimum-link grid)
- obstacle inflation (`clearance_margin` added to `link_radius`)
- heading tolerance width
- probe length
- characteristic length (conditioning and position tolerance)
- orientation rule (blend radius of the dead-band heading rule). Only the
  heading field changes, because no later stage uses it.

`run_geodesic_pipeline.py --field potential` runs the legacy-field ablation.

Usage (from src/):
    ../.venv/bin/python experiments/ablations.py [--problem one_obstacle] [--seeds 3] [--jobs J]
"""

import argparse
import dataclasses
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bounds import compute_bounds  # noqa: E402
from classes import classify_task  # noqa: E402
from geodesic_field import build_geodesic_field  # noqa: E402
from priors import build_priors  # noqa: E402
from problem import ToyProblem, PRESETS  # noqa: E402
from stage5_6 import build_context, search_mechanism  # noqa: E402
from targets import build_target_sets  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _with_tolerance(problem, tol_deg):
    tol = np.deg2rad(tol_deg)
    if problem.targets is None:
        return dataclasses.replace(problem, target_theta_tol=tol)
    return dataclasses.replace(problem, targets=tuple((x, y, th, tol) for x, y, th, _ in problem.targets))


SETTINGS = {
    "grid_resolution": [(v, lambda P, v=v: dataclasses.replace(P, grid_resolution=v)) for v in (10.0, 20.0, 40.0)],
    "clearance_margin": [(v, lambda P, v=v: dataclasses.replace(P, clearance_margin=v)) for v in (0.0, 0.05, 0.15)],
    "tolerance_deg": [(v, lambda P, v=v: _with_tolerance(P, v)) for v in (0.0, 5.0, 15.0, 40.0)],
    "probe_length": [(v, lambda P, v=v: dataclasses.replace(P, probe_length=v)) for v in (0.25, 0.5, 1.0)],
    "char_length": [(v, lambda P, v=v: dataclasses.replace(P, char_length=v)) for v in (None, 5.0, 20.0)],
}
BLEND_RADII = (1.0, 2.0, 4.0)


def _run(args):
    problem, seed = args
    ts = build_target_sets(problem)
    F = build_geodesic_field(problem, ts)
    b = compute_bounds(problem, ts)
    c = classify_task(problem, ts, b)
    ctx = build_context(problem, ts, b, c, build_priors(problem, ts, F))
    S = search_mechanism(problem, ctx, seed=seed)
    return {"selected_n": S["selected_n"], "selected": S["mechanism"]["combo"] if S["selected_n"] else None,
            "first_feasible_evals": S["first_feasible_evals"]}


def _stages_1_to_4(problem):
    ts = build_target_sets(problem)
    b = compute_bounds(problem, ts)
    c = classify_task(problem, ts, b)
    return {"c_k": b["counting"]["c_k"], "counting_bound": b["counting"]["bound"], "m_k": b["min_link"]["m_k"],
            "m": b["min_link"]["m"], "rotating_floor": b["min_link"]["rotating_floor"],
            "length_bound": b["length"]["bound"], "selected_class": c["selected_class"],
            "n_admissible": c["n_admissible"], "admissible": c["admissible"],
            "frac_removed": [t["frac_removed"] for t in ts]}


def main(problem_name="one_obstacle", seeds=3, jobs=None):
    base = ToyProblem(**PRESETS[problem_name])
    jobs = jobs or os.cpu_count() or 1
    rows = []
    jobs_list = []
    for name, variants in SETTINGS.items():
        for value, make in variants:
            P = make(base)
            row = {"setting": name, "value": value, **_stages_1_to_4(P)}
            rows.append(row)
            jobs_list += [(len(rows) - 1, P, s) for s in range(seeds)]
    with Pool(min(jobs, len(jobs_list))) as pool:
        results = pool.map(_run, [(P, s) for _, P, s in jobs_list])
    for (i, _, _), r in zip(jobs_list, results):
        rows[i].setdefault("search", []).append(r)
    for row in rows:
        row["selected_n_counts"] = dict(Counter(str(r["selected_n"]) for r in row["search"]))
        row["selected_counts"] = dict(Counter(str(r["selected"]) for r in row["search"]))
        print(f"{row['setting']:>16} = {str(row['value']):>5}: counting {row['counting_bound']} m {row['m']} "
              f"floor {row['rotating_floor']} length {row['length_bound']:.3f} admissible {row['n_admissible']:>2} "
              f"selected n {row['selected_n_counts']} {row['selected_counts']}")

    # orientation rule: only the heading field changes
    ts = build_target_sets(base)
    blend = []
    for br in BLEND_RADII:
        F = build_geodesic_field(base, ts, blend_radius=br, route_ties=False)
        om = np.concatenate([np.abs(tf["omega"][np.isfinite(tf["distance"])]) for tf in F["targets"] if tf])
        blend.append({"blend_radius": br, "mean_abs_omega": float(np.mean(om)),
                      "frac_cells_rotating": float(np.mean(om > 1e-3))})
        print(f"    blend_radius = {br}: mean |omega| {blend[-1]['mean_abs_omega']:.4f}, "
              f"rotating cells {blend[-1]['frac_cells_rotating']:.3f}")

    out_dir = os.path.join(ROOT, "reports", "geodesic", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"ablations_{problem_name}.json"), "w") as f:
        json.dump({"problem": problem_name, "seeds": seeds, "rows": rows, "orientation_rule": blend}, f,
                  indent=2, default=lambda o: None if o is None else float(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="one_obstacle", choices=list(PRESETS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    main(args.problem, args.seeds, args.jobs)
