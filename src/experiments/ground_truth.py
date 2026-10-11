"""
E4: Ground-truth recovery.

Makes tasks from known mechanisms. For each generator (joint combination and
design), it samples collision-free configurations and makes targets from
their probe-tip poses (heading band centered on the generator heading). Then
it runs Stages 1-4 (and optionally Stage 5) and checks that:

- the generator combination is admissible
- the counting bound is not more than the generator joint count
- the rotating-joint floor is not more than the generator revolute joints
  after the first
- the length bound is not more than the generator total length at full
  extension
- the predicted bend clusters are near the generator elbows. The prior
  location error is the distance from each generator revolute joint after
  the first, at each target configuration, to the nearest bend cluster.

Usage (from src/):
    ../.venv/bin/python experiments/ground_truth.py [--trials 4] [--search]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bounds import compute_bounds, serialize_bounds  # noqa: E402
from classes import classify_task, rotating_after_first  # noqa: E402
from geodesic_field import build_geodesic_field  # noqa: E402
from priors import build_priors  # noqa: E402
from problem import ToyProblem  # noqa: E402
from stage5_6 import (build_context, evaluate_design, fk_batch, search_mechanism, target_order,  # noqa: E402
                      total_length)
from targets import build_target_sets  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Generator mechanisms: (combo, lengths (R link / P housing), strokes, slide angle)
GENERATORS = [
    ("RR", [5.8, 4.2], [0, 0], None),
    ("PR", [2.0, 4.0], [3.0, 0], 0.55),
    ("RRR", [4.0, 3.0, 2.0], [0, 0, 0], None),
    ("RPR", [4.5, 1.0, 2.0], [0, 2.0, 0], None),
    ("PRR", [1.5, 3.5, 2.5], [2.5, 0, 0], 0.6),
]
OBSTACLES = ((5.0, 5.0, 1.0),)


def generate_task(combo, lengths, strokes, slide, K, tol, rng, obstacles=OBSTACLES, max_tasks=200):
    """Task that the generator design solves. The task passes the Stage 5
    feasibility check, with transitions."""
    for _ in range(max_tasks):
        P, design, Q = _sample_task(combo, lengths, strokes, slide, K, tol, rng, obstacles)
        if len(Q) < K:
            continue
        ts = build_target_sets(P)
        if not all(t["feasible"] for t in ts):
            continue
        if evaluate_design(P, design, Q, ts, target_order(P, ts))["feasible"]:
            return P, design, Q
    raise RuntimeError(f"no feasible task generated for {combo}")


def _sample_task(combo, lengths, strokes, slide, K, tol, rng, obstacles, max_tries=20000):
    P0 = ToyProblem(obstacles=obstacles)
    design = {"joint_types": combo, "lengths": np.array(lengths, float), "strokes": np.array(strokes, float),
              "slide_angle": slide}
    targets, configs = [], []
    signs = rng.choice([-1.0, 1.0], len(combo))
    for _ in range(max_tries):
        if len(targets) == K:
            break
        # each elbow bends in one direction for the full task (one assembly mode), up to 90 deg
        q = np.array([signs[i] * rng.uniform(0.0, np.pi / 2) if t == "R" else rng.uniform(0, s)
                      for i, (t, s) in enumerate(zip(combo, design["strokes"]))])
        if combo[0] == "R":
            q[0] = rng.uniform(0.0, np.pi / 2)  # point into the workspace
        pts, phi = fk_batch(P0.base, combo, q[None, :], design["lengths"], P0.base_angle, P0.probe_length, slide)
        tip = pts[0, -1]
        if not (0.5 < tip[0] < 9.5 and 0.5 < tip[1] < 9.5):
            continue
        if np.min(P0.segment_clearance(pts[0, :-1], pts[0, 1:])) <= P0.inflation:
            continue
        if any(np.linalg.norm(tip - np.array(t[:2])) < 1.0 for t in targets):
            continue
        targets.append((float(tip[0]), float(tip[1]), float(phi[0]), tol))
        configs.append(q)
    P = ToyProblem(obstacles=obstacles, targets=tuple(targets))
    return P, design, np.array(configs)


def run_trial(combo, lengths, strokes, slide, K, tol, seed, do_search):
    rng = np.random.default_rng(seed)
    P, design, Q = generate_task(combo, lengths, strokes, slide, K, tol, rng)
    ts = build_target_sets(P)
    b = compute_bounds(P, ts)
    c = classify_task(P, ts, b)
    F = build_geodesic_field(P, ts)
    pr = build_priors(P, ts, F)
    pts, _ = fk_batch(P.base, combo, Q, design["lengths"], P.base_angle, P.probe_length, slide)
    elbows = np.array([pts[k, i] for k in range(len(Q)) for i in range(1, len(combo)) if combo[i] == "R"])
    centers = np.array([cl["center"] for cl in pr["bend_clusters"]]).reshape(-1, 2)
    err = (np.linalg.norm(elbows[:, None, :] - centers[None, :, :], axis=-1).min(axis=1)
           if len(elbows) and len(centers) else np.array([]))
    out = {
        "combo": combo, "seed": seed, "K": len(P.targets_list), "tol_deg": float(np.rad2deg(tol)),
        "generating_total_length": total_length(P, design),
        "admissible": combo in c["admissible"] or any(combo in e["admissible"] for e in c["escalation"]),
        "selected_class": c["selected_class"], "admissible_list": c["admissible"],
        "counting_bound": b["counting"]["bound"], "counting_ok": b["counting"]["bound"] <= len(combo),
        "rotating_floor": b["min_link"]["rotating_floor"],
        "rotating_ok": b["min_link"]["rotating_floor"] <= rotating_after_first(combo),
        "length_bound": b["length"]["bound"],
        "length_ok": b["length"]["bound"] <= total_length(P, design) + 2.0 / P.grid_resolution,
        "prior_location_error": err.tolist(),
        "prior_location_error_median": float(np.median(err)) if len(err) else None,
        "bounds": serialize_bounds(b),
    }
    if do_search:
        ctx = build_context(P, ts, b, c, pr)
        S = search_mechanism(P, ctx, seed=seed)
        out["search"] = {"selected_n": S["selected_n"], "selected": S["mechanism"]["combo"] if S["selected_n"] else None,
                         "selected_n_le_generating": S["selected_n"] is not None and S["selected_n"] <= len(combo)}
    return out


def main(trials=4, K=5, tol_deg=20.0, do_search=False):
    rows = []
    for g in GENERATORS:
        for t in range(trials):
            r = run_trial(*g, K=K, tol=np.deg2rad(tol_deg), seed=t, do_search=do_search)
            rows.append(r)
            print(f"{r['combo']:>4} seed {t}: admissible={r['admissible']} counting {r['counting_bound']} "
                  f"floor {r['rotating_floor']} length {r['length_bound']:.2f}<={r['generating_total_length']:.2f} "
                  f"prior err {r['prior_location_error_median']}"
                  + (f" | search n={r['search']['selected_n']} {r['search']['selected']}" if do_search else ""))
    summary = {
        "n_trials": len(rows),
        "admissible_rate": float(np.mean([r["admissible"] for r in rows])),
        "bounds_ok_rate": float(np.mean([r["counting_ok"] and r["rotating_ok"] and r["length_ok"] for r in rows])),
        "median_prior_location_error": float(np.median([e for r in rows for e in r["prior_location_error"]]))
        if any(r["prior_location_error"] for r in rows) else None,
    }
    if do_search:
        summary["search_selected_n_le_generating_rate"] = float(np.mean([r["search"]["selected_n_le_generating"]
                                                                         for r in rows]))
    print(json.dumps(summary, indent=2))
    out_dir = os.path.join(ROOT, "reports", "geodesic", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump({"summary": summary, "trials": rows}, f, indent=2, default=float)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--targets", type=int, default=5)
    ap.add_argument("--tol", type=float, default=20.0, help="heading tolerance (deg)")
    ap.add_argument("--search", action="store_true", help="also run Stage 5 on each task")
    args = ap.parse_args()
    main(args.trials, args.targets, args.tol, args.search)
