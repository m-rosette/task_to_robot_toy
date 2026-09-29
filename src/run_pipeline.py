"""
Runs the full toy pipeline, Stage 1 -> Stage 6, and saves figures to
../images/ and a JSON report to ../reports/.

Usage:
    python run_pipeline.py [--problem one_obstacle|two_obstacles] [--seeds N] [--jobs J]

Stage 5 is run once per seed (a multi-start: seeds perturb each candidate's
starting design and IK restarts) in parallel; the report records the
per-seed selections so stability is visible, and the mechanism that gets
validated and plotted is the lowest-score feasible design at the most
frequently selected joint count.
"""

import os
import json
import argparse
from collections import Counter
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from problem import ToyProblem, PRESETS
from field import bow_signs, desired_velocity, conservative_potential
from stage1_2 import build_exact_generators
from stage3_4 import analyze_lie_algebra, classify_primitives
from stage5_6 import assemble_mechanism, validate_mechanism, fk

ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGES = os.path.join(ROOT, "images")
REPORTS = os.path.join(ROOT, "reports")
os.makedirs(IMAGES, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)


def _add_obstacle_patches(ax, problem, label=None, **kwargs):
    """Draw one Circle per obstacle in `problem.obstacles`; `label` (if
    given) is only attached to the first patch so a legend doesn't get a
    duplicate "obstacle" entry per obstacle."""
    for i, (cx, cy, r) in enumerate(problem.obstacles):
        ax.add_patch(Circle((cx, cy), r, label=(label if i == 0 else None), **kwargs))


def plot_potential_field(problem, path, n_grid=80):
    """Diagnostic view of the navigation field itself: a heatmap of the
    conservative (non-swirled) potential -- the part of the field that's an
    actual gradient -- plus a quiver of the full `desired_velocity`
    (including the non-conservative swirl term used to route around each
    obstacle)."""
    xmin, ymin, xmax, ymax = problem.bounds
    signs = bow_signs(problem)
    xs = np.linspace(xmin, xmax, n_grid)
    ys = np.linspace(ymin, ymax, n_grid)
    XX, YY = np.meshgrid(xs, ys)
    U = np.zeros_like(XX)
    for i in range(n_grid):
        for j in range(n_grid):
            U[j, i] = conservative_potential(np.array([XX[j, i], YY[j, i]]), problem)

    fig, ax = plt.subplots(figsize=(6, 6))
    cf = ax.contourf(XX, YY, np.log1p(U), levels=30, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="log(1 + conservative potential)")

    qn = 22
    qxs = np.linspace(xmin, xmax, qn)
    qys = np.linspace(ymin, ymax, qn)
    QX, QY = np.meshgrid(qxs, qys)
    QU, QV = np.zeros_like(QX), np.zeros_like(QY)
    for i in range(qn):
        for j in range(qn):
            p = np.array([QX[j, i], QY[j, i]])
            if problem.obstacle_clearance(p) < 0:
                continue
            v = desired_velocity(p, problem, signs)
            n = np.linalg.norm(v) + 1e-9
            QU[j, i], QV[j, i] = v[0] / n, v[1] / n
    ax.quiver(QX, QY, QU, QV, color="white", alpha=0.8, scale=30, width=0.003)

    _add_obstacle_patches(ax, problem, edgecolor="red", facecolor="none", lw=2)
    ax.plot(*problem.base, "ks", ms=10, label="base")
    ax.plot(*problem.target, "r*", ms=16, label="target")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.legend(loc="upper left")
    ax.set_title("Navigation field: conservative potential (heatmap)\n"
                 "+ desired velocity direction incl. swirl (arrows)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_top_k_mechanisms(problem, ranked_mechanisms, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    _add_obstacle_patches(ax, problem, color="0.85", zorder=0)
    colors = plt.cm.tab10.colors
    for rank, m in enumerate(ranked_mechanisms):
        _draw_body(ax, m, color=colors[rank % len(colors)], lw=2.5 if rank == 0 else 1.5,
                   alpha=1.0 if rank == 0 else 0.7,
                   label_prefix=f"#{rank + 1} {''.join(m['joint_types'])} (score={m['score']:.3f})")
    ax.plot(*problem.base, "ks", ms=10)
    ax.plot(*problem.target, "r*", ms=16)
    xmin, ymin, xmax, ymax = problem.bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Stage 5: top-{len(ranked_mechanisms)} feasible candidates at the selected "
                 f"joint count\n(target pose; dashed = prismatic rail)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _serialize_mechanism(m, validation=None):
    out = {
        "joint_types": "".join(m["joint_types"]),
        "n_links": m["n_links"],
        "lengths": np.asarray(m["lengths"]).tolist(),
        "base_angle_deg": float(np.rad2deg(m["base_angle"])),
        "tool_offset_deg": float(np.rad2deg(m["tool_offset"])),
        "home_dof": np.asarray(m["home_dof"]).tolist(),
        "achieved_ee": np.asarray(m["achieved_ee"]).tolist(),
        "joints": [np.asarray(j).tolist() for j in m["joints"]],
        "feasible": m["feasible"],
        "score": m["score"],
        "pos_err": m["pos_err"],
        "orient_err": m["orient_err"],
        "heading_violation": m["heading_violation"],
        "min_clearance_at_target": m["min_clearance_at_target"],
        "mean_alignment": m["mean_alignment"],
        "align_reach_fraction": m["align_reach_fraction"],
        "mean_collision_cost": m["mean_collision_cost"],
        "seed": m.get("seed"),
    }
    if validation is not None:
        out["validation"] = {k: v for k, v in validation.items() if k != "ee_position_samples"}
    return out


def _draw_body(ax, m, color=None, lw=3, alpha=1.0, label_prefix=None):
    """Draw revolute links as solid lines and prismatic rails (full stroke)
    as dashed lines, with the carriage/joint positions as dots."""
    for i, (a, b) in enumerate(m["segments"]):
        is_p = m["joint_types"][i] == "P"
        ax.plot([a[0], b[0]], [a[1], b[1]], "--" if is_p else "-",
                color=color or ("tab:green" if is_p else "k"), lw=lw, alpha=alpha,
                label=None if label_prefix is None else
                (f"{label_prefix}" if i == 0 else None))
    xs = [j[0] for j in m["joints"]]; ys = [j[1] for j in m["joints"]]
    ax.plot(xs, ys, "o", color=color or "k", ms=5, alpha=alpha)


def plot_twist_field(stage2_results, problem, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    _add_obstacle_patches(ax, problem, color="0.85", zorder=0)
    for key, res in stage2_results.items():
        c = res["center"]
        g = res["generators"][0]
        v = g[:2]
        vn = v / (np.linalg.norm(v) + 1e-9)
        color = "tab:red" if res["intrinsic_dim"] >= 2 else "tab:blue"
        ax.arrow(c[0], c[1], 0.3 * vn[0], 0.3 * vn[1],
                  head_width=0.08, color=color, alpha=0.8)
    ax.plot(*problem.base, "ks", ms=10)
    ax.plot(*problem.target, "r*", ms=16)
    xmin, ymin, xmax, ymax = problem.bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title("Stage 1-2: dominant local generator field\n"
                  "(blue = rank-1 cell, red = rank>=2 cell)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_primitives(problem, stage4_result, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    _add_obstacle_patches(ax, problem, color="0.85", zorder=0)
    for r in stage4_result["records"]:
        c = r["center"]
        if r["kind"] == "prismatic":
            d = np.array(r["params"]["direction"])
            ax.arrow(c[0], c[1], 0.3 * d[0], 0.3 * d[1],
                      head_width=0.07, color="tab:green")
        else:
            ax.plot(*c, "o", color="tab:orange", ms=4)
    for cl in stage4_result["icr_clusters"]:
        if cl["center"] is None:
            continue  # near-prismatic cluster: axis is effectively at infinity
        ax.plot(*cl["center"], "P", color="tab:red", ms=14,
                label=f"ICR cluster (n={cl['n_members']}, spread={cl['icr_spread']:.1f})")
    ax.plot(*problem.base, "ks", ms=10)
    ax.plot(*problem.target, "r*", ms=16)
    xmin, ymin, xmax, ymax = problem.bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    ax.set_title("Stage 4: classified primitives\n"
                  "(green = prismatic direction, orange dot = local ICR)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism(problem, mechanism, validation, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    _add_obstacle_patches(ax, problem, color="0.85", zorder=0)
    ee_samp = validation["ee_position_samples"]
    ax.plot(ee_samp[:, 0], ee_samp[:, 1], ".", color="tab:purple",
            ms=1.5, alpha=0.3, label="sampled reachable EE positions")
    joint_types = mechanism["joint_types"]
    for i, (a, b) in enumerate(mechanism["segments"]):
        is_prismatic = joint_types[i] == "P"
        ax.plot([a[0], b[0]], [a[1], b[1]],
                "--" if is_prismatic else "-",
                color="tab:green" if is_prismatic else "k", lw=3,
                label="prismatic rail" if is_prismatic else "revolute link")
    joints = mechanism["joints"]
    xs = [j[0] for j in joints]; ys = [j[1] for j in joints]
    ax.plot(xs, ys, "o", color="k", ms=6)
    ax.plot(*problem.base, "ks", ms=10)
    ax.plot(*problem.target, "r*", ms=16)
    xmin, ymin, xmax, ymax = problem.bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=8)
    ax.set_title(f"Stage 5-6: selected {''.join(joint_types)} mechanism "
                 f"(fewest joints that reach the target collision-free)\n"
                 f"field realizability={validation['mean_generator_alignment']:.2f}, "
                 f"field poses reachable={validation['field_reach_fraction']:.2f}", fontsize=9)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


_WORKER = {}


def _stage5_worker(seed):
    w = _WORKER
    return assemble_mechanism(w["problem"], w["stage3"], w["stage4"], w["stage2"], seed=seed)


def _run_stage5_seeds(problem, stage3_result, stage4_result, stage2_results, seeds, jobs):
    _WORKER.update(problem=problem, stage3=stage3_result, stage4=stage4_result, stage2=stage2_results)
    if jobs > 1 and len(seeds) > 1:
        with Pool(min(jobs, len(seeds))) as pool:  # fork start method: workers inherit _WORKER
            return pool.map(_stage5_worker, seeds)
    return [_stage5_worker(s) for s in seeds]


def _seed_summary(results):
    per_seed = [{"seed": r["seed"], "selected_n": r["selected_n"],
                 "selected": "".join(r["joint_types"]), "score": r["score"],
                 "feasible_by_n": {str(n): v["feasible"] for n, v in r["per_n"].items()}}
                for r in results]
    return {
        "n_seeds": len(results),
        "selected_n_counts": dict(Counter(str(r["selected_n"]) for r in results)),
        "selected_topology_counts": dict(Counter("".join(r["joint_types"]) for r in results)),
        "feasible_topology_counts": dict(Counter(
            jt for r in results for v in r["per_n"].values() for jt in v["feasible"])),
        "per_seed": per_seed,
    }


def main(problem_name="one_obstacle", seeds=10, jobs=None):
    def img(name):
        return os.path.join(IMAGES, f"{name}.png")

    def rpt(name):
        return os.path.join(REPORTS, f"{name}.json")

    jobs = jobs or os.cpu_count() or 1
    problem = ToyProblem(**PRESETS[problem_name])

    print(f"Problem preset: {problem_name}")
    print("Plotting navigation field...")
    plot_potential_field(problem, img("00_navigation_field"))

    print("Stage 1-2: evaluating exact navigation field on a grid...")
    raw_stage2_results, meta = build_exact_generators(problem, n_cells=14)

    print("Stage 3: Lie bracket / involutivity analysis...")
    stage3_result = analyze_lie_algebra(raw_stage2_results, meta)
    stage2_results = stage3_result["closed_stage2_results"]
    coverage_locs = np.array([res["center"] for res in stage2_results.values()])
    print("  " + stage3_result["summary"])

    plot_twist_field(stage2_results, problem, img("02_twist_field"))

    print("Stage 4: classifying kinematic primitives...")
    stage4_result = classify_primitives(stage2_results, stage3_result,
                                        length_scale=problem.length_scale)
    plot_primitives(problem, stage4_result, img("03_primitives"))

    seed_list = list(range(seeds))
    print(f"Stage 5: joint-count / topology search over {len(seed_list)} seeds ({jobs} jobs)...")
    results = _run_stage5_seeds(problem, stage3_result, stage4_result, stage2_results, seed_list, jobs)
    seed_summary = _seed_summary(results)

    ns = [r["selected_n"] for r in results if r["selected_n"] is not None]
    modal_n = Counter(ns).most_common(1)[0][0] if ns else None
    pool = [r for r in results if r["selected_n"] == modal_n] if modal_n is not None else results
    mechanism = min(pool, key=lambda r: r["score"])
    print(f"  selected joint count per seed: {seed_summary['selected_n_counts']}; "
          f"topologies: {seed_summary['selected_topology_counts']}")
    print(f"  reported mechanism: {''.join(mechanism['joint_types'])} from seed {mechanism['seed']}")

    print("Stage 6: validating synthesized mechanism...")
    validation = validate_mechanism(problem, mechanism, coverage_locs, stage2_results,
                                    stage3_result=stage3_result)
    plot_mechanism(problem, mechanism, validation, img("04_mechanism"))

    print(f"Validating top-{len(mechanism['top_k_mechanisms'])} candidate mechanisms (seed {mechanism['seed']})...")
    top_k_serialized = []
    for rank, cand in enumerate(mechanism["top_k_mechanisms"], start=1):
        cand_validation = validate_mechanism(problem, cand, coverage_locs, stage2_results,
                                             stage3_result=stage3_result)
        top_k_serialized.append(_serialize_mechanism(cand, cand_validation))
        print(f"  #{rank} {''.join(cand['joint_types'])}: score={cand['score']:.4f} "
              f"feasible={cand['feasible']} "
              f"realizability={cand_validation['mean_generator_alignment']:.2f} "
              f"min_clearance={cand['min_clearance_at_target']:.2f}")
    plot_top_k_mechanisms(problem, mechanism["top_k_mechanisms"], img("05_top_k_mechanisms"))
    with open(rpt("top_mechanisms"), "w") as f:
        json.dump(top_k_serialized, f, indent=2)

    report = {
        "problem": {
            "preset": problem_name,
            "bounds": problem.bounds, "base": problem.base,
            "target": problem.target, "target_theta_deg": float(np.rad2deg(problem.target_theta)),
            "target_theta_tol_deg": float(np.rad2deg(problem.target_theta_tol)),
            "obstacles": problem.obstacles,
            "length_scale": problem.length_scale,
        },
        "stage2_summary": {
            "n_cells_with_data": len(stage2_results),
            "n_cells_rank_ge_2": sum(1 for r in stage2_results.values() if r["intrinsic_dim"] >= 2),
        },
        "stage3_summary": {
            "dof_lower_bound": stage3_result["dof_lower_bound"],
            "local_rank_histogram": stage3_result["local_rank_histogram"],
            "n_multi_generator_cells": stage3_result["n_multi_generator_cells"],
            "n_commuting": stage3_result["n_commuting"],
            "n_new_direction": stage3_result["n_new_direction"],
            "n_new_direction_low_confidence": stage3_result["n_new_direction_low_confidence"],
            "n_no_estimate": stage3_result["n_no_estimate"],
            "n_closure_promoted": stage3_result["n_closure_promoted"],
            "narrative": stage3_result["summary"],
        },
        "stage4_summary": {
            "n_prismatic": stage4_result["n_prismatic"],
            "n_revolute": stage4_result["n_revolute"],
            "icr_clusters": stage4_result["icr_clusters"],
        },
        "stage5_mechanism": {
            **_serialize_mechanism(mechanism),
            "selected_n": mechanism["selected_n"],
            "dof_lower_bound": mechanism["dof_lower_bound"],
            "lower_bound_consistent": mechanism["lower_bound_consistent"],
            "per_n": {str(n): v for n, v in mechanism["per_n"].items()},
            "candidates_tried": mechanism["candidates_tried"],
            "prefilter_scores": mechanism["prefilter_scores"],
            "top_k_mechanisms_file": "top_mechanisms.json",
        },
        "stage5_seed_summary": seed_summary,
        "stage6_validation": {k: v for k, v in validation.items() if k != "ee_position_samples"},
    }
    with open(rpt("report"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: report[k] for k in ("stage3_summary", "stage5_seed_summary", "stage6_validation")},
                     indent=2)[:4000])
    print(f"\nFigures written to {IMAGES}\nReport written to {REPORTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="one_obstacle", choices=sorted(PRESETS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    main(args.problem, args.seeds, args.jobs)
