"""
Runs the geodesic, reach-only pipeline (Stages 1-6). Writes figures to
../images/geodesic/<run>/ and JSON reports to ../reports/geodesic/<run>/.
<run> is the problem name, with the start mode and the field as suffixes
when they are not the defaults (priors, geodesic).

Usage:
    python run_geodesic_pipeline.py [--problem NAME] [--seeds N] [--jobs J]
                                    [--starts priors|random|legacy_icr]
                                    [--field geodesic|potential]

Stage 5 runs once for each seed, in parallel. The seeds change the start
points and the IK restarts, not the task. The runner validates and plots the
lowest-score feasible design at the most frequent selected joint count.
`run_pipeline.py` runs the legacy bracket-based pipeline.
"""

import argparse
import dataclasses
import glob
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, Wedge  # noqa: E402

from problem import ToyProblem, PRESETS  # noqa: E402
from targets import build_target_sets, serialize_target_sets  # noqa: E402
from geodesic_field import build_geodesic_field  # noqa: E402
from bounds import compute_bounds, serialize_bounds  # noqa: E402
from classes import classify_task  # noqa: E402
from priors import build_priors, serialize_priors  # noqa: E402
from stage5_6 import (build_context, search_mechanism, validate_mechanism, design_segments,  # noqa: E402
                      serialize_design, summarize_fit, START_MODES, _transition_configs)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Categorical colors in fixed order (dataviz reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED, OBST = "#1f1f1e", "#5f5e5a", "#a3a29c", "#d9d8d2"


def _color(i):
    return SERIES[i % len(SERIES)]


def _clean(o):
    """JSON-safe copy: numpy -> python, non-finite floats -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=2)


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------

def _frame(problem, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
    else:
        fig = ax.figure
    for cx, cy, r in problem.obstacles:
        ax.add_patch(Circle((cx, cy), r, color=OBST, zorder=1))
        ax.add_patch(Circle((cx, cy), r + problem.inflation, fill=False, ec=MUTED, ls="--", lw=1, zorder=1))
    ax.plot(*problem.base, "s", color=INK, ms=9, zorder=6, label="base")
    xmin, ymin, xmax, ymax = problem.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    return fig, ax


def _targets_markers(ax, target_sets, labels=True):
    for ts in target_sets:
        c = _color(ts["index"])
        ax.plot(*ts["xy"], "*", color=c, ms=13, mec="white", mew=0.8, zorder=7)
        if labels:
            ax.annotate(f"T{ts['index']}", ts["xy"], xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color=INK)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_targets(problem, target_sets, path):
    fig, ax = _frame(problem, "Stage 1: target pose sets\n(light wedge = original band, "
                              "colored wedge = after probe pruning)")
    R = 0.9
    for ts in target_sets:
        x, y = ts["xy"]
        lo, hi = np.rad2deg(ts["band"])
        # wedges point back along the probe, toward the arm
        ax.add_patch(Wedge((x, y), R, lo + 180, hi + 180, color=OBST, alpha=0.8, zorder=2))
        for a, b in ts["intervals"]:
            ax.add_patch(Wedge((x, y), R, np.rad2deg(a) + 180, np.rad2deg(b) + 180,
                               color=_color(ts["index"]), alpha=0.45, zorder=3))
        if not ts["feasible"]:
            ax.annotate("infeasible", (x, y), xytext=(6, -12), textcoords="offset points", fontsize=8, color=INK)
        u = np.array([np.cos(ts["theta"]), np.sin(ts["theta"])])
        b0 = ts["xy"] - problem.probe_length * u
        ax.plot([b0[0], x], [b0[1], y], color=INK, lw=2, zorder=4)
        ax.annotate(f"c={ts['c_k']}, -{100 * ts['frac_removed']:.0f}%", (x, y), xytext=(6, -12),
                    textcoords="offset points", fontsize=7, color=INK_2)
    _targets_markers(ax, target_sets)
    _save(fig, path)


def plot_field(problem, field, target_sets, path):
    grid = field["grid"]
    feas = [tf for tf in field["targets"] if tf is not None]
    tf0 = feas[0]
    title = (f"Stage 2: {'geodesic distance' if field['mode'] == 'geodesic' else 'legacy potential'} field "
             f"to T{tf0['index']} (heatmap: distance; arrows: field)\nlines: reach paths to every target "
             f"(dashed = tied alternative route)")
    fig, ax = _frame(problem, title)
    d = np.where(np.isfinite(tf0["distance"]), tf0["distance"], np.nan)
    xmin, ymin, xmax, ymax = problem.bounds
    im = ax.imshow(d, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap="Blues_r", alpha=0.75, zorder=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("geodesic distance to target", fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7)
    step = max(int(len(grid["xs"]) / 20), 1)
    sl = (slice(step // 2, None, step), slice(step // 2, None, step))
    occ = grid["occupied"][sl]
    U, V = np.where(occ, np.nan, tf0["vx"][sl]), np.where(occ, np.nan, tf0["vy"][sl])
    ax.quiver(grid["XX"][sl], grid["YY"][sl], U, V, color=INK_2, alpha=0.6,
              scale=30, width=0.0025, zorder=2)
    for tf in feas:
        for j, rp in enumerate(tf["reach_paths"]):
            P = rp["points"]
            ax.plot(P[:, 0], P[:, 1], "--" if j else "-", color=_color(tf["index"]), lw=2, zorder=4,
                    label=f"T{tf['index']} ({rp['length']:.2f})" if j == 0 else None)
    _targets_markers(ax, target_sets)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    _save(fig, path)


def plot_segment_paths(problem, field, priors, target_sets, path):
    fig, ax = _frame(problem, "Stage 4: segment paths (solid) from reach paths (faint)\n"
                              "rings = bend clusters (radius = dispersion)")
    for tf, pt in zip(field["targets"], priors["per_target"]):
        if tf is None or pt is None:
            continue
        c = _color(tf["index"])
        for rp in tf["reach_paths"]:
            ax.plot(rp["points"][:, 0], rp["points"][:, 1], color=c, alpha=0.3, lw=1, zorder=2)
        for j, r in enumerate(pt["routes"]):
            V = r["vertices"]
            ax.plot(V[:, 0], V[:, 1], "o-" if j == 0 else "o--", color=c, lw=2, ms=4, zorder=4,
                    label=f"T{tf['index']}: {r['n_segments']} seg" if j == 0 else None)
    for cl in priors["bend_clusters"]:
        ax.add_patch(Circle(cl["center"], max(cl["dispersion"], 0.12), fill=False, ec=INK, lw=1.5, zorder=5))
        ax.annotate(f"n={cl['n_members']}, disp={cl['dispersion']:.2f}", cl["center"], xytext=(8, 8),
                    textcoords="offset points", fontsize=7, color=INK)
    _targets_markers(ax, target_sets, labels=False)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    _save(fig, path)


def plot_bounds(problem, bounds, target_sets, path):
    info = bounds["_link_info"]
    lv = np.where(info["free"], info["level_map"], -1).astype(float)
    lv[lv < 0] = np.nan
    lv[lv == 0] = np.nan
    c, ml, ln = bounds["counting"], bounds["min_link"], bounds["length"]
    title = (f"Stage 3: visibility levels from the base (minimum-link bound)\n"
             f"counting bound n >= {c['bound']} | m = {ml['m']} -> revolute joints after the first >= "
             f"{ml['rotating_floor']} | length >= {ln['bound']:.2f}")
    fig, ax = _frame(problem, title)
    xmin, ymin, xmax, ymax = problem.bounds
    nlev = int(np.nanmax(lv)) if np.isfinite(lv).any() else 1
    ramp = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281", "#0d366b"]  # one-hue ordinal ramp
    cmap = ListedColormap(ramp[:max(nlev, 1)])
    im = ax.imshow(lv, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap=cmap, vmin=0.5,
                   vmax=nlev + 0.5, alpha=0.7, zorder=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(1, nlev + 1))
    cb.set_label("segments needed to reach the cell", fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7)
    feas = [ts for ts in target_sets if ts["feasible"]]
    for ts, mk, L in zip(feas, ml["m_k"], ln["L_min_k"]):
        ax.plot(*ts["xy"], "*", color=_color(ts["index"]), ms=13, mec="white", mew=0.8, zorder=7)
        right = ts["xy"][0] > 0.7 * xmax
        ax.annotate(f"T{ts['index']}: m={mk}, L>={L:.2f}", ts["xy"], xytext=(-6 if right else 6, 5),
                    textcoords="offset points", fontsize=7, color=INK, ha="right" if right else "left")
    _save(fig, path)


def plot_mechanism(problem, mech, ctx, validation, path):
    design, Q = mech["design"], mech["Q"]
    combo = design["joint_types"]
    ct = validation["min_clearance_transitions"]
    ct = f"{ct:.2f}" if np.isfinite(ct) else "n/a"
    fig, ax = _frame(problem, f"Stage 5-6: selected {combo} at every target "
                              f"(gray = joint-space transitions)\nmin clearance {validation['min_clearance_targets']:.2f} "
                              f"at targets, {ct} along transitions; x = predicted bend cluster")
    Qt, _ = _transition_configs(Q, ctx["order"], 6)
    for q in Qt:
        for a, b in design_segments(problem, design, q):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=0.8, alpha=0.5, zorder=2)
    for k, ts in enumerate(ctx["target_sets"]):
        c = _color(ts["index"])
        segs = design_segments(problem, design, Q[k])
        for i, (a, b) in enumerate(segs):
            is_probe = i == len(segs) - 1
            is_p = (not is_probe) and combo[i] == "P"
            ax.plot([a[0], b[0]], [a[1], b[1]], ":" if is_probe else "--" if is_p else "-",
                    color=c, lw=2.2, zorder=4, label=f"T{ts['index']}" if i == 0 else None)
        pts = np.array([s[0] for s in segs])
        ax.plot(pts[:, 0], pts[:, 1], "o", color=c, ms=4, zorder=5)
    for cl in ctx["priors"]["bend_clusters"]:
        ax.plot(*cl["center"], "x", color=INK, ms=10, mew=2, zorder=8)
    _targets_markers(ax, ctx["target_sets"], labels=False)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, title="dashed = prismatic, dotted = probe",
              title_fontsize=7)
    _save(fig, path)


def plot_starts_comparison(report_paths, path, problem_name):
    """Evaluations to the first feasible design, for each seed and each
    start mode with a report for this problem and field (`report_paths`)."""
    rows = {}
    for rp in report_paths:
        with open(rp) as f:
            rep = json.load(f)
        rows[rep["run"]["starts"]] = [s["first_feasible_evals"] for s in rep["stage5_seed_summary"]["per_seed"]]
    modes = [m for m in START_MODES if m in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, m in enumerate(modes):
        vals = [v for v in rows[m] if v is not None]
        if vals:
            ax.bar(i, np.median(vals), width=0.55, color=_color(i), alpha=0.85, zorder=2)
            jitter = np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else [0.0]
            ax.plot(i + np.asarray(jitter), vals, "o", color=INK, ms=4, zorder=3)
        ax.annotate(f"{len(vals)}/{len(rows[m])} seeds feasible", (i, 0), xytext=(0, -28),
                    textcoords="offset points", ha="center", fontsize=8, color=INK_2, annotation_clip=False)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_ylabel("objective evaluations to first feasible design", fontsize=8, color=INK_2)
    ax.set_title(f"Starting-point comparison on {problem_name}\n(bar = median over seeds, dots = seeds)",
                 fontsize=10, color=INK)
    ax.grid(axis="y", color=OBST, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    _save(fig, path)
    return {m: rows[m] for m in modes}


def _sibling_reports(problem_name, field_mode):
    """report.json of every run of this problem and field (any start mode)."""
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "reports", "geodesic", f"{problem_name}*", "report.json"))):
        try:
            with open(p) as f:
                run = json.load(f)["run"]
        except (OSError, ValueError, KeyError):
            continue
        if run["problem"] == problem_name and run["field"] == field_mode:
            out.append(p)
    return out


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def legacy_icr_clusters(problem, target_sets):
    """ICR clusters of the legacy field to the first feasible target, for
    the `legacy_icr` start points."""
    from legacy.stage1_2 import build_exact_generators
    from legacy.stage3_4 import analyze_lie_algebra, classify_primitives
    t = next(ts for ts in target_sets if ts["feasible"])
    sub = dataclasses.replace(problem, target=tuple(t["xy"]), target_theta=t["theta"],
                              target_theta_tol=t["tol"], targets=None)
    raw, meta = build_exact_generators(sub, n_cells=14)
    s3 = analyze_lie_algebra(raw, meta)
    s4 = classify_primitives(s3["closed_stage2_results"], s3, length_scale=sub.length_scale)
    return s4["icr_clusters"]


_WORKER = {}


def _stage5_worker(seed):
    w = _WORKER
    return search_mechanism(w["problem"], w["ctx"], seed=seed, starts=w["starts"])


def _run_seeds(problem, ctx, seeds, jobs, starts):
    _WORKER.update(problem=problem, ctx=ctx, starts=starts)
    if jobs > 1 and len(seeds) > 1:
        with Pool(min(jobs, len(seeds))) as pool:  # fork start method: workers inherit _WORKER
            return pool.map(_stage5_worker, seeds)
    return [_stage5_worker(s) for s in seeds]


def _seed_summary(problem, results):
    per_seed = []
    for r in results:
        m = r["mechanism"]
        per_seed.append({
            "seed": r["seed"], "selected_n": r["selected_n"], "selected_class": r["selected_class"],
            "escalated": r["escalated"], "selected": m["combo"] if r["selected_n"] else None,
            "score": m["score"], "first_feasible_evals": r["first_feasible_evals"],
            "total_evals_search": r["total_evals_search"], "wall_time_s": r["wall_time_s"],
            "fit_order": r["fit_order"],
            "feasible_combos": [c for c in r["fit_order"] if r["fitted"][c]["feasible"]],
            "minimality": r["minimality"], "minimality_holds": r["minimality_holds"],
            "counting_bound_respected": r["counting_bound_respected"],
            "per_combo": {c: {"feasible": f["feasible"], "n_evals": f["n_evals"],
                              "evals_to_feasible": f["evals_to_feasible"], "wall_time_s": f["wall_time_s"]}
                          for c, f in r["fitted"].items()},
        })
    ff = [p["first_feasible_evals"] for p in per_seed if p["first_feasible_evals"] is not None]
    return {
        "n_seeds": len(results),
        "selected_n_counts": dict(Counter(str(p["selected_n"]) for p in per_seed)),
        "selected_combo_counts": dict(Counter(str(p["selected"]) for p in per_seed)),
        "feasible_combo_counts": dict(Counter(c for p in per_seed for c in p["feasible_combos"])),
        "first_feasible_evals_median": float(np.median(ff)) if ff else None,
        "n_seeds_feasible": len(ff),
        "per_seed": per_seed,
    }


def run_name(problem_name, starts, field_mode):
    parts = [problem_name]
    if starts != "priors":
        parts.append(f"starts_{starts}")
    if field_mode != "geodesic":
        parts.append(f"field_{field_mode}")
    return "-".join(parts)


def main(problem_name="one_obstacle", seeds=10, jobs=None, starts="priors", field_mode="geodesic"):
    name = run_name(problem_name, starts, field_mode)
    img_dir = os.path.join(ROOT, "images", "geodesic", name)
    rpt_dir = os.path.join(ROOT, "reports", "geodesic", name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(rpt_dir, exist_ok=True)
    img = lambda n: os.path.join(img_dir, n)  # noqa: E731
    rpt = lambda n: os.path.join(rpt_dir, n)  # noqa: E731
    jobs = jobs or os.cpu_count() or 1
    problem = ToyProblem(**PRESETS[problem_name])
    print(f"Run: {name}")

    print("Stage 1: target pose sets...")
    target_sets = build_target_sets(problem)
    plot_targets(problem, target_sets, img("00_targets.png"))
    for ts in target_sets:
        print(f"  T{ts['index']}: c_k={ts['c_k']}, removed {100 * ts['frac_removed']:.0f}% of band")

    print(f"Stage 2: {field_mode} field...")
    field = build_geodesic_field(problem, target_sets, field_mode=field_mode)
    plot_field(problem, field, target_sets, img("01_geodesic_field.png"))

    print("Stage 3: lower bounds...")
    bounds = compute_bounds(problem, target_sets)
    sb = serialize_bounds(bounds)
    _dump(sb, rpt("03_bounds.json"))
    plot_bounds(problem, bounds, target_sets, img("03_bounds.png"))
    print(f"  counting n >= {sb['counting']['bound']}; m_k = {sb['min_link']['m_k']} -> "
          f"rotating floor {sb['min_link']['rotating_floor']}; length >= {sb['length']['bound']:.3f}")

    print("Stage 4: motion classes and priors...")
    classes = classify_task(problem, target_sets, bounds)
    _dump(classes, rpt("04_classes.json"))
    priors = build_priors(problem, target_sets, field)
    plot_segment_paths(problem, field, priors, target_sets, img("02_segment_paths.png"))
    print(f"  smallest covering class {classes['smallest_covering_class']}, selected "
          f"{classes['selected_class']}: {classes['n_admissible']}/{classes['n_total']} admissible "
          f"{classes['admissible']} (rho={classes['rho']:.3f})")

    icr = legacy_icr_clusters(problem, target_sets) if starts == "legacy_icr" else None
    ctx = build_context(problem, target_sets, bounds, classes, priors, icr_clusters=icr)

    seed_list = list(range(seeds))
    print(f"Stage 5: reduced search over {len(seed_list)} seeds ({jobs} jobs, starts={starts})...")
    results = _run_seeds(problem, ctx, seed_list, jobs, starts)
    summary = _seed_summary(problem, results)
    print(f"  selected n per seed: {summary['selected_n_counts']}; combos: {summary['selected_combo_counts']}; "
          f"median evals to first feasible: {summary['first_feasible_evals_median']}")

    ns = [r["selected_n"] for r in results if r["selected_n"] is not None]
    modal_n = Counter(ns).most_common(1)[0][0] if ns else None
    pool = [r for r in results if r["selected_n"] == modal_n] if modal_n is not None else results
    chosen = min(pool, key=lambda r: (r["mechanism"]["score"], r["mechanism"]["residual_cost"]))
    mech = chosen["mechanism"]
    print(f"  reported mechanism: {mech['combo']} from seed {chosen['seed']} (feasible={mech['feasible']})")

    print("Stage 6: validation...")
    validation = validate_mechanism(problem, mech, ctx)
    plot_mechanism(problem, mech, ctx, validation, img("05_mechanism.png"))

    report = {
        "run": {"name": name, "problem": problem_name, "seeds": seeds, "starts": starts, "field": field_mode,
                "distance_backend": field["backend"]},
        "problem": {**{k: v for k, v in dataclasses.asdict(problem).items() if k not in ("targets",)},
                    "targets": problem.targets_list, "length_scale": problem.length_scale,
                    "inflation": problem.inflation},
        "stage1_targets": serialize_target_sets(target_sets),
        "stage2_field": {"mode": field["mode"], "backend": field["backend"], "blend_radius": field["blend_radius"],
                         "per_target": [None if tf is None else {
                             "index": tf["index"], "geodesic_length": tf["geodesic_length"],
                             "euclidean_length": tf["euclidean_length"],
                             "reach_paths": [{"label": rp["label"], "length": rp["length"],
                                              "geodesic_length": rp["geodesic_length"], "reached": rp["reached"]}
                                             for rp in tf["reach_paths"]]} for tf in field["targets"]]},
        "stage3_bounds": sb,
        "stage4_classes": classes,
        "stage4_priors": serialize_priors(priors),
        "stage5_mechanism": {
            "seed": chosen["seed"], "selected_n": chosen["selected_n"], "selected_class": chosen["selected_class"],
            "escalated": chosen["escalated"], **summarize_fit(problem, mech),
            "fits": {c: summarize_fit(problem, f) for c, f in chosen["fitted"].items()},
            "minimality": chosen["minimality"], "minimality_holds": chosen["minimality_holds"],
        },
        "stage5_seed_summary": summary,
        "stage6_validation": validation,
    }
    _dump(report, rpt("report.json"))

    comp = plot_starts_comparison(_sibling_reports(problem_name, field_mode),
                                  img("06_starts_comparison.png"), problem_name)
    print(f"  starts comparison over modes with reports: {list(comp)}")

    print(json.dumps(_clean({"bounds": {k: sb[k] for k in ("counting", "min_link")},
                             "classes": {k: classes[k] for k in ("selected_class", "admissible", "rho")},
                             "selected": {"combo": mech["combo"], "feasible": mech["feasible"],
                                          "design": serialize_design(problem, mech["design"])},
                             "validation": {k: validation[k] for k in
                                            ("feasible", "min_clearance_targets", "min_clearance_transitions",
                                             "min_sigma_ratio", "mean_elbow_to_own_bend",
                                             "mean_elbow_to_cluster", "bound_consistency")}}), indent=2))
    print(f"\nFigures written to {img_dir}\nReports written to {rpt_dir}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="one_obstacle", choices=list(PRESETS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--starts", default="priors", choices=list(START_MODES))
    ap.add_argument("--field", default="geodesic", choices=["geodesic", "potential"])
    args = ap.parse_args()
    main(args.problem, args.seeds, args.jobs, args.starts, args.field)
