"""
Stage 1: Target pose sets.

Each target is a position plus a heading band [theta - tol, theta + tol].
The band is sampled, and every heading at which the probe (length
`probe_length`, ending at the target point and pointing along the heading)
would come within the planning inflation (`link_radius + clearance_margin`)
of an obstacle is removed. What remains is the target's pose set:

- empty: no arm can reach the target with an acceptable heading (infeasible);
- a single sample or a band narrower than `NARROW_BAND`: heading is fixed,
  so the target fixes c_k = 3 coordinates (x, y, theta);
- an interval: heading is free inside it, so the target fixes c_k = 2.

The pruned band is stored as a list of contiguous intervals (absolute
angles, unwrapped around the nominal heading), since an obstacle can split
a wide band in two.
"""

import numpy as np

from problem import wrap_angle


SAMPLE_STEP = np.deg2rad(1.0)   # heading discretization of each band
NARROW_BAND = np.deg2rad(2.0)   # a band narrower than this fixes the heading (c_k = 3)


def probe_segment(xy, heading, probe_length):
    """Probe as a segment (back end, tip) with the tip at `xy`."""
    xy = np.asarray(xy, dtype=float)
    u = np.array([np.cos(heading), np.sin(heading)])
    return xy - probe_length * u, xy


def _band_samples(theta, tol):
    if tol <= 0.0:
        return np.array([theta])
    n = max(int(np.ceil(2 * tol / SAMPLE_STEP)), 1)
    return theta + np.linspace(-tol, tol, n + 1)


def _runs(samples, keep):
    """Contiguous runs of kept samples -> list of (lo, hi) intervals."""
    out = []
    start = None
    for i, k in enumerate(keep):
        if k and start is None:
            start = i
        if (not k or i == len(keep) - 1) and start is not None:
            end = i if k else i - 1
            out.append((float(samples[start]), float(samples[end])))
            start = None
    return out


def build_target_sets(problem):
    """Stage 1: one pose set per target in `problem.targets_list`.

    Returns a list of dicts with the nominal target, the original band, the
    sampled headings and which survived pruning, the pruned intervals,
    `c_k`, `frac_removed`, and `feasible`.
    """
    out = []
    infl = problem.inflation
    for k, (x, y, theta, tol) in enumerate(problem.targets_list):
        xy = np.array([x, y])
        samples = _band_samples(theta, tol)
        u = np.stack([np.cos(samples), np.sin(samples)], axis=-1)
        backs = xy[None, :] - problem.probe_length * u
        clear = problem.segment_clearance(backs, xy[None, :])
        keep = clear > infl
        intervals = _runs(samples, keep)
        if not intervals:
            c_k = None
        else:
            widest = max(hi - lo for lo, hi in intervals)
            c_k = 3 if widest < NARROW_BAND else 2
        out.append({
            "index": k,
            "xy": xy,
            "theta": float(theta),
            "tol": float(tol),
            "band": (float(theta - tol), float(theta + tol)),
            "samples": samples,
            "keep": keep,
            "intervals": intervals,
            "c_k": c_k,
            "frac_removed": float(1.0 - keep.mean()),
            "feasible": bool(intervals),
            "target_clearance": float(problem.obstacle_clearance(xy)),
        })
    return out


def heading_residual(heading, intervals):
    """Signed distance from `heading` to the nearest pruned interval (0 when
    inside one). The sign is that of (heading - nearest edge), so the
    derivative with respect to the heading is +1 outside the band."""
    best = None
    for lo, hi in intervals:
        mid = 0.5 * (lo + hi)
        e = wrap_angle(heading - mid)
        half = 0.5 * (hi - lo)
        r = 0.0 if abs(e) <= half else e - np.sign(e) * half
        if best is None or abs(r) < abs(best):
            best = r
    return 0.0 if best is None else float(best)


def common_heading_intervals(target_sets, step=SAMPLE_STEP):
    """Headings (sampled on a circle) that lie inside the pruned band of
    every feasible target. Returns (headings, counts) over a full-circle
    grid, where counts[i] is how many targets' bands contain headings[i]."""
    grid = np.arange(-np.pi, np.pi, step)
    counts = np.zeros(len(grid), dtype=int)
    for ts in target_sets:
        if not ts["feasible"]:
            continue
        inside = np.array([heading_residual(h, ts["intervals"]) == 0.0 for h in grid])
        counts += inside
    return grid, counts


def serialize_target_sets(target_sets):
    return [{
        "index": ts["index"],
        "xy": ts["xy"].tolist(),
        "theta_deg": float(np.rad2deg(ts["theta"])),
        "tol_deg": float(np.rad2deg(ts["tol"])),
        "pruned_intervals_deg": [[float(np.rad2deg(lo)), float(np.rad2deg(hi))] for lo, hi in ts["intervals"]],
        "c_k": ts["c_k"],
        "frac_removed": ts["frac_removed"],
        "feasible": ts["feasible"],
    } for ts in target_sets]
