"""
Stage 1: Target pose sets.

Each target is a position and a heading band [theta - tol, theta + tol].
Stage 1 samples the band. It removes each heading at which the probe is less
than the planning inflation (`link_radius + clearance_margin`) from an
obstacle. The probe has length `probe_length`, ends at the target point, and
points along the heading. The headings that stay are the pose set of the
target:

- Empty: no arm can reach the target with an acceptable heading
  (infeasible).
- One sample, or a band narrower than `NARROW_BAND`: the heading is fixed,
  and the target fixes c_k = 3 coordinates (x, y, theta).
- An interval: the heading is free in the interval, and the target fixes
  c_k = 2 coordinates.

The pruned band is a list of contiguous intervals (absolute angles,
unwrapped around the nominal heading). An obstacle can divide a wide band
into two intervals.
"""

import numpy as np
from scipy import ndimage

from problem import wrap_angle


SAMPLE_STEP = np.deg2rad(1.0)   # heading sample step in each band
NARROW_BAND = np.deg2rad(2.0)   # a band narrower than this fixes the heading (c_k = 3)


def _band_samples(theta, tol):
    if tol <= 0.0:
        return np.array([theta])
    n = max(int(np.ceil(2 * tol / SAMPLE_STEP)), 1)
    return theta + np.linspace(-tol, tol, n + 1)


def intervals_of(samples, keep):
    """Contiguous runs of kept samples -> list of (lo, hi) intervals."""
    labels, _ = ndimage.label(keep)
    return [(float(samples[sl.start]), float(samples[sl.stop - 1])) for (sl,) in ndimage.find_objects(labels)]


def build_target_sets(problem):
    """Stage 1: one pose set for each target in `problem.targets_list`.

    Returns a list of dicts. Each dict has the nominal target, the
    original band, the sampled headings, and the mask of kept headings.
    It also has the pruned intervals, `c_k`, `frac_removed`, `feasible`,
    and `target_clearance`.
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
        intervals = intervals_of(samples, keep)
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
    """Signed distance from `heading` to the nearest pruned interval (0 in
    an interval). The sign is the sign of (heading - nearest edge). Thus
    the derivative with respect to the heading is +1 outside the band."""
    best = None
    for lo, hi in intervals:
        mid = 0.5 * (lo + hi)
        e = wrap_angle(heading - mid)
        half = 0.5 * (hi - lo)
        r = 0.0 if abs(e) <= half else e - np.sign(e) * half
        if best is None or abs(r) < abs(best):
            best = r
    return 0.0 if best is None else float(best)


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
