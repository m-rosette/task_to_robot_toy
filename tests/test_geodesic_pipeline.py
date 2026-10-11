"""
Tests for the geodesic, reach-only pipeline (Stages 1-6).

Run with:
    .venv/bin/python -m pytest tests/
    .venv/bin/python tests/test_geodesic_pipeline.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from bounds import compute_bounds, counting_bound, minimum_link_counts  # noqa: E402
from classes import classify_task  # noqa: E402
from geodesic_field import build_geodesic_field, geodesic_distance, interp, make_grid  # noqa: E402
from priors import build_priors  # noqa: E402
from problem import ToyProblem, PRESETS  # noqa: E402
from stage5_6 import body_segments, build_context, fk, jacobian, search_mechanism  # noqa: E402
from targets import build_target_sets  # noqa: E402


def _tangent_arc_tangent(p, q, c, r):
    """Shortest path length from p to q around a circle (center c, radius r)
    that blocks the straight segment."""
    p, q, c = (np.asarray(v, float) for v in (p, q, c))
    dp, dq = np.linalg.norm(p - c), np.linalg.norm(q - c)
    ang = np.arccos(np.clip((p - c) @ (q - c) / (dp * dq), -1, 1))
    arc = ang - np.arccos(r / dp) - np.arccos(r / dq)
    return np.sqrt(dp ** 2 - r ** 2) + np.sqrt(dq ** 2 - r ** 2) + r * arc


def _stages_1_to_4(P):
    ts = build_target_sets(P)
    b = compute_bounds(P, ts)
    return ts, b, classify_task(P, ts, b)


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------

def test_fk_probe_extends_last_link():
    base = np.array([1.0, 1.0])
    for jt, dof, lengths, slide in (("RPR", [0.3, 0.7, -0.4], [2.0, 1.0, 1.5], None),
                                    ("PRR", [0.5, 0.9, -1.2], [1.0, 2.0, 1.0], 0.4),
                                    ("RR", [1.0, 0.5], [3.0, 2.0], None)):
        tip, joints = fk(base, jt, dof, lengths, base_angle=0.2, probe_length=0.5, slide_angle=slide)
        d = tip[:2] - joints[-1]
        assert abs(np.linalg.norm(d) - 0.5) < 1e-12
        assert abs(np.arctan2(d[1], d[0]) - np.angle(np.exp(1j * tip[2]))) < 1e-12
        last = joints[-1] - joints[-2]
        assert abs(np.angle(np.exp(1j * (np.arctan2(last[1], last[0]) - tip[2])))) < 1e-12
        segs = body_segments(base, jt, dof, lengths, 0.2, 0.5, slide)
        assert len(segs) == len(jt) + 1 and np.allclose(segs[-1][1], tip[:2])


def test_fk_slide_angle_sets_leading_rail():
    tip, joints = fk([0.0, 0.0], "P", [0.5], [1.0], base_angle=0.0, probe_length=0.25, slide_angle=np.pi / 2)
    assert np.allclose(joints[1], [0.0, 1.5]) and np.allclose(tip, [0.0, 1.75, np.pi / 2])


def test_jacobian_matches_finite_differences():
    base = np.array([1.0, 1.0])
    jt, lengths, q = "RPR", [2.0, 1.0, 1.5], np.array([0.3, 0.7, -0.4])
    J = jacobian(base, jt, q, lengths, 0.2, 0.5)
    eps = 1e-6
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        num = (fk(base, jt, q + dq, lengths, 0.2, 0.5)[0] - fk(base, jt, q - dq, lengths, 0.2, 0.5)[0]) / (2 * eps)
        assert np.allclose(J[:, i], num, atol=1e-6)


# --------------------------------------------------------------------------
# Stage 2: geodesic field
# --------------------------------------------------------------------------

def test_geodesic_distance_empty_space_is_euclidean():
    P = ToyProblem(obstacles=())
    grid = make_grid(P)
    t = np.array([8.5, 8.0])
    d = geodesic_distance(grid, t)
    eu = np.hypot(grid["XX"] - t[0], grid["YY"] - t[1])
    assert np.max(np.abs(d - eu)) < 2 * grid["h"]


def test_geodesic_distance_around_one_circle():
    for P in (ToyProblem(link_radius=0.0, clearance_margin=0.0), ToyProblem()):
        grid = make_grid(P)
        d = interp(geodesic_distance(grid, P.target), grid, np.asarray(P.base))
        exact = _tangent_arc_tangent(P.base, P.target, (5.0, 5.0), 1.0 + P.inflation)
        assert abs(d - exact) < 0.005 * exact, (d, exact)
    # tangent-arc-tangent length without inflation
    assert abs(_tangent_arc_tangent((1, 1), (8.5, 8.0), (5, 5), 1.0) - 10.39) < 0.01


def test_reach_paths_start_end_and_stay_free():
    for name in ("one_obstacle", "two_obstacles", "multi_target_wide"):
        P = ToyProblem(**PRESETS[name])
        ts = build_target_sets(P)
        F = build_geodesic_field(P, ts)
        h = F["grid"]["h"]
        for s, tf in zip(ts, F["targets"]):
            for rp in tf["reach_paths"]:
                pts = rp["points"]
                assert rp["reached"], (name, s["index"], rp["label"])
                assert np.allclose(pts[0], P.base)
                assert np.linalg.norm(pts[-2] - s["xy"]) <= h + 1e-9
                # the path stays out of the inflated obstacles, within one cell of grid error
                assert P.clearance_field(pts).min() >= P.inflation - h, (name, s["index"])


def test_route_tie_behind_obstacle_is_resolved():
    """(7, 7) is on the cut locus behind the obstacle. The two walled routes
    reach it with equal length."""
    P = ToyProblem(targets=((7.0, 7.0, np.deg2rad(75.0), np.deg2rad(40.0)),))
    F = build_geodesic_field(P, build_target_sets(P))
    rps = F["targets"][0]["reach_paths"]
    assert len(rps) == 2 and all(r["reached"] for r in rps)
    assert abs(rps[0]["geodesic_length"] - rps[1]["geodesic_length"]) < 1e-6


# --------------------------------------------------------------------------
# Stage 1 and Stage 3
# --------------------------------------------------------------------------

def test_tolerance_pruning():
    # on part of the band, the probe points back toward the obstacle
    P = ToyProblem(targets=((6.4, 5.6, np.deg2rad(65.0), np.deg2rad(40.0)),
                            # every heading puts the probe inside the inflated obstacle
                            (6.2, 5.0, np.deg2rad(0.0), np.deg2rad(5.0))))
    a, b = build_target_sets(P)
    assert 0.0 < a["frac_removed"] < 1.0 and a["c_k"] == 2 and a["feasible"]
    lo = min(i[0] for i in a["intervals"])
    assert lo > a["band"][0]  # Stage 1 removes the low end (probe toward the obstacle)
    assert not b["feasible"] and b["c_k"] is None and b["frac_removed"] == 1.0
    # an exact heading fixes three coordinates
    (c,) = build_target_sets(ToyProblem(target_theta_tol=0.0))
    assert c["c_k"] == 3


def test_counting_bound_table():
    table = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 3), 6: (2, 3), 8: (2, 3), 12: (2, 3)}
    for K, (n2, n3) in table.items():
        assert counting_bound([2] * K) == n2, K
        assert counting_bound([3] * K) == n3, K


def test_minimum_link_bound():
    P0 = ToyProblem(obstacles=())
    assert minimum_link_counts(P0, [P0.target])[0] == [1]
    P1 = ToyProblem()
    assert minimum_link_counts(P1, [P1.target])[0] == [2]


def test_rotating_joint_floor_one_obstacle():
    _, b, c = _stages_1_to_4(ToyProblem())
    assert b["min_link"]["m"] == 2 and b["min_link"]["rotating_floor"] == 1
    assert sorted(c["excluded"]) == sorted(["R", "P", "RP", "PP", "RPP", "PPP"])
    assert c["n_admissible"] == 8 and c["selected_class"] == "se2"


def test_length_bound_one_obstacle():
    P = ToyProblem()
    _, b, _ = _stages_1_to_4(P)
    exact = _tangent_arc_tangent(P.base, P.target, (5.0, 5.0), 1.0 + P.link_radius)
    assert b["length"]["bound"] > b["length"]["euclidean_k"][0]
    assert abs(b["length"]["bound"] - exact) < 0.005 * exact


# --------------------------------------------------------------------------
# Stage 4
# --------------------------------------------------------------------------

def _problem(targets):
    return ToyProblem(obstacles=(), targets=tuple(targets))


def test_motion_class_tests():
    base = np.array([1.0, 1.0])
    tol = np.deg2rad(5.0)
    # common heading, spread positions -> planar translation
    P = _problem([(3.0, 6.0, np.pi / 2, tol), (5.0, 7.5, np.pi / 2, tol), (7.0, 5.5, np.pi / 2, tol),
                  (8.5, 8.0, np.pi / 2, tol)])
    assert _stages_1_to_4(P)[2]["smallest_covering_class"] == "planar_translation"
    # equidistant from the base, heading = bearing + 0.3 -> rotation about the base
    angs = np.deg2rad([10.0, 35.0, 60.0, 80.0])
    P = _problem([(base[0] + 6 * np.cos(a), base[1] + 6 * np.sin(a), a + 0.3, tol) for a in angs])
    c = _stages_1_to_4(P)[2]
    assert c["smallest_covering_class"] == "rotation", c["coverage"]
    # collinear with the base, common heading -> translation along a line
    P = _problem([(1 + 2 * k, 1 + 1 * k, 0.5, tol) for k in (1, 2, 3, 4)])
    assert _stages_1_to_4(P)[2]["smallest_covering_class"] == "line_translation"
    # mixed headings and radii -> SE(2)
    P = _problem([(3.0, 6.0, 0.0, tol), (5.0, 7.5, 1.5, tol), (7.0, 5.5, -1.0, tol), (8.5, 8.0, 2.5, tol)])
    assert _stages_1_to_4(P)[2]["smallest_covering_class"] == "se2"


def test_reduced_design_space_keeps_generating_combination():
    rng = np.random.default_rng(3)
    base = np.array([1.0, 1.0])
    cases = (("RRR", [4.0, 3.0, 1.5], None, 0.0), ("PRR", [2.0, 3.0, 2.0], 0.7, 0.0),
             ("RR", [5.0, 4.0], None, np.deg2rad(20.0)), ("PR", [3.0, 4.0], 0.8, np.deg2rad(20.0)))
    for jt, lengths, slide, tol in cases:
        targets = []
        while len(targets) < 6:
            q = np.array([rng.uniform(-1.0, 1.5) if t == "R" else rng.uniform(0.0, 2.0) for t in jt])
            tip, _ = fk(base, jt, q, lengths, 0.0, 0.5, slide)
            if 0.5 < tip[0] < 9.5 and 0.5 < tip[1] < 9.5:
                targets.append((tip[0], tip[1], tip[2], tol))
        P = _problem(targets)
        _, b, c = _stages_1_to_4(P)
        assert jt in c["admissible"], (jt, c["selected_class"], c["admissible"])
        assert len(jt) >= b["counting"]["bound"]


# --------------------------------------------------------------------------
# Stage 5
# --------------------------------------------------------------------------

def test_search_stops_early_and_reports_smaller_failures():
    P = ToyProblem()
    ts = build_target_sets(P)
    F = build_geodesic_field(P, ts)
    b = compute_bounds(P, ts)
    c = classify_task(P, ts, b)
    ctx = build_context(P, ts, b, c, build_priors(P, ts, F))
    S = search_mechanism(P, ctx, seed=0, starts="priors")
    assert S["selected_n"] == 2 and S["mechanism"]["feasible"]
    assert all(len(cmb) == 2 for cmb in S["fit_order"])  # the search does not reach 3 joints
    assert sorted(m["combo"] for m in S["minimality"]) == ["P", "R"]
    assert S["minimality_holds"] and S["first_feasible_evals"] is not None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
