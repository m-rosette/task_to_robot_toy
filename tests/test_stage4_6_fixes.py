"""
Regression tests for the Phase 13 fixes in Stages 4-6.

Run with:  .venv/bin/python tests/test_stage4_6_fixes.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from problem import ToyProblem  # noqa: E402
from stage3_4 import classify_primitives  # noqa: E402
from stage5_6 import _field_realizability, fk, solve_ik  # noqa: E402


def test_realizability_ik_uses_field_heading():
    """The old alignment check solved IK for heading 0 at every cell, which
    forced PRP onto its singular angle. The mechanism must be placed at the
    pose the field asks for."""
    P = ToyProblem()
    design = {"joint_types": list("RRR"), "lengths": np.array([5.0, 4.0, 1.0]),
              "base_angle": 0.0, "tool_offset": 0.0}
    heading = 1.1
    s2 = {(0, 0): {"center": np.array([6.0, 4.0]), "heading": heading,
                   "generators": np.array([[1.0, 0.0, 0.0]]), "intrinsic_dim": 1}}
    out = _field_realizability(P, design, s2, [(0, 0)], n_restarts=4)
    assert out["reached"] == [True]
    ee, _ = fk(P.base, design["joint_types"], out["configs"][(0, 0)], design["lengths"])
    assert abs(np.angle(np.exp(1j * (ee[2] - heading)))) < 1e-4


def test_prismatic_ik_respects_stroke():
    """Prismatic joints are rails with carriage position in [0, stroke]."""
    dof, cost = solve_ik(np.array([0.0, 0.0]), ["P"], np.array([2.0]),
                         np.array([5.0, 0.0, 0.0]), n_restarts=3, position_only=True)
    assert 0.0 <= dof[0] <= 2.0 + 1e-9
    assert cost > 1.0  # target is out of stroke, so it must NOT be reached


def test_classification_is_unit_invariant():
    """Rescaling lengths by s (twists' v by s, centers by s, L by s) must not
    change which generators are prismatic or revolute."""
    rng = np.random.default_rng(0)
    s2 = {}
    for k in range(60):
        v = rng.normal(size=2)
        omega = rng.normal() * 10 ** rng.uniform(-3, 0)
        s2[(k, 0)] = {"center": rng.uniform(0, 10, 2),
                      "generators": np.array([[v[0], v[1], omega]]), "intrinsic_dim": 1}
    kinds = lambda res: [r["kind"] for r in res["records"]]  # noqa: E731
    base = classify_primitives(s2, length_scale=10.0)
    for s in (0.001, 1000.0):
        scaled = {k: {**r, "center": r["center"] * s,
                      "generators": r["generators"] * np.array([s, s, 1.0])}
                  for k, r in s2.items()}
        assert kinds(classify_primitives(scaled, length_scale=10.0 * s)) == kinds(base)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
