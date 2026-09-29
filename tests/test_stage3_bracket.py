"""
Ground-truth tests for Stage 3's Lie bracket (SIMPLIFICATIONS.md Phase 13).

Run with either:
    .venv/bin/python tests/test_stage3_bracket.py
    .venv/bin/python -m pytest tests/        (if pytest is installed)

Twists are hybrid (EE-point velocity, omega) == coordinate velocities
(xdot, ydot, thetadot), so the correct bracket is the coordinate bracket
DY.X - DX.Y. Three cases with known answers:
  1. constant fields "translate x" and "spin about the EE point" commute;
  2. the joint fields of a 2R arm are coordinate fields on its image, so
     their bracket lies in their span (involutive);
  3. the Heisenberg pair X = d/dx, Y = d/dy + (x/L^2) d/dtheta is NOT
     involutive: [X, Y] = (1/L^2) d/dtheta, outside span{X, Y}. Guards
     against a bracket that trivially returns zero. The 1/L^2 coupling
     keeps Y mostly translational under the unit-consistent metric, i.e.
     it is the textbook pair written in dimensionless coordinates; with a
     coupling of 1, Y would be nearly a pure spin at large x and the
     bracket would sit within tolerance of span{X, Y}.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from stage3_4 import analyze_lie_algebra  # noqa: E402
from stage5_6 import jacobian  # noqa: E402

N = 14
W = 10.0 / N


def _grid(gen_fn):
    res = {}
    for i in range(N):
        for j in range(N):
            p = np.array([(i + 0.5) * W, (j + 0.5) * W])
            G = gen_fn(p)
            if G is None:
                continue
            res[(i, j)] = {"center": p, "generators": np.asarray(G, dtype=float),
                           "intrinsic_dim": len(G)}
    return res


def _run(gen_fn, L=10.0):
    res = _grid(gen_fn)
    return analyze_lie_algebra(res, {"cell_w": W, "cell_h": W, "length_scale": L}), len(res)


def test_constant_translation_and_spin_commute():
    s3, n = _run(lambda p: [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert s3["n_new_direction"] == 0, s3["summary"]
    assert s3["n_commuting"] == n
    assert s3["dof_lower_bound"] == 2


def test_2r_joint_fields_are_involutive():
    B = np.array([1.0, 1.0])
    Ls = np.array([6.0, 6.0])

    def gens(p):
        d = p - B
        c2 = (d @ d - Ls @ Ls) / (2 * Ls[0] * Ls[1])
        if abs(c2) > 0.98:
            return None
        q2 = np.arccos(c2)
        q1 = np.arctan2(d[1], d[0]) - np.arctan2(Ls[1] * np.sin(q2), Ls[0] + Ls[1] * np.cos(q2))
        J = jacobian(B, "RR", np.array([q1, q2]), Ls)
        return (J / np.linalg.norm(J, axis=0)).T

    s3, n = _run(gens)
    frac_new = s3["n_new_direction"] / s3["n_multi_generator_cells"]
    # Finite differences + cross-cell generator matching leave a few false
    # positives on a 14x14 grid (was 177/186 with the old formula).
    assert frac_new < 0.10, f"{s3['n_new_direction']}/{s3['n_multi_generator_cells']} flagged"
    assert s3["dof_lower_bound"] == 2


def test_heisenberg_pair_is_not_involutive():
    L = 10.0
    s3, n = _run(lambda p: [[1.0, 0.0, 0.0], [0.0, 1.0, p[0] / L ** 2]], L=L)
    # Every interior cell (central differences available) must be flagged;
    # edge cells are flagged too but only count as low-confidence.
    n_interior = (N - 2) ** 2
    assert s3["n_new_direction"] >= 0.9 * n_interior, s3["summary"]
    assert s3["n_new_direction"] + s3["n_new_direction_low_confidence"] == n, s3["summary"]
    assert s3["dof_lower_bound"] == 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
