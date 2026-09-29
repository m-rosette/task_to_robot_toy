"""
Stage 1: Learn the Task-Space Twist Field
Stage 2: Discover the Minimal Differential Generators
"""

import numpy as np
from problem import wrap_angle, twist_normalize
from field import bow_signs, desired_velocity, desired_heading


def build_exact_generators(problem, signs=None, n_cells=14, min_twist_norm=1e-2,
                            deriv_step=0.05, **field_kwargs):
    """Stage 1+2, formula-reading: since `field.py`'s navigation field is
    already analytic, read it directly on a grid rather than learning it
    from simulated demonstrations. No streamline integration, no process
    noise, no kernel smoothing, no significance testing -- every kept cell's
    twist is exact by construction, not estimated. This is the pipeline's
    sole Stage 1+2 implementation (an earlier demonstration-learning path --
    noisy simulated streamlines, kernel-smoothed and statistically
    recovered -- existed alongside this one and was removed once it was no
    longer needed; see SIMPLIFICATIONS.md Phase 11).

    Returns a `stage2_results`-shaped dict with the 3 keys every downstream
    stage actually reads ("center", "generators", "intrinsic_dim" --
    verified by grepping every consumer in stage3_4.py and stage5_6.py),
    plus a `meta` dict compatible with `analyze_lie_algebra` (needs
    "cell_w"/"cell_h" for its spatial-derivative brackets).

    `generators` are world-frame twists -- `desired_velocity` is already
    world-frame by construction.

    Trade-off (SIMPLIFICATIONS.md item 19): a single-valued vector field can
    only ever return what's explicitly encoded in its formulas. It cannot
    *discover* multi-modal/redundant admissible motion the way a
    statistical, data-driven approach could from real residual variance in
    demonstrations -- only represent it, if the field is deliberately
    defined to be tolerant somewhere (e.g. this field's target_theta_tol
    dead-band).

    Every cell carries exactly ONE generator: the field's required twist,
    normalized under the unit-consistent twist metric (`problem.twist_norm`).
    A single-valued field is rank 1 everywhere, and a rank-1 distribution is
    always integrable (Frobenius), so Stage 3 correctly finds nothing to
    bracket here. Until Phase 13 a second generator (the translational part
    of the twist, orthogonalized against it) was injected wherever heading
    curvature was significant; its bracket with the first generator left
    their span by construction, which forced the DOF estimate to 3 whenever
    any cell curved (SIMPLIFICATIONS.md Phase 13). Genuinely multi-directional
    requirements have to come from the task definition, not from a
    decomposition of one twist.

    Each cell also stores `"heading"`: the field's desired end-effector
    heading there. The required twist is a tangent vector at the *pose*
    (center, heading), so any stage that evaluates a mechanism against it
    (Stage 5 alignment, Stage 6 validation) must put the mechanism at that
    pose, not at an arbitrary heading.
    """
    if signs is None:
        signs = bow_signs(problem)
    blend_radius = field_kwargs.get("blend_radius", 2.0)
    L = problem.length_scale

    xmin, ymin, xmax, ymax = problem.bounds
    cell_w = (xmax - xmin) / n_cells
    cell_h = (ymax - ymin) / n_cells

    def _heading_at(q):
        if problem.obstacle_clearance(q) < 0:
            return None
        vq = desired_velocity(q, problem, signs, **field_kwargs)
        return desired_heading(q, vq, problem, blend_radius=blend_radius)

    results = {}
    for i in range(n_cells):
        for j in range(n_cells):
            center = np.array([xmin + (i + 0.5) * cell_w, ymin + (j + 0.5) * cell_h])
            if problem.obstacle_clearance(center) < 0:
                continue
            v = desired_velocity(center, problem, signs, **field_kwargs)
            vnorm = np.linalg.norm(v)
            if vnorm < min_twist_norm:
                continue  # no meaningful required motion here (e.g. at rest)
            v_hat = v / vnorm
            theta0 = desired_heading(center, v, problem, blend_radius=blend_radius)

            # Scale-invariant curvature dtheta/ds (radians per unit
            # distance) via central difference along the travel direction --
            # central/one-sided/fallback pattern mirrors `_spatial_jacobian`
            # in stage3_4.py. Computed per unit distance first, then
            # chain-rule scaled by |v| below, so omega is consistent with
            # the translational speed of the same twist.
            theta_f = _heading_at(center + deriv_step * v_hat)
            theta_b = _heading_at(center - deriv_step * v_hat)
            if theta_f is not None and theta_b is not None:
                dtheta_ds = wrap_angle(theta_f - theta_b) / (2 * deriv_step)
            elif theta_f is not None:
                dtheta_ds = wrap_angle(theta_f - theta0) / deriv_step
            elif theta_b is not None:
                dtheta_ds = wrap_angle(theta0 - theta_b) / deriv_step
            else:
                dtheta_ds = 0.0  # boxed in by the obstacle on both sides

            omega = dtheta_ds * vnorm  # chain rule: ds/dt = |v| along the flow
            gen0 = twist_normalize(np.array([v[0], v[1], omega]), L)

            results[(i, j)] = {
                "center": center,
                "heading": float(theta0),
                "generators": gen0[None, :],
                "intrinsic_dim": 1,
            }

    meta = {"bounds": problem.bounds, "n_cells": n_cells,
            "cell_w": cell_w, "cell_h": cell_h, "mode": "exact",
            "length_scale": L}
    return results, meta
