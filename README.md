# task_to_robot_toy

2D (SE(2)) prototype of twist-field-driven kinematic synthesis: from a task
(workspace, target pose with heading tolerance, circular obstacles) build a
navigation twist field, analyze its differential structure, and search for
the smallest planar serial R/P chain that reaches the target without
collision.

## Pipeline

| File | Stage |
|---|---|
| `src/problem.py` | Task definition (`ToyProblem`, presets) and SE(2) / twist-metric helpers |
| `src/field.py` | Analytic navigation field (attraction, obstacle swirl, heading policy) |
| `src/stage1_2.py` | Stages 1–2: read the field on a grid, one required twist per cell |
| `src/stage3_4.py` | Stage 3: Lie bracket / involutivity and DOF lower bound. Stage 4: revolute/prismatic classification and ICR clustering |
| `src/stage5_6.py` | Stage 5: search every R/P chain with 1–3 joints and select the fewest feasible joints. Stage 6: validation |
| `src/run_pipeline.py` | Runs Stages 1–6 and writes figures and JSON reports |

## Setup

Tested with Python 3.12.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
cd src
../.venv/bin/python run_pipeline.py                          # default: one obstacle, 10 seeds
../.venv/bin/python run_pipeline.py --problem two_obstacles --seeds 4 --jobs 4
```

Stage 5 runs once per seed in parallel, at about 1 minute per seed. Figures
are written to `images/` and reports to `reports/` at the repo root. Both are
generated, so they are not tracked.

## Tests

```bash
.venv/bin/python tests/test_stage3_bracket.py
.venv/bin/python tests/test_stage4_6_fixes.py
```

The tests also run under `pytest` if it is installed.
