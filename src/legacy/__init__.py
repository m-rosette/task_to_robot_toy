"""
Bracket-based pipeline (Stages 1-6 as of commit 9643975), kept unchanged so
the earlier preliminary results stay reproducible. `run_pipeline.py` is its
runner; the geodesic, reach-only pipeline lives in `src/` and is run by
`run_geodesic_pipeline.py`. Delete once the proposal no longer cites these
results.

`stage5_6.py` here is a frozen copy of the old Stage 5/6 module: its `fk`
fits `base_angle` and a `tool_offset` angle, unlike the new `src/stage5_6.py`
(fixed base angle, fixed probe).
"""
