# Local validation, 2026-10-20

**Phase 1 remains incomplete. Zero Motus RoboTwin episodes completed. There is
no measured success rate, episode speedup, completion time, policy latency or
compression result for real policy trajectories. Training is not justified by
the available evidence and was not launched.**

| Check | Actual result |
| --- | --- |
| Original rollout launch | Preflight blocked before importing simulator/model or loading a checkpoint. See [preflight.json](local_baseline_attempt/preflight.json) and [failure.json](local_baseline_attempt/failure.json). |
| Shared environment | Expected sibling `uv_envs/robotwin-classic` absent; no environment override was set. WSL Ubuntu also has no shared environment at that location and no visible `nvidia-smi`/WSL GPU interface. No environment was created. |
| Models/data | Checked-in deployment paths are `...`; no released checkpoint, WAN/T5/VLM path was provided. The sibling `data/` contains no demonstration episodes. No demonstration replay or K selection could run. |
| Windows checkout | Sibling `task_config` is a plain file containing a symlink target, rather than a usable directory. Use a correctly checked-out Linux sibling fork for simulation; it was not rewritten here. |
| Numerical checks | 13 unittest checks passed outside the Windows sandbox, using Python 3.13.7, NumPy 2.3.3, SciPy 1.18.0. These are CPU software checks, not RoboTwin smoke tests. |
| Analytic cubic | Fits with four shared controls, meets 1e-5 rad joint tolerances, and preserves the original analog grippers at 1x. Payload accounting shows compression for this constructed example; it does not establish compression of Motus predictions. |
| Adaptive reconstruction | A local curved perturbation requires additional common knots; original-time per-channel error checks pass. A deliberately insufficient four-control fit fails and produces byte-identical dense fallback at all three source speed factors. |
| Dense baseline | Bit-exact dense 1x values, including signed zero and grippers, verified. All six numerical variants share the same source endpoint/prefix and retiming map. |
| Executor recorder | A mock variable-duration executor preserves action values, early success, action limit and exact physics-step counts. Real SAPIEN scene-proxy compatibility remains untested. |
| Acceleration | Four high-speed RoboTwin conditions deliberately blocked because actual TOPP target durations vary. Source-parameter compression is not verified physical temporal scaling. |
| Lint/format | All four new Python files pass Ruff checks and formatting checks. `ruff check .` reports 1,932 existing issues; `ruff format --check .` reports 157 files needing formatting. Those unrelated files were not changed. No pre-commit configuration exists in this checkout. |

Exact commands used from the Motus root in PowerShell:

```powershell
# Motus-only CPU utilities; this does not invoke the RoboTwin simulator with system Python.
& 'C:\Users\Kevin\AppData\Local\Programs\Python\Python313\python.exe' -m unittest discover -s tests -v
& 'C:\Users\Kevin\AppData\Local\Programs\Python\Python313\python.exe' -m inference.robotwin.spline_experiment run --mode original --output experiments/spline/local_baseline_attempt
& '.cache\adapter-check-tools\bin\ruff.exe' check inference/robotwin/spline_actions.py inference/robotwin/spline_rollout.py inference/robotwin/spline_experiment.py tests/test_spline_actions.py
& '.cache\adapter-check-tools\bin\ruff.exe' format --check inference/robotwin/spline_actions.py inference/robotwin/spline_rollout.py inference/robotwin/spline_experiment.py tests/test_spline_actions.py
& '.cache\adapter-check-tools\bin\ruff.exe' check . --output-format concise
& '.cache\adapter-check-tools\bin\ruff.exe' format --check .
```

The test command encounters Windows sandbox denials writing to temporary
directories, including repository-local temporary directories. The complete
suite passed with elevated execution after those denials; the ten purely
numerical checks had also passed inside the sandbox before the final added
timestamp-offset check. Ruff formatting/import sorting
was applied only to the four newly added Python files. No existing policy,
model, simulator, physics, controller, task or success code was edited.

The required next evidence is an actual original-policy smoke rollout using the
existing GPU/shared environment and released models, followed by matched dense
1x and spline 1x rollouts. Resolve fixed-rate physical target execution within
the user's controller constraints before enabling 1.5x/2x. Then evaluate the
six variants, compare time jointly with success (including common-success
paired scenes), inspect boundary/fallback/contact failures, and decide whether
to proceed with the [Phase 2 design](PHASE2.md).
