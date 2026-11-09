# Validation and runtime status

Implementation checked on 2026-10-20; runtime status updated on 2026-10-21.
The initial numerical checks below preceded the GPU pilot. The original Motus
deployment source remains unchanged. Available assets did not permit an original
rollout before integration; `evaluate.py --original` retains that exact execution
body. Measured pilot results and subsequent submissions appear at the end.

## Completed checks

- 37 focused CPU tests passed in 0.85 s; see [saved output](checks/pytest.txt).
- Timestamped nonuniform cubic interpolation across 12 channels, adaptive shared
  knots, explicit per-channel maximum error, strict fitting budgets, invalid
  parameter/domain rejection, NPZ parameter/timestamp roundtrip.
- Identical spline coefficients at all three speeds; direct parameter decoding
  without invoking the encoder; bounded joint-angle convention.
- Complete executed source span at 1×/1.5×/2× for short and normal chunks, exact
  endpoint inclusion, ordered rapid gripper open/close events, identical dense
  and spline query schedules, dense fallback on backend errors, interpolation
  overshoot, and nonfinite decoding. Failure diagnostics remain valid strict JSON.
- Dense 1× command values, dtype, order, call count, and early-success no-op calls
  match the **actual upstream `eval` function body**, executed with a fake policy
  and environment. This is command parity, not a measured simulator rollout.
- A fake environment with varying physics steps verifies measured episode time
  (18 steps × 0.004 = 0.072 s), excluding setup steps. Episode metrics, command
  history, and random seeds reset between episodes.
- Matched launch settings and typed JSON/YAML overrides; successful completion
  times and pairing checks in the report; empty results never create a task score.
- All 18 commands for six variants × three requested seeds generated in a dry run.
- Ruff lint/format checks passed on all eight new Motus Python files. Syntax-level Ruff
  checks passed on the WAM bridge and existing evaluator. Both Git diffs pass
  whitespace checks. No source or weights used for Motus inference were edited.

The required shared simulator environment was absent. Numerical checks used the
existing **uv-managed CPython 3.11.15**, with temporary CPU dependencies in
`/tmp/motus-spline-checks` (NumPy 2.4.6, SciPy 1.16.3, pytest 9.1.1, Ruff 0.17.1).
No replacement virtual environment or system Python was used. The numerical
dependencies were not installed into a simulator or training environment.

Exact numerical command used, from `/project/6101778/kding/Motus`:

```bash
PYTHONPATH=/tmp/motus-spline-checks:/project/6101778/kding/Motus \
  /home/kding/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11 \
  -m pytest -q tests/test_action_spline.py tests/test_spline_evaluation.py
```

## Initial runtime blockers (recorded before container preparation)

[Recorded preflight](checks/preflight/preflight.json) checks the existing
`paths_config.yml` placeholder values, not an invented checkpoint location:

- No executable at the configured default shared interpreter:
  `/project/6101778/kding/uv_envs/robotwin-classic/bin/python`. No
  `ROBOTWIN_CLASSIC_ENV` / `ROBOTWIN_UV_ENV_ROOT` override was set.
- Motus's checked-in checkpoint, Wan, and Qwen paths are all `"..."`.
  No valid `mp_rank_00_model_states.pt`, Wan config/VAE, T5 weights/tokenizer,
  or Qwen processor location was supplied or discovered in this workspace.
- `WAM-RoboTwin/assets/embodiments/aloha-agilex/config.yml` and the objects
  directory exist. This does not establish complete per-task asset integrity.
- `containers/robotwin-cu121.sif` exists. Its existing interpreter was located
  at `/usr/bin/python` by inspecting the image; `/opt/conda/bin/python` does not
  exist. The image was not modified. Its full Motus/CUDA/Transformers dependency
  compatibility remains unverified.
- No GPU allocation was active. No GPU job was submitted because the required
  model assets are not configured. GPU availability and sufficient VRAM have
  therefore not been established.

Exact preflight command used, from the Motus root (exit status 1 as expected):

```bash
PYTHONPATH=/tmp/motus-spline-checks \
  /home/kding/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11 \
  experiments/spline/evaluate.py \
  --checkpoint ... --wan-path ... --vlm-path ... \
  --output experiments/spline/checks/preflight --preflight
```

## Untested behavior

Broader task/seed coverage; existing GUI playback; physical demo timing
calibration; and task-specific gripper sensitivity
to faster sampling. Original and dense 1× rollouts have matching observed success,
action counts, and physics-step counts for one scene; this is not a claim of
bitwise rollout parity or general task success.
No maximum-error claim is made between unqueried timestamps or about resulting
robot tracking error. See [README.md](README.md) for baseline, pilot, matrix, GUI,
and reporting commands and the direct spline-prediction extension points.

## Apptainer Slurm script check (2026-10-21)

Added `run_apptainer.sbatch` using the user's existing image, repository location,
and NVIDIA/Vulkan/EGL setup. Both the batch shell and its embedded container
shell pass `bash -n` and ShellCheck. A CPU dry run in that image, with both
repositories mounted at the batch script's paths, generated all six one-scene
matrix commands. No GPU job was submitted.

Actual container inspection found CPython 3.10.12, Torch 2.4.1+cu121, NumPy
1.26.4, and SciPy 1.15.3. The modern SciPy fitter imports. Importing Qwen3-VL
fails with `ModuleNotFoundError: No module named 'transformers'`. Merely adding
the existing `.runtime/python` directories does not establish a working install:
package discovery there found namespace directories without usable metadata.
The batch script therefore accepts a prepared dependency directory explicitly
via `MOTUS_EXTRA_PACKAGES` or another image via `MOTUS_SIF`, and checks real
symbols before model loading. Complete Motus imports and GPU execution remain
untested. The checkpoint, Wan/T5, and Qwen asset paths are still required inputs.

## Prepared runtime and pilot (2026-10-21)

The initial dependency/model blockers above have been resolved. Extra Python
dependencies are installed under `Motus/.runtime/apptainer`, preserving the
container's Torch 2.4.1+cu121, TorchVision 0.19.1+cu121, NumPy 1.26.4, and SciPy
1.15.3. The image and Motus inference source remain unchanged. Added dependencies
include Transformers 4.57.1, Diffusers 0.36.0, and DeepSpeed 0.16.9. See the
[successful CPU runtime check](checks/container-runtime.txt).

Downloaded about 29 GiB of required public inference assets under
`Motus/pretrained_models`, excluding unused Wan diffusion and Qwen backbone
weights. The checkpoint configuration matches the existing deployment settings.
Every required asset passed [container preflight](checks/container-preflight/preflight.json).
All **38 focused tests pass with the actual container's SciPy 1.15.3**; see
[test output](checks/container-pytest.txt). Setup/batch shells pass Bash and
ShellCheck; the runtime checker passes Ruff lint/format.

Initial one-L40S job **6505604** passed model imports, rendering, model loading,
and expert scene validation, but the first policy call failed: the deployed
inference copy of Wan attention requires FlashAttention. The job exited FAILED
after 6m41s, before any policy action was executed. This is a runtime failure,
not a measured task-success result. It used a sampled peak of about 37 GiB GPU
memory and 46 GiB host RAM. Its logs remain under `runs/slurm-6505604`.

Installed the official prebuilt FlashAttention 2.7.4.post1 wheel for CPython
3.10, Torch 2.4, CUDA 12, CXX11 ABI=False. CPU import passes. This preserves
Motus's existing attention implementation. The runtime check now verifies this
dependency and exercises a tiny GPU kernel before loading the full model.
Failed-inference latency is also recorded; the additional regression test passes.

Retry job **6506539** requests one L40S, eight CPUs, 64 GiB host RAM, and two
hours to accommodate the measured startup cost; see [receipt](checks/slurm-retry-submission.txt).
It runs the original execution followed by all six matched settings on
`click_alarmclock/demo_randomized`, scene 4300000, policy seed 42. The job
completed successfully in **36m01s**, including model initialization. The
original baseline and all six matrix episodes succeeded on that scene.
Initial rollout details:

| Execution | Success | Physics steps | Simulated seconds | Episode wall seconds | Policy calls |
| --- | --- | --- | --- | --- | --- |
| Original | yes | 2459 | 9.836 | 23.478 | 2 |
| Dense 1× | yes | 2459 | 9.836 | 18.061 | 2 |
| Spline 1× | yes | 2415 | 9.660 | 19.139 | 2 |

Episode wall time excludes model loading. Both dense paths issued 21 effective
action calls. Spline 1× had no fallbacks, took 0.00437 s fitting and 0.00075 s
decoding in total, and used 16 then 15 control points for two 16-sample chunks.
These are individual pilot observations, not a multi-seed success estimate.
The complete matrix is saved in
[summary.json](runs/slurm-6506539/summary.json):

| Adapter | Requested speed | Success | Simulated seconds | Episode wall seconds | Achieved speedup vs dense 1× | Fallback chunks |
| --- | --- | --- | --- | --- | --- | --- |
| dense | 1× | 1/1 | 9.836 | 18.061 | 1.000× | 0 |
| spline | 1× | 1/1 | 9.660 | 19.139 | 1.018× | 0 |
| dense | 1.5× | 1/1 | 8.168 | 17.725 | 1.204× | 0 |
| spline | 1.5× | 1/1 | 8.188 | 17.486 | 1.201× | 1 |
| dense | 2× | 1/1 | 7.040 | 17.360 | 1.397× | 0 |
| spline | 2× | 1/1 | 6.988 | 17.565 | 1.408× | 0 |

Spline 1.5× fell back to matched dense interpolation for its first chunk:
maximum error at execution query times was 0.01227 rad, above the 0.01 rad
per-channel tolerance. The second chunk used the spline. Source-sample error
alone would not have caught this. The reported spline 1.5× result therefore
includes that fallback. Requested 2× yielded about 1.4× achieved simulated
speedup; unchanged TOPP/controller timing makes the two quantities different.
This single scene does not establish success rates across scenes or tasks.

A startup-inclusive GPU snapshot (256 samples at five-second intervals) showed
1.97% mean utilization, 75% peak utilization, and 42363 MiB peak GPU memory.
Repeated model initialization dominates wall time; the successful policy
episodes themselves took about 18–23 s. Further jobs use one L40S sequentially,
with no parallel GPU sweep. Inference/model initialization are unchanged.

Retry output: `/project/6101778/kding/motus-spline-6506539.out`.
Retry metrics: `experiments/spline/runs/slurm-6506539/`.

## Five requested task pilots (2026-10-21)

Submitted one scene per task, using `demo_randomized`, scene seed 4300000,
policy seed 42, and the same frozen RoboTwin-finetuned checkpoint. Each job runs
the original baseline and all six dense/spline × 1×/1.5×/2× configurations.
The pilot dependency has now completed successfully. `adjust_bottle` is running;
the remaining four wait for the preceding task job to finish, so at most one of
these evaluations uses a GPU at once. A task-specific failure does not suppress
the other requested tasks.

| Task | Slurm job |
| --- | --- |
| adjust_bottle | 6507360 |
| grab_roller | 6507361 |
| handover_block | 6507362 |
| hanging_mug | 6507363 |
| lift_pot | 6507364 |

Each requests one L40S, eight CPUs, 64 GiB host RAM, and two hours. Receipt:
[slurm-five-task-submission.tsv](checks/slurm-five-task-submission.tsv).
Results will be under `experiments/spline/runs/slurm-JOBID/`; console output is
`/project/6101778/kding/motus-spline-JOBID.out`. No results for these five tasks
are available at submission time. Rejected scenes remain explicit failures;
the harness does not silently substitute a different scene seed.
