# Frozen Motus action fitting experiment

Motus weights, joint video–action inference, future-video generation, RoboTwin
physics, controller limits, task definitions, and success checks are unchanged.
Implementation lives in `Motus/motus_actions/`; the sibling WAM-RoboTwin checkout
contains a small `policy/Motus` bridge and optional episode measurement hooks in
its existing evaluator. The repositories keep their own Git histories. No model
or source tree is copied into the other repository.

## Traced baseline

- `inference/robotwin/Motus/deploy_policy.py`: three camera views become a
  384×320 image; T5 language features and Qwen3-VL inputs condition joint
  video/action denoising. `models/motus.py::inference_step` uses ten Euler
  denoising steps under the checked-in configuration. Eight future video frames
  and 16 dense actions are produced together. All 16 actions are executed before
  obtaining a new observation and replanning; early success/step limit can stop
  physical execution within that prefix.
- Action layout is `[L1..L6, left_gripper, R1..R6, right_gripper]`. The 12 arm
  channels are **absolute joint positions in radians**, not deltas, Cartesian
  rotations, or velocity commands. Joint rotations follow the URDF joint
  coordinates; the adapter does not unwrap bounded joints. No quaternion/Euler
  conversion is involved. Grippers are scalar targets, 0 closed / 1 open. The
  simulator clips them to [0,1] and applies its original drive behavior.
- Training dataset normalization is commented out. Deployment computes
  `current_state_norm` but supplies raw `current_state`; predicted actions are
  passed directly to `take_action(..., action_type="qpos")` without unnormalizing.
  The adapter operates on those physical-unit outputs and adds no normalization.
- `WAM-RoboTwin/envs/_base_task.py::take_action` plans separately from each arm's
  measured current qpos to its target using MPLib TOPP, sampled at 1/250 s.
  Planning failure/empty output selects the existing 50-sample fallback. A loop
  advances the two arm trajectories and the original gripper interpolation,
  stepping physics and checking success after every iteration. **One action call
  has variable duration**, depending on the planned sample counts, arm scheduling,
  and early success. It is not one physics tick, nor a fixed number of ticks.
- `_step_scene` increments `live_sim_step` on every `scene.step()`. Measurements
  use the difference of this counter across the policy episode multiplied by
  `scene.get_timestep()` (default 0.004 s). Expert validation and scene setup are
  excluded. Render and task settings remain those of the existing configuration.
- The upstream `tasks_all.txt` lists all 50 tasks; `eval.sh` hardcodes
  `click_alarmclock` (despite the README suggesting a positional task argument).
  `paths_config.yml` selects `demo_randomized` and seed 42, but its model/runtime
  paths are `"..."`. The legacy evaluator normally begins scene seeds at
  `100000*(1+seed)` = 4300000 and evaluates 100 accepted scenes. It validates
  scenes with the expert and normally substitutes later seeds for rejected scenes.
  The new pilot selects exactly 4300000, 4300001, 4300002, one subprocess per seed,
  and prevents substitution with `ROBOTWIN_EVAL_MAX_ATTEMPTS=1`. Inspect rejected
  scene journals separately; no policy episode exists for an invalid scene.

## Source timeline and actual speed

The dataset selects actions at `condition_index + (i+1)*3`, and video frames at
every second selected action. Its converter writes video at 30 FPS. Consequently
the default `source_dt=0.1` is an explicit **nominal frame-index timeline**, with
actions at 0.1,…,1.6 and video frames at 0.2,…,1.6. Deployment itself originally
had no timestamp/duration contract. Converter FPS does not prove physical demo
timing: this fork saves demonstrations every 15 physics iterations plus motion
boundaries, and the Motus converted data discard physical timestamps. Use actual
demonstration timestamps with the mathematical encoder when available. Do not
claim that 1.6 nominal source seconds equals 1.6 simulated seconds.

Each fit includes measured initial qpos at source time zero and the full predicted
prefix at its original source timestamps. The sampling schedule maps a nominal
execution grid through `source_time = speed_factor * execution_time`, preserves
the exact original endpoint, and inserts every gripper open/close edge. The final
nominal interval can be shorter; gripper events can add commands. No initial
zero-time command is dispatched. Dense interpolation and spline decoding share
this schedule. Dense 1× returns the original action array unchanged and follows
the original `take_action` loop, including its no-op calls after early success.

The smallest integration compatible with the existing executor is **resampling
its absolute position waypoints while retaining TOPP**. The nominal grid is not
a real-time dispatch deadline. TOPP still chooses actual command durations; a
requested 1.5× or 2× therefore does not guarantee that much simulated speedup.
Per-command physics steps and episode simulated duration are logged. Fewer
waypoints may reduce repeated trajectory starts, but measured speedup can be
smaller, absent, or negative. Gripper events and failures can further limit it.
The command-count episode limit remains the existing task's limit.

## Spline and gripper conventions

`motus_actions/spline.py` has no policy, camera, simulator, or logging imports:

```python
from motus_actions.spline import fit_spline, evaluate_spline, save_spline, load_spline

spline = fit_spline(source_times, continuous_actions, tolerances,
                    smoothing=1.0, max_control_points=None)
decoded = evaluate_spline(spline, requested_source_times)
save_spline("parameters.npz", spline, source_times)
restored, timestamps = load_spline("parameters.npz")
```

`ActionSpline` contains degree 3, a nondecreasing knot vector, control points
`[number_of_control_points, continuous_channels]`, and the active source-time
domain. All parameters use original source timestamps and physical action units.
NPZ serialization uses numeric arrays, without pickle. Decoder extrapolation and
nonfinite/invalid parameters are rejected. The encoder accepts nonuniform
timestamps and ground-truth demonstration actions as well as model outputs.

Fitting uses SciPy >=1.15 `make_splprep`, with one shared timestamp parameter and
knot vector across all 12 arm channels. Channels are divided by their explicit
per-channel physical tolerances during fitting; returned control points are
converted back immediately. `fitting_smoothing` is normalized summed-square
residual budget, **not** a maximum-error guarantee. Maximum absolute error and
RMS error are checked at every source sample, including the observation anchor.
The adapter also checks maximum error relative to dense linear interpolation at
every executed query timestamp, rejecting cubic inter-sample overshoot there.
This is not a continuous-domain bound between all possible timestamps.

Default tolerances are 0.01 radians per arm channel, explicitly ordered
`L1..L6,R1..R6`. Configure all 12 values; zero, negative, nonfinite, or incorrectly
sized tolerances are configuration errors. `max_control_points` is optional and
must be >=4. SciPy's knot budget is a target and may overshoot; the returned
control-point count is checked as a hard acceptance budget. Numerical exceptions,
too-short cubic chunks, excess reconstruction error, or excess output budget use
the corresponding dense path. At 1× this is the exact original array; at faster
settings it is the same dense interpolation schedule as the comparison arm.
Failures and available reconstruction errors are recorded explicitly.

Grippers never enter spline fitting. The adapter stores their original scalar
samples separately and uses zero-order hold. A threshold of 0.5 identifies every
open/closed edge and inserts its original source timestamp into the schedule.
It does not binarize the commanded scalar. Both hands' event order is retained,
including open–close transitions between ordinary accelerated samples. Scalar
variations within one open/closed state are sampled at the selected timestamps;
the original executor still interpolates toward each selected gripper target.
This adapter supports absolute joint-position actions only. Pose, delta, and
velocity policies need an explicit convention-specific adapter before reuse;
they must not be treated as absolute qpos or naively interpolated as rotations.

The maintained SciPy implementation and its BSD-3-Clause license were inspected:
[implementation](https://github.com/scipy/scipy/blob/v1.16.3/scipy/interpolate/_fitpack_repro.py),
[license](https://github.com/scipy/scipy/blob/v1.16.3/LICENSE.txt),
[fitting API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splprep.html).
No BSP or SciPy source code is vendored and no custom knot selection is added.

## Commands

Use the existing simulator environment as specified by WAM-RoboTwin's
`.agents/skills/use-wam-python/SKILL.md`. Install only the additional numerical
dependency into that configured environment after checking its compatibility:

```bash
cd /project/6101778/kding
export ROBOTWIN_CLASSIC_ENV="${ROBOTWIN_CLASSIC_ENV:-$PWD/uv_envs/robotwin-classic}"
export MOTUS_ROOT="$PWD/Motus"
export SIM_PYTHON="$ROBOTWIN_CLASSIC_ENV/bin/python"
uv pip install --python "$SIM_PYTHON" -r Motus/experiments/spline/requirements.txt

# Required real asset locations; none are configured in the checked-in Motus paths.
export MOTUS_CHECKPOINT=/absolute/path/to/checkpoint_directory
export WAN_PATH=/absolute/path/to/Wan2.2-TI2V-5B
export VLM_PATH=/absolute/path/to/Qwen3-VL-2B-Instruct

"$SIM_PYTHON" Motus/experiments/spline/evaluate.py \
  --checkpoint "$MOTUS_CHECKPOINT" --wan-path "$WAN_PATH" --vlm-path "$VLM_PATH" \
  --output Motus/experiments/spline/runs/preflight --preflight
```

The full environment also needs the original Motus deployment dependencies,
including a compatible Torch/CUDA stack and Qwen3-VL-capable Transformers,
and the original RoboTwin/SAPIEN/MPLib/CuRobo stack. The existing container is
`containers/robotwin-cu121.sif`; `WAM-RoboTwin/robotwin.sh` describes its bindings.
Do not assume that image satisfies Motus's newer dependencies. The launcher
accepts `--sim-python` for an existing compatible runtime; it does not build one.

On this cluster, obtain a short single-GPU allocation before actual rollouts:

```bash
srun --account=aip-rjliao --time=00:30:00 --cpus-per-task=8 \
  --mem=64G --gres=gpu:l40s:1 --pty bash -l
# Re-export the paths above in this allocated shell if necessary.
cd /project/6101778/kding

# Unchanged upstream execution body, one scene, before comparing adapters.
"$SIM_PYTHON" Motus/experiments/spline/evaluate.py \
  --checkpoint "$MOTUS_CHECKPOINT" --wan-path "$WAN_PATH" --vlm-path "$VLM_PATH" \
  --original --scene-seeds 4300000 --output Motus/experiments/spline/runs/original

# Six variants, one task and three exact requested scenes; identical policy seed.
"$SIM_PYTHON" Motus/experiments/spline/evaluate.py \
  --checkpoint "$MOTUS_CHECKPOINT" --wan-path "$WAN_PATH" --vlm-path "$VLM_PATH" \
  --task click_alarmclock --task-config demo_randomized \
  --scene-seeds 4300000 4300001 4300002 --policy-seed 42 \
  --output Motus/experiments/spline/runs/matrix

"$SIM_PYTHON" Motus/experiments/spline/summarize.py \
  Motus/experiments/spline/runs/matrix
```

Start with `--scene-seeds 4300000` for the six-mode pilot before extending to
three scenes. Inspect memory/utilization during the allocated run (for example
`nvidia-smi -l 5` in that allocation), stop if incompatible, and do not reserve
more GPUs without measurements. The L40S pilot has completed original/dense/spline
1× episodes, with a sampled peak of 42363 MiB GPU memory. See VALIDATION.md.
Add `--dry-run` to print the exact subprocess commands without loading the model.
Use `--adapters dense --speeds 1` for dense 1× alone, or `--adapters spline
--speeds 1.5` for one spline setting. `--source-dt`, `--tolerances` (12 values),
`--smoothing`, and `--max-control-points` expose the fit options. Use fresh output
directories; the launcher refuses to overwrite completed experiment logs.

For inspection, use the existing GUI with one separate episode. Export
`ROBOTWIN_LIVE_PREVIEW_PATH=/absolute/path/to/live-preview.jpg` and
`ROBOTWIN_EVAL_VIDEO_CONTROL_STRIDE=5` before the single-variant command. In a
second terminal, run:

```bash
"$SIM_PYTHON" WAM-RoboTwin/scripts/robotwin_live.py \
  --snapshot /absolute/path/to/live-preview.jpg --port 8765
```

Open the existing dashboard at `http://localhost:8765` (forward the port from the
compute host when needed) to inspect its cameras, robot, and saved timeline.
The viewer's task-start controls belong to its existing policy workflow; start
Motus with the scripted command above. Use separate output for measured runs.
No GUI features were added. Predicted future videos remain on Motus's original
timeline and are not validated forecasts of accelerated execution.

## Slurm with the existing Apptainer image

Prepare missing inference packages on CPU before reserving a GPU:

```bash
cd /home/kding/projects/aip-rjliao/kding
bash Motus/experiments/spline/prepare_apptainer.sh
source Motus/.runtime/apptainer/env.sh
```

This stages extra Python packages under `Motus/.runtime/apptainer`, pins the
container's installed Torch/TorchVision/NumPy/SciPy versions, and checks inference
dependencies on CPU. Full deployment imports run in the GPU job because the
existing Wan T5 class initializes CUDA while importing. The SIF stays unchanged.
`bash Motus/experiments/spline/prepare_apptainer.sh --check` repeats the import
check without installing packages. DeepSpeed is needed by an existing inference
utility import; installing it here adds no training pipeline.

If model files are not already available, explicitly download the required
public inference assets (about 30 GB; no GPU required):

```bash
bash Motus/experiments/spline/prepare_apptainer.sh --download-models
source Motus/.runtime/apptainer/env.sh
sbatch Motus/experiments/spline/run_apptainer.sbatch
```

Downloads go under `Motus/pretrained_models` by default. They contain the
RoboTwin-finetuned Motus checkpoint, Wan config/VAE/T5/tokenizer, and Qwen
config/processor/tokenizer files. Unused Wan diffusion and Qwen backbone weights
are excluded. Existing downloads are reused. Set `MOTUS_MODELS_ROOT` to choose
another model storage directory, or export existing `MOTUS_CHECKPOINT`,
`WAN_PATH`, and `VLM_PATH` before running the helper without `--download-models`.
The generated `env.sh` sets the batch job's dependency and model paths.

[`run_apptainer.sbatch`](run_apptainer.sbatch) adapts the existing
`WAM-RoboTwin/robotwin.sh` environment setup for a noninteractive job. Defaults
are one L40S, eight CPUs, 64 GiB memory, one hour, task `click_alarmclock`, scene
4300000, and policy seed 42. It runs the original execution body once, then all
six dense/spline speed settings on the same scene. Pilot 6506539 was submitted
with `--time=02:00:00` to accommodate repeated frozen-model initialization.
See VALIDATION.md for measured episodes and the five subsequent task submissions.

```bash
cd /home/kding/projects/aip-rjliao/kding
export MOTUS_CHECKPOINT=/absolute/path/to/checkpoint_directory
export WAN_PATH=/absolute/path/to/Wan2.2-TI2V-5B
export VLM_PATH=/absolute/path/to/Qwen3-VL-2B-Instruct
sbatch Motus/experiments/spline/run_apptainer.sbatch
```

The checkpoint directory must contain `mp_rank_00_model_states.pt`. The three
model directories are mounted read-only at `/models/{motus,wan,qwen}`; ensure
their files are readable inside those mounts. Cached Hugging Face snapshots
with symlinks to an external `blobs` directory need that cache mounted as well
(or materialized model directories). Both repositories are mounted, and the
launcher explicitly selects `/workspace/RoboTwin` as the simulator root.

The batch script keeps the supplied `--nvccli`, `--writable-tmpfs`, `--cleanenv`,
Vulkan/EGL bindings, and NVIDIA capabilities. `NVIDIA_VISIBLE_DEVICES` selects the
allocated GPU for `--nvccli`; this single-GPU container uses CUDA ordinal zero
inside. The Slurm job ID is passed through `--cleanenv`. Output stays on the host:

```text
motus-spline-JOBID.out
Motus/experiments/spline/runs/slurm-JOBID/
  gpu-utilization.csv
  preflight/preflight.json
  original/original_1x/seed4300000/...
  matrix/{dense,spline}_{1,1.5,2}x/seed4300000/...
  summary.json
```

Per-episode `console.log`, `process.json`, and `episodes.jsonl` are beneath the
original/matrix variant directories. Startup/runtime failure of the original
episode stops the job before launching the matrix. Matrix failures keep their
logs and yield a failed batch exit status; a summary is attempted for any
measured episodes. GPU utilization/memory are sampled every five seconds.

**Container prerequisite:** a CPU inspection on 2026-10-21 found Python 3.10.12,
Torch 2.4.1+cu121, NumPy 1.26.4, and SciPy 1.15.3 in the supplied image.
`make_splprep` imports successfully, but `transformers` is not installed in the
base image. Use the CPU preparation helper above before submission, a compatible image
via `MOTUS_SIF`, or supply already-prepared Python 3.10 dependencies via
`MOTUS_EXTRA_PACKAGES=/absolute/path/to/packages`. That directory is mounted
read-only at `/opt/motus-python` and prepended to `PYTHONPATH`. The script checks
real Qwen3-VL, scientific, and simulator imports plus CUDA availability before
loading a checkpoint; it installs nothing during the GPU job. CPU dependencies
were verified with the prepared overlay. The deployed Wan code also requires
FlashAttention: preparation now installs its official prebuilt wheel for this
specific Python 3.10 / Torch 2.4 / CUDA 12 image, and checks a GPU kernel before
loading Motus. Pilot 6505604 exposed that missing dependency; retry 6506539 uses
the completed overlay. See VALIDATION.md for status.

After inspecting a successful one-scene pilot, expand within one allocation:

```bash
SCENE_SEEDS="4300000 4300001 4300002" \
  sbatch --time=03:00:00 Motus/experiments/spline/run_apptainer.sbatch
```

Override `TASK_NAME`, `TASK_CONFIG`, or `POLICY_SEED` through exported environment
variables. `MOTUS_RESULTS_ROOT` changes host result storage; `MOTUS_SIF`,
`MOTUS_WORKSPACE`, `ROBOTWIN_REPO`, `MOTUS_ROOT`, and `MOTUS_CONTAINER_PYTHON`
override runtime locations. Additional fitting flags after the filename are
forwarded to the existing launcher, for example:

```bash
sbatch Motus/experiments/spline/run_apptainer.sbatch \
  --max-control-points 12 --smoothing 1.0
```

## Measurements and results

Every measured policy episode writes `episodes.jsonl`: task/configuration, scene
and policy seeds, instruction, success/failure notes, actual physics steps and
seconds, wall seconds, policy calls, individual inference latencies, fitting and
decoding overhead, dense sample counts, per-chunk control-point/knot counts,
source/query timestamps, reconstruction errors, fallback counts, and boundary
jumps. Per-command physics counts expose variable command durations. Wall time
includes observation, original video grid writing, inference, adaptation,
artifact writing, and execution; model loading/expert validation/setup are excluded.
Inference latency covers the original `get_action` call, including language/video
work and its original image writing, with CUDA synchronization at its boundaries.

Each chunk saves dense actions, initial state, separate gripper samples/events,
query timestamps, and selected commands in `*_actions.npz`. Candidate spline
parameters are in `*_spline.npz` when fitting returns a candidate (consult the
fallback flag before interpreting it as executed). Episode logs are outside the
mathematical functions. `--original` uses upstream `eval` unchanged and records
episode timing/inference counts, without adapter chunk diagnostics. Exceptions
after episode start produce a failed episode record before the existing evaluator
re-raises. Earlier startup failures remain in `console.log`/`process.json`.

The report gives success rate and mean completion time **only among successful
episodes**, alongside achieved speedup on matched pairs successful in both dense
1× and the comparison. Pairing checks task/config, actual scene seed, policy seed,
instruction, and model asset paths. Requested speed is separate from achieved
simulated speedup. Rejected scenes/startup failures have no measured policy
duration; inspect their journals/process logs instead of treating them as times.
An empty report refuses to invent results. This experiment still pays the full
cost of dense joint video–action inference on every policy call.

See [VALIDATION.md](VALIDATION.md) for checks performed here and exact blockers.

### TCP position reconstruction

Adapted evaluations now automatically add `tcp_position_metrics` to each chunk
and episode in `episodes.jsonl`. Each arm reports Euclidean position RMSE and
maximum error in mm, sample count, and summed squared error in mm² for pooling.
`candidate_source` compares the candidate spline with its own dense prediction
at the original source samples (including the measured t=0 fitting anchor).
`candidate_query` uses dense **joint** linear interpolation at the saved query
source timestamps as reference, followed by FK. `dispatched_query` measures the
actual selected commands, including dense fallback. Physics-positive query
subsets exclude logged no-op calls; they still measure target reconstruction,
not actual motion or controller tracking. Accepted and rejected candidates are
pooled separately. Candidates rejected before returning parameters are marked
unavailable. Dense adapters have no spline-candidate measurements.

FK reuses the live robot's SAPIEN Pinocchio model, ordered by the configured arm
joint names. The TCP transform includes the end joint's frame in its child link,
`global_trans_matrix @ delta_matrix`, and the full `gripper_bias` (120 mm for
aloha-agilex). Each episode records the URDF hash and checks FK against RoboTwin's
TCP getters without writing robot state. Diagnostic failures are logged and leave
execution/fallback unchanged. `tcp_metric_seconds` records per-chunk diagnostic
overhead; wall time includes it. Original upstream execution has no chunk metrics.

No evaluation flags are needed: use the existing `evaluate.py`/Apptainer commands
above with a fresh output directory. To produce a table/CSV/JSON for future saved
chunks, or reproduce analysis of the three existing runs, see the exact CPU-only
commands in [offline_tcp/README.md](offline_tcp/README.md). That entire analysis
directory is removable; production logging does not import it. Actual tracking
error is unavailable because contemporaneous controller references and measured
TCP streams at simulator timestamps are not saved. Existing joint boundary-jump
diagnostics are not TCP tracking measurements.

## Direct spline prediction later

Replace `fit_spline(dense_output, ...)` with a producer of `ActionSpline` parameters
and timestamped gripper samples; reuse `evaluate_spline`, `decode_chunk`, the
sampling schedule, and the unchanged 14-channel simulator interface. The core
encoder already accepts ground-truth demonstrations and saved physical timestamps.
Adaptive fitting produces variable-size targets; no fixed target size is assumed.
A fixed-budget/fixed-knot target encoder can be added later without changing that
decoder or simulator interface. A SciPy decoder is sufficient for this experiment.
There is no training pipeline or differentiable decoder in this change.

Future fine-tuning should retain a shared **physical** timeline for videos and
actions: each frame corresponds to a timestamp where the action spline can be
evaluated, not to an individual control point. Preserve or reconstruct real
demonstration timestamps before treating converted video FPS as physical time.
