# Staged Motus spline experiment

**Phase 1 is implemented as an opt-in adapter and evaluation scaffold, but is not
complete. No original Motus rollout or six-variant RoboTwin comparison has run.**
The original policy, models, and sibling simulator remain unchanged. No training
has been started. Phase 1 cannot establish inference speedup or reproduce the
complete B-spline Policy method.

## Baseline audit

Inspected Motus `88f6fe30c62af033874b0a2e48ad1cbdaf12cfdf` and the existing sibling
WAM-RoboTwin `87942e06cdf9968a41a5a1320946e10a9fdd706d`. The initial Motus working
tree was clean. Read `CONTRIBUTING.md`, inference guides, RoboTwin's `AGENTS.md`,
`INTEGRATION.md`, and `.agents/skills/use-wam-python/SKILL.md`. Use RoboTwin's
existing shared `robotwin-classic` interpreter; do not install Motus dependencies
into its simulator environment without first checking compatibility. This
implementation adds no dependency: Motus already requires NumPy and SciPy >=1.11.

| Component | Actual behavior and source |
| --- | --- |
| Entry point | [Original wrapper](../../inference/robotwin/Motus/deploy_policy.py), `get_model`, `eval`, `MotusPolicy.get_action`. The existing legacy sibling `script/eval_policy.py` selects this interface. XPolicyLab is a separate workflow and is not used. |
| Observations | Head and wrist images are concatenated and resized to 384x320. State is `joint_action.vector`. Instruction goes through T5 and Qwen; joint video/action denoising runs for 10 steps. Frame-grid generation/saving is part of original wrapper latency. |
| Action convention | 14 absolute joint-position/gripper targets: left joints 0:6, left analog gripper 6, right joints 7:13, right analog gripper 13. Joint angles are radians; these are neither end-effector rotations nor delta/velocity commands. No angular unwrapping is appropriate for bounded joint coordinates. |
| Normalization | Dataset `_load_robot_data` has action/state normalization commented out. Inference passes raw `current_state`, and returns `predicted_actions` directly. `current_state_norm` and wrapper normalization helpers exist but are unused by inference. Do not denormalize twice. |
| Horizon and prefix | `num_video_frames=8`, `video_action_freq_ratio=2`, so N=16 predicted actions. Original `eval` submits all 16; there is no shorter configured prefix. Early success or action limit makes subsequent `take_action` calls no-ops. The numerical adapter also supports an explicitly shorter prefix without extending its source span. |
| Dataset timeline | [Sampling](../../data/robotwin2/robotwin_agilex_dataset.py), `_calculate_sampling_indices`: with raw interval d, action sample i (one-based) is at `3*i*d`, i=1..16. Video sample j is at `6*j*d`, j=1..8. Short episodes clamp repeated targets to the last raw sample. |
| Physical timestamp caveat | Conversion writes MP4 at configurable/default 30 fps, but the sibling demo config saves every 15 physics steps with default 1/250 s. MP4 frame rate alone does not prove physical demonstration timing. Released-data timestamps must be checked before assigning d in seconds. The Phase 1 harness explicitly records action-sample indices, not invented physical timestamps. |
| Video VAE | [VAE](../../bak/wan/modules/vae2_2.py), `encode`: first frame separately, then groups of four. Nine pixel frames (condition + eight future) become three temporal latents (condition + two future). Causal convolution/cache receptive fields overlap; a future latent is not a single video sample or action. |
| Video tokens | [Motus](../../models/motus.py), `grid_sizes`: three temporal grids, spatial grids 384/32 by 320/32. Actual checkpoint patch settings must agree. WAN applies 3-D RoPE to video Q/K. |
| Action tokens | [Action encoder](../../models/action_expert.py): one state + 16 action + four register tokens = 21 tokens entering each of 30 action blocks. A shared per-action MLP encodes 14 values. Positions are fixed 1-D sinusoidal sequence indices, including state/register slots; they are not physical-time embeddings. Noise-time embeddings/AdaLN are separate. |
| Multimodal attention | [Joint attention](../../models/motus.py), `VideoModule.process_joint_attention`, and [WAN](../../bak/wan/modules/model.py), `WanSelfAttention.forward`: concatenate video/action/understanding QKV and use full bidirectional joint attention with lengths. This path has no frame-to-action causal alignment mask. Only video receives WAN RoPE; action gets the encoder's additive positions. Qwen's text/image padding mask and positions apply in its feature extraction, not a video/action temporal mask. |
| Executor | Sibling `envs/_base_task.py`, `take_action(qpos)`: read actual joints, TOPP each arm to the next absolute target at 1/250 s, then step physics until both paths finish. Planner failures use a 50-step fallback. Grippers are interpolated over the resulting arm-specific path lengths. `robot.set_gripper` clips open fraction to [0,1] and preserves its existing rate-limiting behavior. Success is checked after each physics step. |

The deployment tree contains its own copies of `models`, `utils`, and `bak`.
The harness deliberately imports that existing deployment package. Future
training changes belong in root `models/` and must also reach deployment; editing
only one copy would leave evaluation on the wrong representation.

## Timing blocker

RoboTwin's **physics/controller rate** is 250 Hz, but its **policy target calls
are variable duration**. A constant 250 Hz internal TOPP sample rate does not
make `take_action` a fixed-period policy executor. Removing every other target
does not implement `source_time = source_start + 2 * execution_time`.

The numerical adapter implements one explicit map for dense interpolation and
spline evaluation: submit the first predicted target in slot zero, advance
source time by `speed_factor * source_period` each execution slot, and clamp the
last slot to the endpoint of the original executed prefix. The ceiling of the
number of intervals can add less than one target period. Dense 1x returns the
original dtype and bytes. Grippers use zero-order hold on the same source map,
keeping analog values and reporting multiple changes collapsed into a slot.
Spline fitting never smooths or thresholds the grippers. This does not replace
the executor's own gripper interpolation.

`run --speed-factor 1.5` and `2.0` therefore **fail explicitly** for the current
TOPP executor. The six requested variants are recorded in `phase1.json`; four
accelerated rollout variants remain blocked. A physically valid timing contract
must be resolved before enabling them. Reusing `take_dense_action` would also
change the execution path and its success/step-limit handling, so it is not
silently substituted. No physics, controller, task, or success criterion was
changed to work around this issue.

## Fitting and measurement

[Spline utility](../../inference/robotwin/spline_actions.py): adaptive clamped
cubic B-splines share one temporal knot vector across all 12 joint channels.
Least squares uses SciPy's basis matrix and NumPy's solver. Insert knots near
the worst tolerance-scaled reconstruction residual; allow up to 16 controls.
At the full budget use SciPy's cubic interpolant as the final candidate.
Check all original timestamps separately for every joint, including conversion
back to the controller dtype. Default tolerance is 0.01 rad per joint, an
explicit experimental setting that still needs replay validation. Failed,
rank-deficient, or nonfinite decoded fits use the dense path at the same speed
map, and record the reason. Nonfinite original actions are rejected, not passed
to the robot. No boundary alignment is applied.

Inspected [BSP's fitter](https://github.com/B-spline-policy/bspline-policy/blob/main/bspline_policy/bspline_policy/common/bspline_action.py)
and [MIT license](https://github.com/B-spline-policy/bspline-policy/blob/main/LICENSE).
It searches knots with SciPy's `generate_knots`, fits coordinated trajectories,
and also carries replay-buffer/training dependencies. Its failure behavior does
not provide this experiment's strict fallback contract. Motus's supported SciPy
minimum lacks `generate_knots`. No BSP implementation was copied or vendored;
the small SciPy-based implementation avoids those dependencies.

[Recorder](../../inference/robotwin/spline_rollout.py) wraps the existing policy
extension points. A forwarding scene object counts the real `scene.step()` calls
and actual `get_timestep()` without changing them. Mock-executor checks cover
this forwarding; SAPIEN runtime compatibility is untested. Every chunk saves
predictions, submitted targets, and source times in an NPZ failure/replay artifact.

Measurements are:

- Per-episode success from existing criteria; simulated execution seconds from
  actual physics steps; wall seconds from episode reset through final policy
  execution, including observations, inference, rendering and logging. Setup,
  expert scene validation, model loading, and environment teardown are excluded.
- Policy calls, `get_action` wall time, observation-capture time, and latency from
  completed observation acquisition to first action submission. The latter
  includes wrapper preprocessing, T5/Qwen/video/action inference, frame saving,
  adapter work and NPZ logging. Original action `.cpu().numpy()` synchronizes
  action production; no synthetic CUDA timing is substituted.
- Adapter processing seconds, per-joint source reconstruction error, fallback
  frequency/reasons, original and representation payload bytes, and debug NPZ
  file bytes. Payload counts use float64 knots/coefficients and original gripper
  dtype; uniform source timing/layout is shared metadata. Debug artifacts retain
  dense arrays, so these statistics do not claim reduced actual debug storage,
  fewer model tokens, lower model memory, or faster inference.
- Command position jumps from the last actually submitted target to the first
  new target. Velocity jumps are null for TOPP because target timing varies and
  desired boundary velocities are not exposed. The utility can compute secant
  velocity jumps if a genuine fixed target period is supplied.

Model-generated videos remain on the original model timeline. Spline
postprocessing does not make them predictions of accelerated physics.

## Commands

Keep `Motus/` and `WAM-RoboTwin/` as siblings. On the existing GPU cluster, run
inside a SLURM allocation as required by RoboTwin's integration guide. Resolve
the existing shared environment; set actual checkpoint/backbone locations:

```bash
cd /path/to/Motus
ENV_ROOT=${ROBOTWIN_UV_ENV_ROOT:-$(dirname "$PWD")/uv_envs}
CLASSIC_ENV=${ROBOTWIN_CLASSIC_ENV:-${ENV_ROOT}/robotwin-classic}
export MOTUS_CHECKPOINT=/path/to/released/Motus-RoboTwin
export MOTUS_WAN=/path/to/Wan2.2-TI2V-5B
export MOTUS_VLM=/path/to/Qwen3-VL-2B-Instruct
COMMON=(--robotwin ../WAM-RoboTwin --checkpoint "$MOTUS_CHECKPOINT"
        --wan "$MOTUS_WAN" --vlm "$MOTUS_VLM")

"$CLASSIC_ENV/bin/python" -m inference.robotwin.spline_experiment check "${COMMON[@]}"
"$CLASSIC_ENV/bin/python" -m inference.robotwin.spline_experiment plan

# Required original smoke rollout FIRST, on the existing subset's first task.
"$CLASSIC_ENV/bin/python" -m inference.robotwin.spline_experiment run "${COMMON[@]}" \
  --mode original --task adjust_bottle --split demo_clean --seed 42 --episodes 1 \
  --output logs/spline/smoke_original

# Only after the original rollout succeeds, compare identical scene/instruction.
"$CLASSIC_ENV/bin/python" -m inference.robotwin.spline_experiment run "${COMMON[@]}" \
  --mode dense --task adjust_bottle --split demo_clean --seed 42 --episodes 1 \
  --baseline logs/spline/smoke_original --output logs/spline/smoke_dense_1
"$CLASSIC_ENV/bin/python" -m inference.robotwin.spline_experiment run "${COMMON[@]}" \
  --mode spline --task adjust_bottle --split demo_clean --seed 42 --episodes 1 \
  --baseline logs/spline/smoke_original --output logs/spline/smoke_spline_1
"$CLASSIC_ENV/bin/python" -m inference.robotwin.spline_experiment summarize --output logs/spline
"$CLASSIC_ENV/bin/python" -m unittest discover -s tests -v
```

For the campaign, reuse the 12 tasks from sibling
`experiments/fastwam12/tasks.txt` (also identical to its DP12 subset), both splits,
seed 42 and 20 episodes per condition from `phase1.json`. Repeat the explicit
original/dense/spline commands for each task and split with matching episode
counts and fresh output directories. An original baseline must match revisions,
inference-source digest, checkpoint path/size/mtime, task, split, instruction type,
seed and episode count. Checkpoint identity uses file metadata, not a costly full
checkpoint hash. Completed comparisons also require identical scene seeds and
instructions. No incomplete run is automatically resumed. Preserve rejected-scene
journals and raw artifacts. `summary.json` includes successes and failures;
successful completion averages must always be read with success rate. Use common
successful matched scenes for achieved speedup, rather than ratios of nominal
speed factors or different survivor sets.

The CLI never launches a campaign or training automatically. See
[RESULTS.md](RESULTS.md) for actual local checks and blockers, and
[PHASE2.md](PHASE2.md) for the gated learned-representation design.
