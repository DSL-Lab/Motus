# Frozen Motus spline pilot: recorded results

This report summarizes the saved Slurm runs from 2026-10-21. It was checked
against episode JSONL records, process exit codes, rejected-scene journals,
the six-mode summary files, and Slurm accounting. All seven jobs have ended;
none was in the queue when the report was prepared.

## Scope and counts

- Six task scenes were requested, using scene seed **4300000** for each task.
- Three tasks completed the full comparison: `click_alarmclock`, `adjust_bottle`,
  and `handover_block`. Each has six comparison episodes plus one original
  execution reference: **18 successful comparison episodes + 3 successful
  original references = 21 successful policy rollouts**.
- One earlier attempt failed during its first inference because FlashAttention
  was missing. It executed zero policy actions and advanced zero policy physics
  steps. This is an additional runtime failure, not a task-performance sample.
- `grab_roller`, `hanging_mug`, and `lift_pot` failed the evaluator's expert scene
  validation on the requested seed. Policy execution did not begin, and their
  six-mode comparisons did not run. They supply no policy success-rate result.
- The run folders contain 22 started-policy episode records, of which 21
  succeeded; the three expert-rejected scenes are logged separately. There were
  25 policy subprocess attempts in seven Slurm jobs: three completed jobs and
  four failed jobs. No new training demonstrations or fine-tuning were produced.

This is one scene per evaluated task, not a multi-seed benchmark. The observed
18/18 comparison successes must not be generalized to unseen scenes or to the
three tasks whose scene validation failed.

## Setup

Frozen `motus-robotics/Motus_robotwin2` in the WAM-RoboTwin simulator fork;
`demo_randomized`; policy seed **42**; instruction type `unseen`. Per task,
the requested scene, instruction, model paths, and policy seed match across the
six variants. The model retains joint future-video/action inference, eight
predicted video frames, 16 dense actions per policy call, and ten denoising steps.

Runs used one L40S at a time, eight CPUs, 64 GiB host RAM, and the existing
`robotwin-cu121.sif` with prepared additional Python dependencies and FlashAttention.
Completed jobs requested two hours each. Physics timestep was approximately
0.004 s. Original physics, controller settings, task definitions, success checks,
and command-count episode limits remain in use.

## Simulated completion times

All 18 entries below succeeded. Times are seconds derived from actual policy
physics steps times the scene timestep, excluding scene setup/expert validation.

| Execution | click_alarmclock | adjust_bottle | handover_block |
| --- | ---: | ---: | ---: |
| Dense 1× | 9.836 | 24.504 | 100.388 |
| Spline 1× | 9.660 | 23.944 | 76.764 |
| Dense 1.5× | 8.168 | 20.128 | 55.280 |
| Spline 1.5× | 8.188 | 20.068 | 61.384 |
| Dense 2× | 7.040 | 16.992 | 51.316 |
| Spline 2× | 6.988 | 16.920 | 53.768 |

| Task | Dense 2× achieved speedup | Spline 2× achieved speedup | Spline 2× episode wall time | Spline 2× wall speedup |
| --- | ---: | ---: | ---: | ---: |
| click_alarmclock | 1.397× | 1.408× | 17.565 s | 1.028× |
| adjust_bottle | 1.442× | 1.448× | 22.237 s | 1.095× |
| handover_block | 1.956× | 1.867× | 60.378 s | 1.443× |

Every speedup is relative to the same task's matrix dense 1× episode. Episode
wall times exclude model loading and expert scene validation. Full batch jobs
took 36m01s, 37m07s, and 42m04s respectively, including repeated model loading.

Dense 1× and the original execution reference matched success, effective action
counts, policy-call counts, and physics steps for all three evaluated scenes:

| Task | Original and dense 1× physics steps | Effective action calls | Policy calls |
| --- | ---: | ---: | ---: |
| click_alarmclock | 2459 | 21 | 2 |
| adjust_bottle | 6126 | 43 | 3 |
| handover_block | 25097 | 252 | 16 |

Accelerated dense interpolation produced similar improvements to accelerated
splines. At requested 2×, splines were slightly faster on the first two tasks
and slower on `handover_block`. These observations do not establish a consistent
benefit from spline fitting beyond temporal resampling.

## What was rewritten

Adaptation occurs online for each newly predicted chunk:

1. Frozen Motus predicts 16 absolute, 14-channel actions from the current
   observation and instruction.
2. Prepend the observed initial state at source time zero. Fit one cubic
   B-spline to the 12 arm-joint channels, with shared adaptive knots and source
   timestamps. Preserve the two scalar gripper streams/events separately.
3. Keep the fitted coefficients on the original source timeline. Select query
   timestamps according to `source_time = speed_factor * nominal_execution_time`.
   Evaluate the spline at these timestamps, then combine the arm positions with
   held gripper values and preserved open/close events into 14-channel qpos targets.
4. Send these targets through the original TOPP-based `take_action` executor.
   TOPP generates low-level joint trajectories, steps physics, and performs the
   original success checks. After the chunk, obtain a fresh observation and replan.

Thus the pipeline is dense action chunk → fitted spline → resampled joint targets
→ existing simulator controller. Control points parameterize the curve;
evaluation produces the actual commanded positions. The model does not predict
spline parameters directly. No weights or training pipeline were changed.

The nominal source spacing is 0.1 s, derived from the converted Motus frame-index
timeline. A requested 2× ordinarily queries at 0.2, 0.4, …, 1.6 instead of
0.1, 0.2, …, 1.6, with extra gripper-event timestamps as needed. The full original
source endpoint is preserved before replanning. Each command's actual duration
is still chosen by TOPP, so requested 2× is not a promise of twice the physical
execution rate. Recorded video FPS is not a calibrated physical demonstration
clock. Saved spline parameters do not change with the requested speed.

The accelerated dense control uses linear interpolation on exactly the same
query schedule. Dense 1× retains the original array and execution calls.
Later chunks can differ between variants because observations and replanning
are closed-loop; this is not a replay of a single fixed whole-episode trajectory.
Future-video generation and its original timeline remain unchanged.

## Fitting diagnostics

The runs used cubic adaptive SciPy fitting, 0.01 rad maximum tolerance per arm
channel, normalized smoothing budget 1.0, and no fixed control-point cap. The
maximum error is checked both at source samples and at executed query timestamps
against the corresponding dense path. This is not an all-time tracking guarantee.

- Nine spline episodes attempted **50 chunk fits**. **47 were accepted** and
  **three used dense fallback**: one in alarmclock 1.5× and two in handover 1.5×.
- All three rejected fits exceeded the execution-query tolerance (approximately
  0.01227, 0.01106, and 0.01150 rad). Their source-sample errors alone were within
  tolerance. Reported spline episode times include these fallbacks.
- Accepted fits had worst observed source-sample error 0.004679 rad and worst
  execution-query error 0.009312 rad.
- Fitting averaged **2.313 ms/chunk** and decoding/checking **0.377 ms/chunk**.
- Candidates used **14–17 control points**, mean **15.38**, for 16 dense predicted
  samples plus the initial observation anchor. Knot counts were 18–21. Strong
  parameter-count compression was not demonstrated by this fit setting.

Dense inference is still performed in full on every policy call. Changing the
executed trajectory can change how many policy calls a task needs; it does not
make an individual Motus forward pass cheaper.

## Raw sources

| Task / attempt | Job | Result location |
| --- | --- | --- |
| click_alarmclock completed pilot | 6506539 | [summary](runs/slurm-6506539/summary.json) |
| adjust_bottle completed pilot | 6507360 | [summary](runs/slurm-6507360/summary.json) |
| handover_block completed pilot | 6507362 | [summary](runs/slurm-6507362/summary.json) |
| grab_roller expert rejection | 6507361 | [journal](runs/slurm-6507361/original/original_1x/seed4300000/_rejected_seeds.jsonl) |
| hanging_mug expert rejection | 6507363 | [journal](runs/slurm-6507363/original/original_1x/seed4300000/_rejected_seeds.jsonl) |
| lift_pot expert rejection | 6507364 | [journal](runs/slurm-6507364/original/original_1x/seed4300000/_rejected_seeds.jsonl) |
| Initial FlashAttention runtime failure | 6505604 | [episode record](runs/slurm-6505604/original/original_1x/seed4300000/episodes.jsonl) |

The corresponding variant directories contain the raw `episodes.jsonl`,
`_episodes.jsonl`, `command.json`, `process.json`, dense chunks, and fitted
parameter artifacts. The next evaluation step is more valid scene seeds per
task, including valid scenes for the three expert-rejected tasks, with the same
six-way matched comparison.
