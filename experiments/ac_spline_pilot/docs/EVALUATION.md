# MOTUS A/C evaluation on Place Can Basket

## Existing evaluation

The submitted `train_dense_A.sh` and `train_splinewam_C.sh` jobs call
`experiments.ac_spline_pilot.trainer` with the expanded-data A and C
configurations. Both specify 150 training demonstrations, 20 validation
demonstrations, 15 epochs, a source-row stride of three, and eight future
video frames. The shared trainer constructs a held-out loader and evaluates
two batches at the configured validation interval. The pilot’s post-training
generation diagnostic samples fixed **training** windows and saves action and
spline validity plots. Those paths provide no TCP error or simulator success
measurement for the expanded comparison.

The pilot simulator adapter restores the saved MOTUS configuration, compatible
base weights, and adapter/action projection tensors. It uses the three RoboTwin
RGB views, absolute joint state, and task instruction. Variant A returns 16
dense commands. Variant C uses its saved adaptive spline decoder and returns
however many stride-three commands its supported execution interval provides.
The XPolicyLab RoboTwin evaluator queries the policy again at chunk boundaries,
passes each command through the existing joint/TOPP controller, and reports
`eval_success` when `place_can_basket.check_success()` is satisfied. This
predicate uses basket and can height, basket orientation, proximity, and
contact. The legacy logs contain policy, controller, and episode wall times.

## Checkpoints and cases

| Variant | Checkpoint |
|---|---|
| A, expanded-data benchmark | Expected under `outputs/ac_splinewam_A_170/train/robotwin_pilot_A_170/pilot-seed-0/`. No final bundle was found. Slurm job 7743139 was cancelled at training step 153, before the step-250 save, and 7743636 stopped at startup. |
| A, simulator smoke only | `outputs_backup/ac_pilot_7676974/A/robotwin_pilot_A/pilot-seed-0/checkpoint_step_1052`, from the older 45/5 split |
| C, expanded-data benchmark and smoke | `outputs/ac_splinewam_C_170/train/robotwin_pilot_C_splinewam_170/pilot-seed-0/checkpoint_step_3630` |

The expanded dataset has 512 fixed windows from 20 held-out demonstrations.
The original 100 scene seeds and instructions come from
`outputs_backup/ac_pilot_7676974/simulator/cases.json`. Both Slurm jobs read
that same manifest. Set `MOTUS_EVAL_CASES` to its new absolute path if the
archive moves. To generate a replacement when the original cases are
unavailable, run the existing simulator CLI on an allocated GPU with
`--cases` pointing to a fresh file and seed 0, then pass that complete
manifest to both Slurm jobs. Case generation uses the existing expert filter.
Policy rollouts use the selected scenes and run only predicted actions.

## Commands

Run from the MOTUS checkout. Each job requests one H100 and writes into its
own job-ID directory under `outputs/evaluation/A/` or `outputs/evaluation/C/`.
Each uses `scripts/cluster/environment.sh`.

```bash
cd /project/6101778/kding/Motus

# One-window offline smoke and one simulator case for each checkpoint.
sbatch experiments/ac_spline_pilot/scripts/eval_C.slurm offline 1
sbatch experiments/ac_spline_pilot/scripts/eval_A.slurm simulator 1 \
  outputs_backup/ac_pilot_7676974/A/robotwin_pilot_A/pilot-seed-0/checkpoint_step_1052
sbatch experiments/ac_spline_pilot/scripts/eval_C.slurm simulator 1

# Matched evaluation after locating the expanded-data A checkpoint and verifying simulator setup.
sbatch experiments/ac_spline_pilot/scripts/eval_A.slurm offline
sbatch experiments/ac_spline_pilot/scripts/eval_C.slurm offline
sbatch experiments/ac_spline_pilot/scripts/eval_A.slurm simulator 100
sbatch experiments/ac_spline_pilot/scripts/eval_C.slurm simulator 100
```

To use an expanded-data bundle in a different directory, pass its path
after the limit. For all offline windows with an explicit bundle, use
`sbatch experiments/ac_spline_pilot/scripts/eval_A.slurm offline '' /path/to/checkpoint`.

Generate the comparison with the job IDs from the four completed runs.

```bash
python -m experiments.ac_spline_pilot.evaluation_report \
  --offline "outputs/evaluation/A/${A_OFFLINE_JOB}-offline" "outputs/evaluation/C/${C_OFFLINE_JOB}-offline" \
  --simulator "outputs/evaluation/A/${A_SIM_JOB}-simulator" "outputs/evaluation/C/${C_SIM_JOB}-simulator" \
  --output outputs/evaluation/comparison.md
```

## Metric definitions

- Offline scoring checks the saved window-manifest hash and validation split,
  then applies the same deterministic noise seeds and recorded timestamp
  grid for both policies. The grid preserves the training loader’s terminal
  row clamping. Inputs come from the existing validation dataset. Actions
  are six absolute arm angles in radians and a normalized gripper opening
  for each arm, in left-then-right order.
- The offline scorer loads RoboTwin’s Aloha AgileX articulation from the
  configured URDF and uses the robot’s world-frame left and right TCP
  getters for both predictions and targets. It averages squared Euclidean
  position errors over timestamps and arms before taking the square root.
  Orientation uses the mean SO(3) geodesic angle. Arm angles and gripper
  openings have separate RMSEs. It validates FK against SAPIEN’s Pinocchio
  implementation at the configured home and a perturbed joint pose.
- `windows.jsonl` retains supported timestamp masks and failures.
  `demonstrations.json` and `results.json` give pooled errors and
  per-demonstration mean and population standard deviation. Complete-window
  metrics cover windows with all 16 timestamps. Supported-timestamp metrics
  also include partial windows. Both show their coverage denominators.
- Simulator jobs use `demo_clean`, the same camera settings and controller,
  the saved inference schedule, and the official command limit. A and C
  replan on chunk boundaries, so C’s adaptive prediction can change policy
  call frequency. `episodes.jsonl` records each seed, official result, any
  rollout error, and per-call times. Scene/setup exceptions are counted
  among scheduled failures and flagged as missing official results.
- Model inference times the synchronized `inference_step` including its
  denoising schedule. Decoder time surrounds spline decoding. Total
  policy-call time also covers preprocessing and transfers. Simulated
  duration is the controller’s actual physics-step count multiplied by
  SAPIEN’s timestep, excluding reset. Wall-clock episode time includes
  setup, rendering, and video logging. Evaluation wall time also includes
  checkpoint load and language encoding.
- Success uses a 95% Wilson interval. Short failed episodes can dominate
  all-episode time averages. The report also compares wall time on seeds
  where both policies succeed when such cases exist. Spline diagnostics
  include the fraction of finite-knot calls with a raw inversion or repair,
  mean repair displacement in source rows, and counts of invalid
  multiplicity, insufficient coverage, and decoder failures.

## Smoke-test results

| Check | Result |
|---|---|
| Evaluation and simulator unit tests, job 7766326 | 9 passed |
| Aloha AgileX FK parity | Maximum discrepancy 2.91 × 10⁻⁷ m and 3.93 × 10⁻⁷ rad |
| C checkpoint and one offline window, job 7766215 | Loaded successfully. 16/16 timestamps covered. TCP position RMSE 0.03988 m, mean orientation error 0.1106 rad, and arm-joint RMSE 0.1152 rad. |
| A and C on H100, jobs 7766178 and 7766179 | Both bundles loaded. Scene reset failed in cuRobo before the first policy call or official result. |
| A on L40S, job 7766365 | Official success after 81 controller commands and six policy calls on seed 100000. |
| C on L40S, job 7766377 | Official result was unsuccessful on seed 100000. C executed 79 commands before the fourth query failed the strict decoder with interior knot multiplicity 5. |
| C repeat on L40S, job 7766655 | Same official outcome, 79 commands, and the same invalid knot multiplicity on the fourth query. |

### Offline smoke measurements

| Metric | Variant A | Variant C (one held-out window) |
|---|---:|---:|
| EE position RMSE (m) | Unavailable | 0.03988 |
| Mean SO(3) orientation error (rad) | Unavailable | 0.1106 |
| Arm-joint RMSE (rad) | Unavailable | 0.1152 |
| Full-window coverage | Unavailable | 1/1 |

### Closed-loop smoke measurements on L40S, seed 100000

| Metric | Variant A (archived 45/5 model) | Variant C (expanded 150/20 model) |
|---|---:|---:|
| Success rate | 1/1 | 0/1 |
| Successful episodes | 1 | 0 |
| Failed episodes | 0 | 1 |
| Simulated episode duration (s) | 45.25 | 42.66 |
| Wall-clock episode duration (s) | 93.46 | 221.42 |
| Policy calls | 6 | 4 |
| Commands per policy call | 13.50 | 19.75 |
| Inference latency (ms) | 3694 | 9955 |
| Decoding latency (ms) | N/A | 3.17 |
| Total policy latency (ms) | 3784 | 10186 |
| Wall-clock evaluation time (s) | 405.79 | 1047.27 |

All four C calls had at least one raw knot inversion and applied running
maximum repair, with a mean displacement of 0.515 source rows across knots
and calls. The simulator recorded one decoder failure from invalid interior
multiplicity and zero temporal coverage failures. The single-window offline
score measures reproduction of recorded commands. The closed-loop run
exercises replanning and object-level task success. The differing training
splits, C’s early stop, and the single scene prevent claims about comparative
success or speed. The expanded A bundle is required for a matched benchmark.
The C repeat changed mean inference latency from 9.95 s to 4.20 s per call on
a different L40S node, with the same commands and simulated duration.

The H100 reset-only diagnostic (job 7766610) reproduced the illegal
instruction in cuRobo's fused L-BFGS kernel during planner warmup. The same
environment passed model inference and FK on H100. Use
`MOTUS_CONTAINER` with a validated H100 RoboTwin image for the simulator,
or select the same L40S partition and GRES for both jobs. The upstream
cuRobo H100 report describes a similar failure with older CUDA builds.
[cuRobo H100 issue](https://github.com/NVlabs/curobo/issues/481)

```bash
sbatch --partition=gpubase_l40s_b1 --gres=gpu:l40s:1 \
  experiments/ac_spline_pilot/scripts/eval_A.slurm simulator 100
sbatch --partition=gpubase_l40s_b1 --gres=gpu:l40s:1 \
  experiments/ac_spline_pilot/scripts/eval_C.slurm simulator 100
```
