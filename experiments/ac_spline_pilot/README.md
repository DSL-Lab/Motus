# Method A vs. Method C: Adaptive Spline pilot

A uses dense action targets; C predicts packed cubic spline parameters. Both use
50 clean `place_can_basket` demonstrations with the same seed-0 45/5 split,
observations, language, optimizer, and rank-64 / alpha-128 LoRA. Adapter targets
are `video_model.wan_model.*` and `action_expert.*`; VLM and Understanding remain
frozen. The complete action input/output projections retain their full tuning.

## Prepare, train and evaluate

Run from the Motus repository root. The cluster wrapper uses the existing
Apptainer image and Python overlay; see [environment.sh](../../scripts/cluster/environment.sh).

```bash
# Check all 50 source recordings and local model/tokenizer assets.
bash scripts/cluster/environment.sh --cpu python ../WAM-RoboTwin/scripts/prepare_motus_clean50.py --check

# Inspect the exact commands without submitting jobs.
MOTUS_PREP_FRESH=0 bash experiments/ac_spline_pilot/scripts/submit_prepare_clean50.sh --dry-run
bash experiments/ac_spline_pilot/scripts/submit_full_pilot.sh --dry-run

# One submission: reuse/prepare data, train A/C, then run both in RoboTwin.
bash experiments/ac_spline_pilot/scripts/submit_full_pilot.sh --submit --task place_can_basket
```

[Preparation](scripts/submit_prepare_clean50.sh) requests one L40S GPU, eight
CPUs, 64 GB RAM, and three hours. It writes PT actions, composed three-camera
videos, T5 embeddings from each episode's first seen instruction, canonical
RoboTwin spline sidecars, training-only statistics, and aligned window manifests.
The log is `outputs/ac_prepare_<job-id>.out`.

The one-command submission forces preparation reuse (`MOTUS_PREP_FRESH=0`) and
submits training/evaluation with `afterok` on preparation. Verified existing
outputs are reused; missing preparation outputs are built. Source/contract
mismatches fail instead of rebuilding datasets. The separate preparation script
still supports its explicit guarded fresh mode. Raw recordings and checkpoints
are preserved. This experiment currently supports `place_can_basket` only.

[Training](scripts/submit_full_pilot.sh) requests one L40S GPU on
`gpubase_l40s_b3`, eight CPUs, 64 GB RAM, and one day under account `aip-rjliao`.
It runs A then C, stopping if either fails. Each variant uses 1,000 optimizer
updates, microbatch 1, accumulation 16, 200 warmup updates, a 1,000-update schedule,
and checkpoints every 250 updates. `--report_to none` disables logging trackers;
TensorBoard is optional. Logs are `outputs/ac_pilot_<job-id>.out`, with results
under `outputs/ac_pilot_<job-id>/{A,C}`. After both finish, the same job evaluates
both exact `checkpoint_step_1000` bundles through RoboTwin simulator rollouts;
it never searches an older run or chooses a checkpoint by glob.

If preparation is already complete, check the paths and submit training plus
simulator evaluation directly:

```bash
bash experiments/ac_spline_pilot/scripts/submit_full_pilot.sh --check
sbatch experiments/ac_spline_pilot/scripts/submit_full_pilot.sh
```

The local `--check` verifies path existence. Full source, split, cache, and
alignment validation happens before model creation on compute.

## Inputs and alignment

The default artifact root is `outputs/ac_spline_pilot/clean50/`. Export overrides
before submission when using a different prepared location:

| Variable | Default under the artifact root |
| --- | --- |
| `MOTUS_DATASET_ROOT` | `dataset/` |
| `MOTUS_WINDOWS` | `windows.json` |
| `MOTUS_NORMALIZATION` | `normalization.json` |
| `MOTUS_KNOT_SCALING` | `knot_scaling.json` |

The dataset must contain exactly 50 complete episodes under
`clean/place_can_basket/{qpos,videos,umt5_wan,metas}`. The converter sorts resolved
paths, applies `np.random.default_rng(0).permutation(50)`, reserves five episodes
for validation, and uses the remaining 45 for training. Both normalization and
knot scaling use those same 45 sources. Manifests must cover every episode in its
correct split. Randomized data, incomplete episodes, and stale statistics fail
validation.

HDF5 image/action rows define the alignment axis. Timelines declare
`source_action_step`; physical seconds remain unknown. Video playback FPS and
legacy recording frequency fields do not establish a controller clock.

Model paths can be overridden with `MOTUS_BASE_CHECKPOINT`, `MOTUS_WAN`,
`MOTUS_VAE`, and `MOTUS_VLM`. To fetch missing tokenizer/configuration metadata
without downloading model weights:

```bash
bash scripts/cluster/environment.sh --cpu env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python ../WAM-RoboTwin/scripts/prepare_motus_clean50.py --download-metadata
```

## Entry points and ownership

- [WAM-RoboTwin preparation](../../../WAM-RoboTwin/scripts/prepare_motus_clean50.py)
  owns the conversion bridge and uses RoboTwin's `robotwin_spline` package for
  fitting. Motus supplies its model-specific packing, video selection, and
  training contract.
- [trainer.py](trainer.py) supplies the A/C factory and trainer subclass.
  [research.py](research.py) owns adapter injection and portable checkpoints;
  [data.py](data.py) enforces the clean-50 split.
- Run the experiment directly with `python -m experiments.ac_spline_pilot.trainer`.
  It reuses [train/train.py](../../train/train.py) for the training loop and
  Accelerator setup. The general `python train/train.py` path remains available.
  Dataset options stay in their existing YAML fields; only the fixed-window
  manifest and contract are passed through `dataset.params`.
- Preparation and loading share `prepare_window` in
  [research_windows.py](../../data/robotwin2/research_windows.py). Provenance,
  reconstruction and clock checks run once per episode in each manifest load;
  coverage, round trips and video validity remain checked per observation.
- [run_tiny_research.sh](scripts/run_tiny_research.sh) retains the 100-update
  training and fresh-process reload check for an existing GPU allocation.
  Generation previews use at most four windows; training still covers the
  complete 45-episode window set.

## Offline diagnostics

Training generates the original four-window, seeds-0–3 offline preview automatically.
`generation/results.json` explicitly labels this as offline fixed-window
generation. These predictions do not measure simulator task success.
For fresh-process evaluation, run the following inside an existing Slurm GPU
allocation. Use a new `eval_root`; the output guard rejects existing run paths.
The saved, resolved checkpoint config supplies the original dataset/model paths,
training contract, windows, seeds and denoising settings, so no asset-variable
setup is repeated. The fresh reload check compares raw generated parameters
against the checkpoint's adjacent `generation/predictions.npz`.

```bash
run_root=outputs/ac_pilot_7643090
# Choose a new destination for each evaluation.
eval_root=outputs/ac_eval_NEW
for variant in A C; do
  checkpoint="$run_root/$variant/robotwin_pilot_$variant/pilot-seed-0/checkpoint_step_1000"
  srun --ntasks=1 --gpus=1 --cpus-per-task=8 \
    bash scripts/cluster/environment.sh --gpu \
    python -u -m experiments.ac_spline_pilot.trainer \
    --config "$checkpoint/config.json" \
    --diagnostic-eval-only "$checkpoint" \
    --checkpoint_dir "$eval_root/$variant" --report_to none
done
```

Method C uses the shared repaired absolute-knot decoder from
[the safeguard fix](../../SPLINE_DECODING_FIX.md). Raw and repaired validity,
repair magnitude, prefix coverage and valid decoded error are reported
separately. To re-score the original saved C predictions without a GPU or model
regeneration:

```bash
bash scripts/cluster/environment.sh --cpu python -m data.robotwin2.redecode_predictions \
  --checkpoint "$run_root/C/robotwin_pilot_C/pilot-seed-0/checkpoint_step_1000" \
  --output outputs/ac_C_redecode_NEW
```

The commands above are offline diagnostics only. The new simulator bridge below
reconstructs research bundles; the legacy dense RoboTwin wrapper is unchanged.

## Actual closed-loop RoboTwin evaluation

The launcher now calls `python -m experiments.ac_spline_pilot.simulator` for A and
C after training. [rollout_policy.py](rollout_policy.py) reconstructs the model
using the saved config, loads the original frozen base and exact research
adapter bundle with the existing strict loader, and uses training's RGB camera
composition, image padding, raw state layout, T5 prompt and VLM preprocessing.
T5 instructions are encoded before loading MOTUS so both large models need not
remain on the GPU together. C uses the same running-maximum/endpoint/domain
safeguards as offline decoding. Invalid prefixes fail the rollout without
replacement commands, extrapolation, or retiming.

[simulator.py](simulator.py) reuses RoboTwin's existing
`eval_policy_xpolicylab.run_one_batch_episode`, observation conversion, action
conversion, `take_action`/TOPP controller, scene success checks, command limit,
and optional video logging. It runs in-process rather than starting a redundant
policy server on the same GPU. No simulator controller or task reward was changed.

Defaults are **100 episodes per variant**, evaluation seed **0**, clean task
config, and its `seen` instruction setting. Override only the evaluation count
and seed with `MOTUS_EVAL_EPISODES` and `MOTUS_EVAL_SEED` before submission.
The evaluator expert-filters a single scene list beginning at RoboTwin's normal
`100000 * (1 + seed)` offset, saves exact scene seeds and instructions once, then
reuses it for both variants. Policy noise is seeded by scene/call while preserving
simulator RNG state. No policy, setup or decoder failure replaces an assigned
scene. These failures stay in the denominator; model/startup failures mark the
run incomplete and fail the job. A failure in A evaluation still allows C's
matched evaluation to be attempted.

Each `outputs/ac_pilot_<job-id>/` contains:

- `A/.../checkpoint_step_1000/`, `C/.../checkpoint_step_1000/`: this run's bundles.
- `{A,C}/.../generation/`: **offline** previews and decoded prediction errors.
- `simulator/cases.json`: shared expert-valid scenes/instructions and rejected
  candidate seeds; both result files record its hash.
- `simulator/{A,C}/results.json`: checkpoint/contract provenance, completion,
  success/failure counts, success rate and load/total wall times.
- `simulator/{A,C}/episodes.jsonl`: individual rollout outcomes, failure reasons,
  executed commands, policy calls, per-call decoder reports, synchronized model
  inference, policy/controller wall times and scene timestep.
- `simulator/{A,C}/videos/`: videos when enabled by the existing task config.
- `logs/train_{A,C}.log`, `logs/simulator_{A,C}.log`: persistent separate logs,
  alongside Slurm's preparation and pipeline logs.

Timing includes first-call overhead; this is not a warmed latency benchmark.
A decoded source-row stride is not a physical clock, and a command is not a
physics step. Scene timestep is reported; no unmeasured simulated duration or
TOPP failure count is invented. Training wall time remains in the trainer logs.

## Verification

CPU regression checks, including a synthetic 50-episode preparation test:

```bash
bash scripts/cluster/environment.sh --cpu python -m unittest discover -s experiments/ac_spline_pilot/tests -t . -v
bash scripts/cluster/environment.sh --cpu python -m unittest discover -s tests -v
```

The Python overlay must include the repository requirements, in particular
`qwen-vl-utils` and `decord` for real data loading. Research design and historical
validation reports remain in [docs/](docs/); operational commands above supersede
earlier prototype launchers and recorded restrictions in those reports.

The 2026-11-16 cleanup was checked against all 1,122 training and 126 validation
windows in the existing clean-50 corpus. Window identities, dense/spline targets,
conditioning, video indices and masks are unchanged; preparation selects the
same windows in all 50 episodes. Effective tiny/pilot A/C configurations and
contract identities match, except for redundant copies of dataset options in
`dataset.params`. The original Method C decoder replay report is identical.
Local verification artifacts are `outputs/ac-cleanup-20261116/equivalence.json`
and `outputs/ac-cleanup-20261116/redecode/results.json`. No preparation rebuild,
training or GPU generation was run during that cleanup. See the current rollout
integration validation below for the subsequent simulator connection.


## Rollout integration validation — 2026-11-16

Before this change, `submit_full_pilot.sh` launched only A/C training, and
`ACSplineFactory.run` ended with `generate_diagnostic`; it never invoked a
simulator. The existing dense deployment wrapper could not load research
bundles. The new connection above adds actual closed-loop execution.

Passed: 44 full experiment regressions, 10 shared regressions, 24 decoder
regressions, plus the subsequently added setup-failure test and the targeted
adapter/launcher rerun. This covers training-compatible RGB/state/prompt inputs,
exact requested-bundle loading, shared expert-selected scenes/instructions,
fixed seeds after setup/decoder errors, existing rollout-loop success and
failure handling, command counts/timing, missing-producer-checkpoint rejection,
`afterok` submission with fake Slurm commands, and evaluating C after A errors.
The real launcher `--dry-run` and saved-checkpoint `--check` also passed.

A bounded live check used one L40S allocation, Slurm job **7675647**, and the
existing `ac_pilot_7643090` step-1000 bundles. This was a test of the bridge;
the production launcher derives future checkpoints from its own run directory.
Both used the same expert-valid scene **100000** and exact saved instruction:

| Policy | Simulator result | Commands | Policy calls | Episode wall time |
| --- | --- | --- | --- | --- |
| A | success | 82 | 6 | 73.83 s |
| C | failure: invalid knot multiplicity or degenerate domain | 64 | 5 | 85.18 s |

C executed four valid chunks, then rejected its fifth prediction. No replacement
commands were issued. This verifies actual scene creation, both full checkpoint
loads, observation-conditioned policy inference, decoder integration, RoboTwin
controller execution, task success/failure recording and videos. One scene is
an integration smoke test, not a task-success comparison.

Artifacts: `outputs/ac-rollout-integration/smoke_7675647/{A,C}/results.json`,
`episodes.jsonl`, videos and logs; shared `cases.json`; compact
`outputs/ac-rollout-integration/validation.json`. Cold model/language load and
scene-selection time are recorded separately from per-episode policy/controller
timing. The full new prepare/train/100-episode-per-policy pipeline has **not**
been launched or validated end to end, as requested. The full submission chain
and its current-run checkpoint handoff were tested with fake Slurm/training
processes; the live smoke used existing checkpoints and did not retrain.
