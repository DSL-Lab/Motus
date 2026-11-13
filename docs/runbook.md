# Verified environment and continuation commands

## Fix the missing 500-demo cache first

The archive has now been staged. From Motus, submit the preparation job:

```bash
cd /project/6101778/kding/Motus
bash scripts/experiments/prepare_500.sh submit
squeue -u kding
```

This requests one L40S, 4 CPUs, 64 GB for at most 30 minutes and prints the job ID
and exact log path (`logs/ac-JOBID.log`). It performs archive hash verification,
extraction, reference format conversion, existing spline fitting, and T5 language
encoding inside Slurm. It does not submit training. Check Slurm completion and the
log before retrying; all resource use including failures counts toward the budget.

```bash
# After preparation actually completes:
bash scripts/experiments/prepare_500.sh check
```

The intended output is `outputs/place_can_basket-500/manifest.json` with language
embeddings, authoritative seed.txt and scene_info.json. Numerical arrays and
converted HDF5 remain under Motus staging and are referenced by the manifest.
The adapter is `WAM-RoboTwin/tools/convert_legacy_data.py`, source commit
`74f239e1a10f7b528d7488af956ee1e75f959415`; it is reused without reference edits.
Its next-state action convention and dropped final frame are recorded in the
manifest. No physical clock is inferred from its frequency metadata.

For a fresh workspace, `bash scripts/experiments/prepare_500.sh` first downloads
the pinned public archive on the login node and then submits preparation. The
`download` mode alone was executed successfully here (79 seconds, 9.86 GB).
Three CPU inventory tests passed. Allocated preparation has not yet executed;
schema/runtime checks and the resulting cache are not claimed as completed.

Successful preparation does **not** manufacture the missing replay/repair reports
or verified test plan. Complete those before running the training readiness check
below. The original download and the existing 50-demo cache are never overwritten.

## Current request: fine-tune spline only, then test both

```bash
cd /project/6101778/kding/Motus
# No allocation: verify inputs before submitting.
bash scripts/experiments/train_spline_then_test.sh place_can_basket --check
mkdir -p logs
sbatch scripts/slurm/finetune_spline_then_test.sh place_can_basket
squeue -u kding
```

One job requests one L40S, 4 CPUs, 64 GB, four hours. The payload fine-tunes C
only (rank 256, 1000 updates, effective batch 16), then evaluates original
pretrained dense MOTUS and the final step-1000 C checkpoint. It writes
`outputs/place_can_basket-C-r256-JOBID/test-JOBID/summary.json` and per-policy
`episodes.json`; logs are `logs/spline-test-JOBID.log`. The pretrained model is
never fine-tuned. Both policies retain common experiment input/controller
adaptations. This is a pretrained-versus-fine-tuned comparison.

**Not ready to execute successfully yet.** The data and validation prerequisites
below are absent. No jobs or simulator episodes were executed while preparing
this workflow. CPU outcome tests use fake environments; they do not establish
live headless inference, source timing, scene-reset compatibility, or rank-256
training feasibility. Four hours is an allocation limit, not a completion estimate.

Required inputs (environment variables may override the default paths):

- `MOTUS_EPISODE_MANIFEST`: `outputs/place_can_basket-500/manifest.json`, prepared
  500-demo cache with 450/50 episode split and scene-prefixed language embeddings.
- `MOTUS_DEMO_SEEDS`: `outputs/place_can_basket-500/seed.txt`, authoritative
  source seed mapping for all 500 demonstrations, preserved from the raw archive.
- `MOTUS_REPLAY_REPORT` and `MOTUS_REPAIR_REPORT`: completed existing validation
  reports. These requirements are unchanged, not files to fabricate.
- `MOTUS_TEST_PLAN`: `outputs/place_can_basket-500/test_scenes.json`, frozen before
  training. Fields: `dataset_identity` matching the manifest;
  `demonstration_seeds_sha256`; `task_config_sha256` of the installed
  `WAM-RoboTwin/env_cfg/task_config/demo_randomized.yml`; `task_name`;
  `task_config: demo_randomized`; verified `control_dt_s` and `clock_provenance`;
  and `cases` containing exactly 100 objects with integer `seed`, fixed
  `instruction`, and optional integer `policy_seed`. All scene seeds must be
  distinct and disjoint from all demonstration seeds. Do not guess timing or
  choose the test suite after observing policy results.

If the job reaches its wall limit, inspect the log/accounting before continuation.
Use the same run directory and an existing safe checkpoint, for example by setting
`MOTUS_RUN_DIR` and `MOTUS_RESUME` to their actual paths when submitting the same
Slurm script. `MOTUS_STAGE=train` trains only; `MOTUS_STAGE=test` tests an existing
final checkpoint without retraining. Frozen inputs must match the original run.
Each continuation consumes another allocation and requires an authorized budget.

Success rates include reset errors, invalid predictions, and timeouts in the fixed
scene denominator; errors are also reported separately. The evaluator exits 2 on
infrastructure errors. No final rate is assigned to an incomplete suite. Wilson
intervals condition on the one checkpoint; they do not estimate training-seed
uncertainty. Disjointness from the pretrained checkpoint's training scenes is
unknown; the seed check covers this new fine-tuning dataset.

## Single entry point: one task, 500 demos, rank 256

```bash
cd /project/6101778/kding/Motus
bash scripts/experiments/finetune_one_task.sh place_can_basket both --check
bash scripts/experiments/finetune_one_task.sh place_can_basket both
# Use C instead of both to submit only the spline condition.
```

The second command submits jobs only after the preflight succeeds. Defaults:
rank 256, alpha 32, seed 0, 1000 updates per condition, microbatch 1 with
accumulation 16. A and C share a 450/50 episode split and run sequentially.
Each job requests one L40S, 4 CPUs, 64 GB, up to 4 hours (up to 8 GPU-hours for
both). This is a pilot recipe, not a measured cost or completed experiment.
Executing submission requires resource authorization; none was renewed during
preparation of this script.

Prerequisites: a complete scene-prefixed manifest at
`outputs/place_can_basket-500/manifest.json` (or `MOTUS_EPISODE_MANIFEST`), plus
truthful completed reports in `MOTUS_REPLAY_REPORT` and `MOTUS_REPAIR_REPORT`.
The new script uses the existing shared trainer and gates, creates a unique run
directory, stores the manifest/reports/job IDs, and generates the resolved recipe
inside Slurm. `MOTUS_SEED` and `MOTUS_WALLTIME` override the seed and wall limit;
limits longer than four hours are rejected. No test set is used for fitting.

Verified locally: `bash -n`, `--help`, and `--check` missing-cache rejection.
No submission tested. The 500-demo cache does not yet exist, so the command
currently exits without submitting jobs. It does not automatically download or
convert a dataset or claim that incomplete replay gates have passed.

500-demo source: public Hugging Face metadata API
`https://huggingface.co/api/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset/place_can_basket`,
resolved revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`.
Use `aloha-agilex_randomized_500.zip`, not `demo_clean.zip` or a guessed
`demo_randomized.zip`. Its older raw HDF5 layout requires inspection before
reusing `WAM-RoboTwin/tools/convert_legacy_data.py` (reference commit
`74f239e1a10f7b528d7488af956ee1e75f959415`). That converter defaults to next-state
actions and drops the last frame. Its `frequency` metadata is not proof of Hz.
No reference repository edits are needed or made. See status.md for verified
sizes, expected hash, metadata contents, and remaining work.

## Repair CPU checks (executed 2026-11-12)

No GPU allocation or training was used. Existing installed SciPy module:
Python 3.11.5, NumPy 1.25.2, SciPy 1.11.2. The training container remains the
recorded Torch 2.4.1 environment; these CPU checks do not replace its tests.

```bash
cd /project/6101778/kding
module load scipy-stack/2023b
export PYTHONPATH=/project/6101778/kding/Motus:/project/6101778/kding/Motus/tests
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest test_replay_repair -v
python Motus/tests/allocated_decode.py --cpu --all-windows \
  --output /project/6101778/kding/Motus/outputs/decode-stages-cpu-all-20261112
```

Replay instrumentation: 3 tests passed. Decoder numerical stages: 8269 windows,
maximum normalized local/global error 1.0603004962916884e-7, zero repairs. Decoder
command exits 2 intentionally because actual Torch BF16/model/Euler stages remain
blocked; its JSON report distinguishes these from the historical learned failure.
Choose a new output path for any rerun; previous artifacts are immutable.

Allocated continuation remains pending diagnostic-budget authorization. The new
config expects a new `outputs/repair-cache-20261111/manifest.json`, which has not
been generated. Prepare it with `python -m motus_actions.prepare
--language-from-manifest ... --wan-path ... --output ...` through the existing
allocated conversion launcher; this verifies/reuses spline arrays and encodes the
scene-prefixed instructions. Do not repoint to the legacy raw-language cache.
Then run the existing smoke entry point's numerical/decode stages, followed by
the `overfit A|C --varied --output ... --until 2` diagnostic, exit, and a second
process with `--resume .../checkpoint_step_2 --until 4`; inspect its bit-exact
resume audit before continuing the fixed 256-update tiny schedule. These are
prepared commands, not completed checks or authorization to submit jobs.

Workspace shell path resolves to `/home/kding/projects/aip-rjliao/kding`; persistent
cluster path is `/project/6101778/kding` (same shared workspace). Use persistent
absolute paths in jobs. Not a presumed root Git checkout. Implementation repo is
`/project/6101778/kding/Motus`. No new pip dependencies installed.

## Environment

Verified in job7470734: Apptainer1.4.5,
`/project/6101778/kding/containers/robotwin-cu121.sif`, Python3.10.12,
torch2.4.1+cu121, scipy1.15.3, transformers4.57.1, accelerate1.12.0,
diffusers0.36.0. Python overlay:
`Motus/.runtime/apptainer/lib/python3.10/site-packages`.
Single L40S on account aip-rjliao,4CPUs,64GB for model smoke. Slurm chooses verified
available GPU partition; none guessed. Host `scontrol show config` reports Killarney.
Read-only scratch bind supplies metadata recovery provenance only.

Missing primary metadata restored into `.runtime/model-metadata` from
`/scratch/kding/Motus_old_outdated/pretrained_models/`:
Qwen processor/tokenizer/config files, WAN config and UMT5 tokenizer. T5 weight
symlink points to unchanged persistent pretrained_models weight file. Actual MOTUS
checkpoint is `pretrained_models/Motus_robotwin2/mp_rank_00_model_states.pt`;
VAE `pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`. No alternate model weights.
Initial checkpoint compatibility is recorded in allocated model smoke artifacts.

`environment.sh` explicitly passes NVIDIA_DRIVER_CAPABILITIES and visible devices
to the container (export alone leaves image defaults under --cleanenv). Vulkan/EGL
JSON host binds mirror working reference script. With this fix, job7471089 rendered
a finite64x64x4 headless image. See Apptainer's official GPU docs:
https://apptainer.org/docs/user/main/gpu.html . Failed graphics initialization is
preserved in7471040; this was not a spline replay failure.

Do not run GPU/training/rendering on login nodes. Login container mount failed for
FUSE; compute node container worked. All numerical tests also ran in allocations.

## Commands already run

From Motus (create logs directory before submitting):

```bash
cd /project/6101778/kding/Motus
mkdir -p logs
squeue -u kding
sbatch --parsable scripts/slurm/slurm_single_node.sh
sbatch --parsable --time=00:30:00 scripts/slurm/slurm_single_node.sh \
  scripts/experiments/convert_splines.sh \
  --output /project/6101778/kding/Motus/outputs/pilot-cache-guard-20261111
sbatch --parsable scripts/slurm/slurm_single_node.sh scripts/experiments/smoke_test.sh numerical
sbatch --parsable --time=00:15:00 scripts/slurm/slurm_single_node.sh scripts/experiments/smoke_test.sh model
sbatch --parsable --time=00:05:00 scripts/slurm/slurm_single_node.sh scripts/experiments/replay.sh
```

Conversion rejects an existing output directory; choose a new explicit path when
source/settings change. Cache has50 source hashes,45/5 split,8269 eligible windows;
1864 video-padding candidates and62 short-coverage candidates excluded for both A/C.
`example.npz` / `example.png` link actual image indices and parameter/action curves.
Shared RGB helper dependency:
`WAM-RoboTwin/XPolicyLab/utils/process_data.py:decode_image_bit`, submodule
`5d0be1499634d3176258c8f2e31f5fe816497b0a` (submodule itself clean; parent records a
different gitlink). Do not duplicate its two-format JPEG logic.

## Job tracking and training gate

Every new allocated entry writes `outputs/jobs/JOBID/{source_commit.txt,source.patch,
source.tar.gz,allocation.txt,command.txt}`; logs `logs/ac-JOBID.log`. Early diagnostics
preceded this hook; status distinguishes those. Slurm exit2 from replay means
blocked preflight, not task failure or success.

```bash
squeue -u kding
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocTRES
cat logs/ac-JOBID.log
```

Shared training launcher is `scripts/train.sh A|C SEED OUTPUT [RESUME]`, invoked
through the same Slurm script. It refuses duplicate output reuse without resume
and requires a truthful replay report (`MOTUS_REPLAY_REPORT`) with passed original,
spline and matched-controller gates. None is available now; do not fabricate one.
The main routine writes adapter targets/counts and complete key-load report. Native
Accelerator checkpoint saves model/adapters/projections, optimizer, scheduler and RNG;
resolved_config.yaml and trainer_state.json are added. Shared config defaults to
pilot1000 updates, accumulation16. Main matrix not submitted.

## Unavailable execution

The current WAM-RoboTwin harness is a remote XPolicyLab client/server harness;
upstream Motus eval.sh calls nonexistent local script/eval_policy.py in this
checkout. Adapting that harness and collecting requested episode latency records
remains unvalidated. Do not claim legacy deployment wrapper changes constitute
an executed learned-policy evaluation. No aggregate policy-success table exists.
The existing deploy wrapper now routes to primary model and strict native LoRA
checkpoint loading, and C decodes at matched source offsets. Controller clock,
scene replay and live deployment remain gates, documented in experiment_spec.md.

Original weight download metadata (not a new content verification): MOTUS HF
revision `ab74bc214f7fc5952c85198e062c5f8bf1dc47a7`, recorded LFS SHA256
`70bc1741b45db5e3eae86510ad3bb1bd1aef04f10b17d8c1ff4dbe099300eb65`.
VAE revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`, recorded LFS SHA256
`20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`.
Allocated Slurm entry now hashes container/MOTUS/VAE/T5 once into
`outputs/asset_sha256.txt`; do not confuse downloaded metadata with a completed
local hash until that file exists. Reference source files were not edited;
imports may create ignored bytecode caches; future containers disable bytecode writes.

## Final verified diagnostics and limitations

Latest canonical cache is `outputs/pilot-cache-guard-20261111`; all prior caches
are preserved as explicitly versioned exploratory artifacts. Actual SHA256 hashing
completed and is recorded in `outputs/asset_sha256.txt`, not merely download metadata.

```bash
cd /project/6101778/kding/Motus
# These commands were executed with unique job IDs/output directories:
sbatch --parsable --time=00:30:00 scripts/slurm/slurm_single_node.sh \
  scripts/experiments/smoke_test.sh overfit A
sbatch --parsable --time=00:30:00 scripts/slurm/slurm_single_node.sh \
  scripts/experiments/smoke_test.sh overfit C
sbatch --parsable --time=00:20:00 scripts/slurm/slurm_single_node.sh \
  scripts/experiments/smoke_test.sh inference
sbatch --parsable --time=00:05:00 scripts/slurm/slurm_single_node.sh \
  scripts/experiments/smoke_test.sh numerical
bash scripts/experiments/aggregate.sh outputs/reports/NEW_UNIQUE_REPORT
```

Do not blindly repeat completed diagnostics: checkpoints are in
`outputs/tiny-A-7471646/checkpoint_step_50` and
`outputs/tiny-C-7471647/checkpoint_step_50`; full-state/native API reload passed.
The inference diagnostic currently refers to these exact checkpoints on purpose.
Latest wrapper records: `outputs/wrapper-inference-7471966/results.json`, raw C
parameters `C-prediction.npz`. A decoded16x14, C rejected multiplicity after32 counted
knot clamps. Those are offline current-observation tests, not successful robot
rollouts or warmed benchmarks. No partial success should be inferred from Slurm0.

Final focused test log: `logs/ac-7471967.log`,16 tests passed. Final diagnostic
report: `outputs/reports/20261111-final/report.md`, with raw ledger/summary.json.
All25 allocations ended; consumed1.121944 GPU-hours including failed/retried setup.
No pilot, main matrix or simulator episode evaluation was submitted.

Native actual checkpoint reload was exercised within the training process, and
fresh-process deployment loaded full saved models strictly. A fresh Slurm training
resume through the full CLI has not been exercised. Final evaluator/controller
integration and full replay remain unvalidated. Do not manufacture a passed replay
report to bypass the shared training launcher's explicit gates.

Source was frozen for7471966/7471967.7471801's Python diagnostics completed but its
shell failed after a shared-script edit during execution; retained as a failed
attempt. Superseding frozen-source run7471966 exited0 and independently recorded
C invalidity. Keep launcher/source stable throughout each running job, or execute
an immutable job source snapshot when extending this workflow.
