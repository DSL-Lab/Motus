# Current status — 2026-11-11 UTC

## Missing-cache follow-up (2026-11-12)

The user's readiness command stopped at the absent 500-demo manifest; its `&&`
prevented Slurm submission. No training job was started. The missing preparation
path is now implemented in `scripts/experiments/prepare_500.sh` and
`motus_actions/prepare_500.py`.

**Completed:** downloaded the pinned 9,855,736,739-byte archive to
`.runtime/datasets/place_can_basket-500-3dc3b798668feb99ac61cc9086d84cbcc3d79186/`.
Local central-directory and small-sidecar inspection confirms 500 HDF5 episodes,
500 distinct recorded seeds, 500 instruction files with `seen`/`unseen` fields,
and `scene_info.json`. Three focused inventory tests pass; shell/Python syntax
checks pass. Full archive SHA-256 verification is deliberately deferred to Slurm.
Host Python and the recorded scipy-stack lack h5py, so a real HDF5 schema check
was not performed on the login node. No dependencies were installed.

**Prepared, not submitted:** one 30-minute L40S preparation job through the
existing Slurm/container wrapper. It verifies the published archive SHA-256,
extracts into new Motus staging, reuses the existing read-only reference converter,
fits all 500 episodes with the existing spline converter (450/50 split), encodes
scene-prefixed T5 text, and publishes the final manifest only after completion.
Reference next-state-as-action alignment is recorded explicitly, not claimed as
verified original timing. Raw archive/checkpoints/existing outputs stay preserved.

Run `bash scripts/experiments/prepare_500.sh submit` from Motus to submit this
preparation job using the staged archive. The job captures source and allocation
through the existing wrapper. No GPU allocation has been submitted by the agent;
the resource-budget question is still unanswered. Preparation is not replay or
repair acceptance: those gates and a verified controller/test plan remain pending.
Do not promise that the training launcher will pass immediately after preparation.

## Latest scope: train C only, test pretrained original and C (2026-11-12)

The user clarified that only the spline policy is to be fine-tuned. Added
`scripts/slurm/finetune_spline_then_test.sh` and its experiment payload: one L40S,
rank 256, 1000 C updates, then paired testing of the original pretrained dense
checkpoint and the final C checkpoint. No A training job is submitted. This
supersedes the matched-training interpretation below for the current request;
fine-tuning, representation, and video alignment all differ in this comparison.

`inference/robotwin/evaluate_pair.py` now composes the existing policy wrapper,
RoboTwin reset/config loader, scene snapshots, and fixed-rate deployed controller.
It requires 100 frozen scenes disjoint from all 500 demonstration seeds, fixed
instructions, task-config and seed-file hashes, and verified controller timing.
Both policies use the shared experiment input path. Original weights are loaded
strictly with zero-output adapters; there is no original-policy fine-tuning.
This preserves pretrained weights, with common input/controller adaptations
explicitly retained rather than claiming the entire upstream deployment is exact.
Per-scene failures/timeouts/repairs and task success rates are recorded. A partial
suite has no final success rate. Simulator and model execution remain unvalidated.

CPU outcome tests and shell syntax checks have been run; no new allocation,
conversion, training, or simulator test was submitted. The 500-demo cache, frozen
test plan, authoritative seed file, and completed validation reports are still
missing. The Slurm workflow intentionally stops at those missing prerequisites.
Its four-hour allocation may require continuation; periodic checkpoints remain
native resumable checkpoints. See runbook.md for submission and continuation.

## One-task rank-256 request — 2026-11-12

Current user scope: one task with 500 distinct demonstrations and LoRA rank 256.
Retain `place_can_basket`; one seed-0 A/C comparison replaces the proposed
four-task/three-seed matrix for this request. The prepared pilot remains 1000
optimizer updates per condition, effective batch 16. Use a fixed 450/50 episode
split across the 500-demo collection; these are not 500 training episodes plus
an additional validation collection. No new training or data preparation has run.

Added `scripts/experiments/finetune_one_task.sh`: one user-facing Slurm submission
entry point, rank 256 fixed, task selectable, A/C/both, same existing trainer.
It validates the 500-episode manifest/split and existing repair/replay reports
before submitting anything. A/C jobs are sequential, one L40S per job, default
4-hour limit each (8 GPU-hours maximum for both). These are script resource
requests, **not renewed authorization or measured runtime**. The existing shared
training launcher now accepts `MOTUS_TRAIN_CONFIG` for the resolved one-task recipe.
Help, shell syntax, and missing-data preflight were checked; actual submission,
rank-256 memory/throughput, training and evaluation remain unvalidated.

Public source metadata verified with HTTP byte ranges: dataset
`TianxingChen/RoboTwin2.0` revision
`3dc3b798668feb99ac61cc9086d84cbcc3d79186`, archive
`dataset/place_can_basket/aloha-agilex_randomized_500.zip` contains 500 HDF5 files,
`seed.txt`, `scene_info.json`, and per-episode `_traj_data` files. This differs from
the locally downloaded 50-demo `demo_clean.zip`. Compressed size 9,855,736,739
bytes; total expanded member size 12,880,601,228 bytes. Expected LFS SHA-256:
`e448a8e4f5049b8ab75c500dee55ef958c3f5326f8fc04630f2eca52ac1c871c`.
The archive has not been downloaded or locally hashed. Its metadata may help
resolve scene replay; full scene reset compatibility and source timing are still
unverified. Do not transfer the old clean-cache validation results to this dataset.

Storage quota checked using `diskusage_report`: project 6112 GiB / 20 TiB,
16M / 100M files; sufficient reported capacity for the archive and extraction.
No raw data, pretrained weights, reference source, or historical output changed.
Slurm confirmed Killarney and an empty user queue during this review. No jobs
submitted. The diagnostic-budget question remains unanswered.

Next actions: stage the pinned 500-demo source; inspect its HDF5 schema and
scene/timing sidecars; reuse `WAM-RoboTwin/tools/convert_legacy_data.py` if its
documented state/action convention is verified for this archive, writing only new
Motus outputs. Prepare a new 500-episode spline/language cache, complete the
existing repair/replay gates on allocated hardware, then run the single entry
point. The command currently exits before submission because that cache is absent.

**Repair in progress — checked 2026-11-12. Not ready for a pilot. Historical diagnostics below predate the conditioning/accumulation audit and are not passes for the repaired implementation. No jobs are running or pending.**

## Repair follow-up

Pre-repair source/status/remotes/diffs and untracked files are preserved in
`docs/snapshots/repair_20261111T215653Z`. Exact audited configurations and manifest
hashes are in `audited_runs.json` there. Both tiny checkpoints actually used
`outputs/pilot-cache-final-20261111/manifest.json`. Their raw-instruction training
and scene-prefixed wrapper deployment are incompatible with the corrected recipe.
The old checkpoints and outputs have not been overwritten.

Implemented, still requiring runtime validation: shared checkpoint-style scene
prompt; common camera mosaic; explicit legacy-checkpoint rejection; accumulation
across DataLoader epochs with update example counts; safe-boundary checkpoint
metadata and fresh-process audit; varied-noise tiny-subset generation diagnostics;
staged deployed decoding; original-first replay using the existing RoboTwin
fixed-rate controller and physical gripper measurements. No token/rank budget change.

Measured CPU evidence: `outputs/decode-stages-cpu-all-20261112/report.json` verifies
all 8269 cached local windows against independent global SciPy curves at the
controller offsets, including normalization inversion. Maximum normalized error
`1.0603004962916884e-7`, zero repairs. These checks execute the deployed decoder
method source on CPU; they do not load the full Torch deployment module. Model
target BF16 formatting and the actual Euler oracle remain blocked. The preserved
learned C prediction fails ordering, then interior multiplicity after 32 repairs;
its candidate postprocessed domain starts after the requested execution prefix.
This is an incompatible historical checkpoint, not a corrected learning result.

Three CPU replay tests passed: measured gripper qpos differs from commanded
opening; complete scene identity includes velocities/actors; replay acceptance
requires five pairs and a predeclared tracking threshold. No simulator replay
has executed. The fixed-rate protocol still requires authoritative source timing,
action timing semantics, recorded initial physics states/object identities, and a
predeclared tracking tolerance. Seed alone is insufficient.

Gate status: conditioning **blocked** on new T5 cache/runtime entry-point tests;
accumulation **blocked** on the real prepared-DataLoader boundary test; full
ground-truth decoder **blocked** on BF16/model/Euler stages (CPU numerical stages
passed); corrected tiny generation **blocked** on allocated runs; original/spline
replay **blocked** on recorded scene/timing metadata; fresh-process resume
**blocked** on allocated restart; token-budget audit **passed** for counts only,
physical horizon/latency **blocked**; pilot **blocked** on these gates and budget.

No new Slurm jobs were submitted. The completed overnight resource authorization
has not been assumed to renew. The pending question requests at most 2 GPU-hours
for repair diagnostics, one L40S at a time, at most 30 minutes per job. No reply
has authorized that budget. Continue CPU work where useful; do not submit the
pilot or production training on the basis of old smoke results.

## Measured results

- All16 focused tests passed in7471967, including independent analytic spline evaluation, local/global extraction, time/image alignment, real shared A/C conditioning, fused-QKV LoRA, merge agreement, gradient accumulation and optimizer/scheduler/RNG checkpoint resume.
- All50 clean Place Can Basket episodes converted;45 train/5 validation,8269 eligible windows. Maximum normalized fit error0.00990022 <=0.01.1864 video-padding and62 short-coverage candidates excluded for both conditions.
- Actual pretrained A and C forward/backward passed on allocated L40S hardware. All required gradients present/finite. Peak allocated memory about23GB; first-step times9.53s/9.15s are not steady-state throughput or policy latency.
-50 fixed-example/fixed-noise updates through the existing trainer: A loss0.167053->0.076579; C1.041750->0.209088. Both passed the predeclared final<=80% initial criterion. Actual full checkpoints restored trainable tensors and Torch RNG exactly; scheduler matched and optimizer restored. Focused tests additionally compare optimizer tensors and Python/NumPy/Torch RNG exactly.
- Headless synthetic rendering passed in7471089. Recorded demonstration replay was not performed; preflight stopped at missing seeds.
- Existing deploy wrapper exercised saved tiny checkpoints using current validation image/state/instruction only. Latest7471966: A decoded16x14 values; C rejected excessive knot multiplicity after32 counted running-maximum knot repairs. Zero fallback, no simulator commands executed. A decoded values have not been validated in a live scene.

## Scientific and execution blockers

1. All4 requested tasks have50 clean demonstrations, but their source ZIP archives contain no recorded episode-to-scene seed mapping or timestamps. Do not infer seeds or Hz from episode IDs/additional_info/frequency. move_can_pot has seeds but is not a permitted substitute.
2. Native take_action(qpos) TOPP-retimes each command. Fixed-frequency matched controller execution has not been established; changing it without verified timestamps would violate the intended comparison.
3. C has not demonstrated valid learned controller predictions on held-out observations.50-update one-example overfit is not policy learning evidence. Normalized parameter diagnostics show prediction errors; no rank/encoding sweep or silent fallback was added.
4. Installed WAM-RoboTwin uses a remote XPolicyLab harness; upstream Motus eval.sh points to missing script/eval_policy.py. A verified bridge and complete per-episode metrics remain outstanding, tied to the controller contract. Its UniWAM adapter is a different history-flow method and was not substituted for vanilla MOTUS.
5. Official SplineWAM code is not released on the checked project page. Numeric alignment offset/rounding are explicit paper adaptations. Published36 C rows exceed native A16; no token compression or latency benefit is established.

## Prepared implementation and preservation of A

Primary data/model/trainer/deployment and existing Bash/Slurm entry points extended; exact changes in code_map.md. Default dense model/data path remains; explicit A/C LoRA config uses same checkpoint/split/adaptation recipe. A retains dense actions and original uniform future-video selection on a shared eligible observation set. Observed state comes from current raw state, not future actions.
New cohesive motus_actions package owns fit/extraction/alignment/decode/conversion; utils/lora.py supports actual attention modules and custom fused QKV. No parallel trainer/evaluator, dependencies, quantization, speed scaling, asynchronous execution or adaptive replanning.
Native gripper clipping is explicit/counted for both A/C. Predicted running-maximum knot repair follows Appendix D and preserves control-point order; nonfinite, multiplicity, degenerate-domain, coverage and arm-limit failures reject commands. The earlier strict failure remains recorded, not relabeled successful.
VLM prompt ordering/generation flag mismatch in upstream deployment corrected only for explicit A/C; the shared helper now yields identical training/inference inputs.

## Artifacts

- Canonical cache: outputs/pilot-cache-guard-20261111/manifest.json; example.npz/example.png link actions, knots, times and images. Prior caches preserved; instruction-hashed T5 embeddings reused with the same immutable weights.
- Tiny checkpoints: outputs/tiny-A-7471646/checkpoint_step_50 and outputs/tiny-C-7471647/checkpoint_step_50 (native full model/adapters/projections/optimizer/scheduler/RNG; periodic step25 also retained). Each condition uses about31GB persistent storage.
- Actual wrapper records and raw C predictions: outputs/wrapper-inference-7471966; prior diagnostic/parameter error details in outputs/wrapper-inference-7471801.
- Machine-readable report, raw Slurm ledger and summary tables: outputs/reports/20261111-final/. No simulator episode evaluation records or task-success table exist.
- Large asset hashes: outputs/asset_sha256.txt. Container5b7a19c07eb53bfb34690df85bcff686f764cb91ea1a3ddfec88bd8d4ed3739c. MOTUS/VAE match HF LFS digests. Raw data/pretrained weights/reference source unchanged; initial dirty WAM submodule gitlink preserved.
- Each later allocated job captures source commit, dirty patch, complete source archive, allocation and command under outputs/jobs/ID; logs in logs/ac-ID.log. Early environment/initial conversion predate the snapshot hook and are explicitly exploratory. Metadata recovered from scratch is documented in runbook; no shared environment was altered.

## Resource consumption and job ledger

Consumed **1.1219 GPU-hours** across all25 allocations including failed attempts/retries. Authorization16GPU-hours,1GPU/job,2 concurrent,<=4h/job. All used1L40S,4CPU,64GB; actual job wall limits were5–30min. No pilot/main/evaluation jobs submitted; no running or pending jobs at session end.

| Job | State / exit | Elapsed seconds | Outcome |
|---|---|---:|---|
| 7470734 | COMPLETED / 0:0 | 29 | Environment/schema probe; completed. |
| 7470828 | COMPLETED / 0:0 | 18 | Initial conversion; superseded JPEG decoder. |
| 7470940 | COMPLETED / 0:0 | 44 | 9 focused tests passed. |
| 7470955 | COMPLETED / 0:0 | 20 | RGB-correct cache conversion. |
| 7470978 | FAILED / 1:0 | 17 | Failed unconditional TensorBoard import; fixed lazy logging import. |
| 7471039 | FAILED / 1:0 | 42 | Failed train module import resolution; fixed test entry. |
| 7471040 | FAILED / 1:0 | 10 | Failed Vulkan initialization; fixed explicit graphics environment. |
| 7471063 | FAILED / 1:0 | 360 | A exact initial checkpoint match; failed missing Qwen helper; vendored helper reused. |
| 7471064 | FAILED / 1:0 | 8 | Renderer diagnostic; unsupported introspection API; captured environment informed fix. |
| 7471089 | FAILED / 2:0 | 10 | Headless render passed; replay preflight exited2 for missing recorded seeds. |
| 7471240 | FAILED / 1:0 | 50 | Failed unconditional Decord import; made video-file dependency lazy. |
| 7471251 | FAILED / 1:0 | 546 | A full forward/backward passed; C duplicate registered checkpoint aliases rejected. |
| 7471310 | COMPLETED / 0:0 | 54 | 11 focused tests passed. |
| 7471447 | COMPLETED / 0:0 | 340 | C full forward/backward passed after alias adaptation fix. |
| 7471448 | COMPLETED / 0:0 | 53 | 12 focused tests passed. |
| 7471539 | COMPLETED / 0:0 | 180 | Cache refresh and actual large-asset SHA256 verification. |
| 7471646 | COMPLETED / 0:0 | 471 | A50-update fixed-example loss check and actual checkpoint reload passed. |
| 7471647 | COMPLETED / 0:0 | 367 | C50-update fixed-example loss check/reload passed; strict inference knots invalid. |
| 7471709 | COMPLETED / 0:0 | 48 | 14 focused tests passed. |
| 7471801 | FAILED / 2:0 | 604 | Python wrapper diagnostics completed; shell failed after live script edit. Superseded by7471966. |
| 7471802 | COMPLETED / 0:0 | 18 | Final cache source identity refresh for counted prediction safeguard. |
| 7471819 | COMPLETED / 0:0 | 49 | 15 focused tests passed. |
| 7471891 | COMPLETED / 0:0 | 37 | 16 focused tests passed, including identical VLM preprocessing. |
| 7471966 | COMPLETED / 0:0 | 626 | Frozen-source wrapper rerun completed; A decoded, C multiplicity rejection with32 repairs. |
| 7471967 | COMPLETED / 0:0 | 38 | 16 focused tests passed; exact optimizer/scheduler and Python/NumPy/Torch RNG checks. |

## Continuation

1. Recover authoritative recorded seeds/timestamps or agree on a newly collected, explicitly different dataset/protocol; keep existing data immutable. No conservative numeric clock can be inferred from these files.
2. Establish original replay and matched native controller clock, then local-window/prefix spline replay against the predeclared experiment_spec criteria. Do not waive gates or fabricate MOTUS_REPLAY_REPORT.
3. Integrate existing deploy wrapper into the installed evaluator under that controller protocol; record all outcomes, invalid predictions/repairs/fallbacks and synchronized warmed timings.
4. Profile effective batch16 on allocated hardware, then run authorized1000-update A/C pilot plus20 validation scenes only after gates pass. The one-example fixed-noise timing is not a reliable pilot/main cost estimate.
5. Log pilot learning/cost before any main matrix; full main matrix needs explicit authorization and sufficient budget.10000-update/3-seed plans remain unsubmitted.

See runbook.md for exact commands and paths. Monitoring ends with this session; none is claimed afterward.

Common input adaptation caveat: the raw-HDF5 manifest path directly resizes the
three-camera mosaic, whereas upstream MP4 loading supports aspect-preserving
padding. Both A/C use the same tested transform; exact pretrained-checkpoint image
preprocessing provenance remains unresolved. The original default MP4 dense path
is unchanged. This must be recorded/settled before claiming a fully vanilla data
pipeline reproduction; see experiment_spec.md. No hidden transformation change was
made after the diagnostic runs.
