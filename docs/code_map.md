# Proposed minimal integration map (2026-11-11)

500-demo preparation (2026-11-12): `scripts/experiments/prepare_500.sh` stages the
pinned archive on the network-enabled login host, then submits the existing Slurm
wrapper. `motus_actions/prepare_500.py` validates/extracts the ZIP inside Slurm,
reuses `WAM-RoboTwin/tools/convert_legacy_data.py:convert_job`, and calls existing
`motus_actions.prepare:prepare` / `prepare_language`. The final 500-episode manifest
is published only after all stages finish, with raw/archive/adapter/seed provenance.
`tests/test_prepare_500.py` checks inventory completeness and safe member paths.
Full allocated execution remains pending; no reference repository was edited.

Latest scope (2026-11-12): `scripts/slurm/finetune_spline_then_test.sh` calls the
shared Slurm snapshot wrapper and `scripts/experiments/train_spline_then_test.sh`.
The payload reuses the existing one-task rank-256 configuration and C trainer,
then `inference/robotwin/evaluate_pair.py` tests pretrained dense and fine-tuned C.
`MotusPolicy._load_model` has an explicit pretrained/zero-adapter loading branch
for the shared experiment input path. Core outcome/held-out-suite tests are in
`tests/test_paired_evaluation.py`. Full runtime validation remains pending; the
reference repositories were read without editing them.

One-task entry point (2026-11-12):
`scripts/experiments/finetune_one_task.sh TASK A|C|both` fixes LoRA rank 256,
checks a 500-episode/450-50 manifest and existing validation reports, and submits
sequential jobs through the existing Slurm wrapper. Each allocated job derives a
resolved configuration from `configs/robotwin_ac.yaml` and calls the existing
`scripts/train.sh` via its new optional `MOTUS_TRAIN_CONFIG` setting. No new
training loop or evaluator was introduced. Dataset preparation remains pending.

Repair follow-up (2026-11-12): `utils/robotwin_prompt.py` owns the original
converter's scene-prefix convention. Dataset/deployment now share this text and
camera layout. `train/train.py:create_accelerator` carries accumulation across
epochs; the existing trainer records per-update examples and safe resume state.
`MotusPolicy.decode_spline_prediction` is the staged diagnostic decoder.
`allocated_overfit.py --varied` extends the same trainer for varied noise and
fresh-process audits. `allocated_replay.py` reuses
`WAM-RoboTwin/trajectory_tools/replay.py:execute_fixed_rate`; it adds scene checks,
physical gripper traces, and original-first execution. New tests/diagnostic
branches are validation code, not a second training/evaluation framework.
See status.md for verified results versus pending runtime checks.

Source inspection precedes edits. Proceeding under unattended authorization.

- `data/robotwin2/robotwin_agilex_dataset.py`, `data/dataset.py`: retain existing sampling/loader; add explicit episode manifests, cached spline targets and knot-selected video indices for C. Existing multitask loader already samples tasks uniformly.
- `motus_actions/spline.py` (new): cohesive offline adaptive fit, exact local curve extraction, packing/scaling, video index mapping and strict matched-prefix decoder. This separates operations without introducing alternate training infrastructure.
- `models/motus.py`: representation switch derives action tensor width/length; proprioception remains 14-D; preserve original flow objective and current-frame conditioning. Add checkpoint compatibility report.
- `utils/lora.py` (new): only if no existing support is found; adapters on actual WAN attention linear projections and custom action fused QKV parameter, plus action input/output projections. No generic all-linear targeting.
- `train/train.py`: extend existing trainer for matched optimizer recipe, real gradient accumulation and resumable state; use same path for A/C.
- Existing `scripts/train.sh`, `scripts/slurm/slurm_single_node.sh`: parameterize existing launchers. Small conversion/replay/smoke entry points reuse existing simulator integration, without copying reference repositories.
- `inference/robotwin/Motus`: first inspect copied model path, then route experiment inference through the primary implementation to avoid maintaining duplicate spline models.
- `tests/`: numerical independent spline checks, alignment, custom QKV adapters, gradients, state round trip and model integration.

Validation order: one real offline example and decode, numerical checks, original replay and paired spline replay, allocated GPU smoke, then pilot only after replay/model checks pass. Missing verification is a gate, not a successful result.

## Provenance at start

- Motus: `git@github.com:DSL-Lab/Motus.git`, `88f6fe30c62af033874b0a2e48ad1cbdaf12cfdf`, clean.
- bspline-policy: `git@github.com:B-spline-policy/bspline-policy.git`, `61ed5f42fced971d50a89b46417493790876ccd1`, clean.
- WAM-RoboTwin: `git@github.com:qiyan98/WAM-RoboTwin.git`, `74f239e1a10f7b528d7488af956ee1e75f959415`, dirty `XPolicyLab` submodule preserved. This is not verified official SplineWAM code.
- No nested AGENTS.md in Motus or bspline-policy. WAM's nested `XPolicyLab/AGENTS.md` must be read before any edits there (none planned).
- Scratch `Motus_old_outdated` is only a candidate source of missing immutable metadata/environment assets, not the implementation target.

## Implemented scope after allocated smoke

Primary implementation follows the map above. `utils/vlm_utils.py` reuses the
vendored Qwen helper when the external package is absent; `data/utils/image_utils.py`
loads Decord only for video-file operations. No dependency installed.

`motus_actions/prepare.py` reads existing raw HDF5 into versioned numerical caches
and an inspectable example; `spline.py` owns fitting/extraction/scaling/alignment/
decoding/native command validation. `utils/lora.py` is necessary for the custom
fused QKV tensor and strict initial weight adaptation, including registered aliases.

Tests: `test_ac.py` numerical/model/accumulation/state checks;
`allocated_model_smoke.py` actual pretrained forward/backward;
`allocated_overfit.py` invokes the existing trainer on a fixed example/noise, reloads
its actual checkpoint and exercises current-only inference;
`allocated_replay.py` only renders/preflights recorded replay, which remains blocked.
There is no second trainer or policy evaluator.

One `smoke_test.sh` selects numerical/model/overfit stages; temporary diagnostic
wrappers were removed after preserving their job snapshots/logs. Shared
`scripts/train.sh`/`slurm_single_node.sh` replace placeholder launch settings.
`convert_splines.sh`, `replay.sh`, `aggregate.sh` are thin allocated conversion,
render/replay preflight and read-only diagnostics/report entry points.
A working simulator evaluation adapter and full raw episode metrics are still
outstanding; no stub evaluator is presented as a working implementation.

Final review removed redundant one-off smoke wrappers and unused imports. Core
model diff is limited to representation validation/derived tensor dimensions,
action projection width and inference noise shape. Existing LoRA-disabled dense
behavior remains, and no reference source file was edited. Production simulator
metrics/evaluation scripts were not fabricated: their unresolved integration is
listed in status.md, with no claim that the requested experiment has completed.
