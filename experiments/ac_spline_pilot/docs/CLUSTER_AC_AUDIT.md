# Compute Canada A/C audit — 2026-11-15

> Historical research notes. Earlier commands, gates, and parameter counts
> describe previous stages. Use the [experiment README](../README.md)
> for the maintained preparation and training workflow.


Current configuration update: both variants require exactly 50 complete clean
`place_can_basket` demonstrations with the standard seed-0 45/5 split. They use
LoRA rank 64 / alpha 128 / dropout 0 on WAN and Action linears only, plus
`--report_to none`. The full manifest covers every episode; four windows limit
only generation previews. Earlier scope/count restrictions and unresolved scope
discussions below are historical evidence, not the current configuration.

Verdict: **blocked**. Independent audit and focused corrections are implemented.
No jobs were submitted. The requested full submission pipeline is **not yet
wired**: the session requests full fine-tuning, but every local definition of
A/C specifies matched LoRA. The user explicitly requested clarification before
wiring runs when definitions conflict. The pending question is whether to retain
the documented LoRA scope or use full backbone fine-tuning with frozen VAE/VLM.
This is an experiment-definition question, not a request for compute permission.

## Verified checkout and environment

`~/projects/aip-rjliao/kding` and `/project/6101778/kding` identify this workspace.
The workspace is not a Git repository; its children are separate repositories.

| Repository | HEAD before edits |
|---|---|
| Motus | `649063e76e41600856dcfd79dc5960230b631584` |
| WAM-RoboTwin | `58e399bf95cc1ab6423f26d7dd727d45030be170` |
| bspline-policy, read only | `61ed5f42fced971d50a89b46417493790876ccd1` |

Motus started clean. WAM-RoboTwin already had modified `.gitignore` and staged
`robotwin.sh`; these are preserved. No ancestor/Motus AGENTS.md was found.
XPolicyLab/AGENTS.md was read; no submodule files were edited.

Host is `klogin01`, with no current GPU allocation found by `squeue -u kding`.
All executed numerical/model-component checks used CPU in the existing
`containers/robotwin-cu121.sif`, not the full GPU model or simulator.
The working dependency overlay is
`Motus/.runtime/apptainer/lib/python3.10/site-packages`; `.runtime/python` is an
incomplete older overlay. Container Python is 3.10.12, torch 2.4.1+cu121,
NumPy 1.26.4, SciPy 1.15.3. Apptainer module is 1.4.5.

`scripts/cluster/environment.sh` reuses the existing image, project/RoboTwin
mounts, nvccli GPU access, Vulkan/EGL settings and headless runtime directory.
It supports an explicit CPU path and rejects GPU use without allocated Slurm
compute. It sets offline Hugging Face mode; jobs must not download assets.

## Experiment definitions and active entry points

Sources: workspace `motus_robotwin_research_plan.md`, `experiments/ac_spline_pilot/docs/RESEARCH_TRAINING.md`,
`experiments/ac_spline_pilot/docs/PILOT_REPORT.md`, `experiments/ac_spline_pilot/configs/robotwin_research_{A,C}.yaml`,
`data/robotwin2/{research_windows,adaptive_spline}.py`, `experiments/ac_spline_pilot/research.py`.

| Property | A | C |
|---|---|---|
| Output target | 16 x 14 dense absolute joint/gripper commands | 36 x 15 packed cubic spline parameters |
| Future action grid | source offsets 3,6,...,48, tail clamped | same grid at decoding; no per-window refit |
| Video targets | 8 uniform targets at offsets 6,12,...,48 | 8 targets at packed knot rows 4,8,...,32 |
| Action branch tokens | 21 including state + 4 registers | 41 including state + 4 registers |
| Current conditioning | current image, raw 14D command state, T5/VLM language | same |
| Precision | FP32 action targets/noise/projections; BF16 backbone computation | same |

C is native parameter prediction: the model's noisy target, input/output
interfaces, flow loss and generated sequence are 36x15 parameters. It does not
fit a spline after predicting dense actions. A/C jointly change representation
and video timing; they do not isolate those two effects.

Training entry: `experiments/ac_spline_pilot/train.py --config experiments/ac_spline_pilot/configs/robotwin_research_A.yaml` (or C).
Existing tiny launcher: `experiments/ac_spline_pilot/scripts/run_tiny_research.sh A|C`.
The checked-in research configs request **100 diagnostic updates**, microbatch 1,
accumulation 1, 10 warmup steps, 100-step cycle, four windows. They are not the
1000-update candidate schedule recorded in `experiments/ac_spline_pilot/docs/PILOT_REPORT.md`.
The trainer explicitly rejects phase != tiny or >100 updates; `load_contract`
rejects >4 training windows. These gates were not silently removed.

Existing diagnostic evaluation is `experiments/ac_spline_pilot/diagnostic.py`, called from
the trainer after training or `--diagnostic-eval-only CHECKPOINT`. This measures
generated action validity/error, not RoboTwin task success.
Deployment bridge is `inference/robotwin/Motus/deploy_policy.py`, with a separate
copy of the upstream model tree. It loads dense full checkpoints only and does
not load research bundles. Its `eval.sh` targets absent `script/eval_policy.py`.
The installed simulator launcher is `WAM-RoboTwin/scripts/eval_policy.sh`, which
routes through XPolicyLab. There is no active A/C deployment integration there.

Both research variants reset/train the entire action input MLP and output
projection. Compatible hidden projection initialization is name seeded across
variants. LoRA rank/alpha/dropout are 16/32/0, on WAN self-attention q/k/v/o and
FFN 0/2; action output-attention and FFN 0/2; understanding output-attention and
FFN 0/2 except the dead final understanding block. VLM, VAE, raw tensor QKV,
state/time/modulation weights and remaining pretrained weights are frozen.
All other compatible tensors load strictly before adapter insertion.
Exact trainable names/counts are emitted at actual model startup; fixture counts
are not full-model counts. Full backbone fine-tuning is not implemented by this
research configuration and is not substituted with LoRA without clarification.

## Demonstrations, action semantics and numerical checks

Exact installed task: `place_can_basket`, verified module/class and instructions.
Available clean dataset: 50 HDF5 episodes under
`WAM-RoboTwin/data/demo_clean/place_can_basket/aloha_agilex/data`.
These are v1.0 state/action HDF5, not original `/joint_action/vector` HDF5.
Installed `envs/utils/pkl2hdf5.py` and `scripts/process_data_xpolicylab.py`
establish action alignment: next recorded commands, with final observation
dropped. `/state/joint_states` is distinct current conditioning. It must not be
replaced with `/action/joint_states`, nor may the converter shift actions again.
The original MOTUS video/PT converter does not support this complete schema;
the new reader enables numerical spline conversion only. Full image/state/PT
adaptation remains a blocker, with row alignment preserved explicitly.

The shared reader now accepts verified v1.0 `/action/joint_states` and checks it
against the four component arrays in order 6 left joints, left gripper, 6 right
joints, right gripper. Original `/joint_action/vector` and PT paths are retained.
The installed aloha-agilex config identifies `fl_joint1..6`, `fr_joint1..6`;
arm commands are absolute radians, grippers continuous open fractions at columns
6 and 13. Existing simulator saturation to [0,1] is retained; no threshold added.

One real converted episode, selected after seed-0 splitting, is
`episode_0000003.hdf5`, 249x14. Bounds used only 45 training episodes; 5 validation
episodes are disjoint. No final held-out simulator scene list exists yet.
The real numeric report is `outputs/cluster-audit-20261115/real-v2/real-window.json`:

* Cubic fit maximum sampled normalized residual: **0.00964818292754055** (<= .01).
* Serialized FP32 36x15 packet vs global curve: max error **6.329058519632946e-8**.
* Local valid interval at observation 0: [0,134] source steps; all 16 dense-prefix
  commands decode with finite outputs and shape 16x14.
* Video indices/masks were inspected using source-row correspondence only.
  **No seconds/camera clock, actual image decode or video VAE was validated.**

Fitting uses normalized [-1,1] actions, shared adaptive cubic knots, clamped
endpoint knots, SciPy generate_knots and QR least squares. Endpoint action values
are not constrained. Canonical len(knots)=len(coefficients)+4. Local extraction
slices global support; 36 knots have 32 effective coefficients plus 4 structural
coefficient rows. Negative support offsets remain. Knot time is observation-
relative absolute source-step position, affinely scaled from training-only data.
All parameter rows are supervised, including deterministic padding. Decoder
rejects decreasing knots, invalid multiplicity/domain and short coverage;
only declared surplus tail endpoint repeats are trimmed. No sorting/extrapolation.
Action constants restore their training value; validation outliers are not clipped.

C uses conservative causal VAE validity masks: current+8 frames -> 3 latent
positions; invalid keys and padded losses are excluded, current latent loss is
zero, original video loss scale retained. A preserves repeated-tail behavior.
Actual pretrained VAE/mask receptive fields remain a GPU preflight requirement.

The old 500-episode randomized manifest and caches in
`outputs/place_can_basket-500` and `.runtime/datasets` use
`cubic-absolute-containing-v1`, a different preprocessing implementation and
schema. They cannot be treated as current `robotwin-episode-cubic-v1` sidecars.
New loader checks reject modified metadata and train/validation path aliases.
Full supported-prefix coverage across all episodes is still unverified.

## Checkpoints, optimizer and inference limitations

Existing base: `pretrained_models/Motus_robotwin2/mp_rank_00_model_states.pt`
(16,046,580,367 bytes). FakeTensorMode/mmap inspected 5,431 tensor entries without
allocating the full model. It includes VLM and module alias keys. Dense action
input weights are 1024x14; output 14x1024. C changes those to 1024x15 and 15x1024.
Research loading excludes/reset action projections and positional embeddings in
both variants, rejects other incompatible/missing tensors, then inserts adapters.
Actual full-model strict compatibility has not been executed. Prior task exposure
of this RoboTwin checkpoint is unknown; do not claim unseen-task generalization.

WAN VAE/T5 large weight files exist. WAN config.json, Qwen config/processor and
T5 tokenizer metadata are absent from the selected original/runtime metadata
directories. Their directories existing is insufficient. decord and wandb are
absent from the working environment; FlashAttention is present in the working
overlay but its GPU kernel was not run. WandB is now optional when reporting none.

The shared candidate schedule from PILOT_REPORT (not a selected full run) is
1000 updates, effective batch 16 = 1x16 accumulation on one process, AdamW
lr 1e-4 both parameter groups, betas .9/.95, weight decay .01, clipping .5,
warmup 200, linear cycle 5,000,000, f_max .99, f_min .4, training seed 0.
FP32 action arithmetic, BF16 transformer execution. Euler generation: 10 steps.
The first optimizer update previously bypassed warmup at full LR; scheduler
initialization now applies its starting multiplier before that first update.

Research checkpoints save canonical trainable adapters/projections plus embedded
contract/base identities; Accelerator saves optimizer, scheduler and RNG.
`research_adapter.pt` is not a standalone dense checkpoint. Missing resume paths
now raise instead of silently starting from step zero. Current resume derives
update count from the checkpoint filename and restarts a shuffled loader; exact
mid-epoch sample-order continuation is not implemented. Model/optimizer/RNG
resumption needs a full-model fresh-process check. Evaluation must select an
explicit completed checkpoint, never glob an arbitrary latest file.

`take_action(qpos)` uses TOPP, variable scene-step counts, fallback 50 steps on
planner failure, default timestep 1/250. Sixteen commands are not sixteen physics
steps. Keep matched commands per policy call; log actual physics steps, TOPP
failures and scene timestep. Physical retiming, adaptive horizon and speed scaling
were not introduced. Existing saved seed/path inputs for exact clean-demo replay
are missing (`seed.txt`, `_traj_data`), so exact controller-clock replay is blocked.

## Checks and remaining work

Passed: 16 spline regressions; deployment/loader/HDF5 tests; real-data conversion
and strict decoder above; CPU A/C component forward/backward/optimizer and
gradient checks; CPU accumulation; shell syntax and git diff --check.
The combined component suite's fresh-subprocess test timed out at 60 s under the
working overlay; a separate direct fresh reload completed successfully. This is
not reported as an entirely passing combined suite. See terminal/fixture outputs.
Full GPU model, VAE masks, real-image conditioning, optimizer/resume/reload,
ground-truth replay and RoboTwin rollout remain unverified. No task-success,
prediction-latency, policy-call or completion-time comparison has been measured.
No speedup is inferred from coefficient count. Training duration, prediction
latency and task completion time must be reported separately, with eval count,
success count, warmup/synchronization, denoise steps, execution horizon, seeds,
controller settings and paired-success counts.

Verified prior short jobs request account `aip-rjliao`, partition
`gpubase_l40s_b1`, 1 L40S, 4 CPUs, 64G RAM, at most 30 minutes. These are **short-
job settings**, not evidence a 1000-update full fine-tune fits this allocation.
Training/evaluation walltime and full-fine-tune memory feasibility remain unset.
The upstream 8-GPU/1.5TB launcher has placeholders and is not resource evidence.

From Motus:

```bash
bash scripts/cluster/submit_ac.sh --task place_can_basket --dry-run
bash scripts/cluster/submit_ac.sh --task place_can_basket --submit
```

Dry run prints resolved existing/candidate paths/resources, repo status, CPU check
commands and explicit blockers. **Submit currently exits 2 before sbatch.** It is
not a usable full A/C launcher yet. No training commands, dependencies, job IDs,
checkpoint/result paths or resume execution are claimed finalized. Planned
persistent layout is `outputs/ac/RUN/{A,C}/`. The resolver accepts
`--experiment A|C|both` and validates `--resume PATH` syntax, but does not launch.

After resolving the training scope, remaining implementation is to adapt the
real data with observed state/action alignment; establish verified clocks/splits;
integrate research bundles with the existing installed evaluator; put GPU model,
resume/reload and simulator checks into a fail-fast short preflight; gate full
runs on success; freeze both repos/configs/dependencies into a checked snapshot;
create logs before sbatch; record job IDs; orchestrate A/C independently with
success dependencies and combined measured results. Do not run from a mutable
checkout while queued. No snapshot here is being used by a queued job.

No job inspection/resume command is actionable yet. Future Slurm inspection is
`squeue -u "$USER"`, `sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS`,
and `tail -f RUN/logs/JOB_ID.log` once actual job IDs/logs exist.
