# A/C experiment contract

## Current one-task override (2026-11-12)

The latest user clarification supersedes matched A/C fine-tuning for this request:
one task (`place_can_basket`), 500 distinct randomized demonstrations, LoRA rank
**256**, fine-tune **C only** with seed 0, and test both pretrained dense MOTUS
and fine-tuned C. Use the deterministic 450-train/50-validation episode split,
with normalization fit only on the 450 training episodes. Start with the existing
1000-update pilot schedule; no sweep or three-seed/four-task submission is
included. Extra demonstrations do not guarantee against overfitting.

This comparison changes fine-tuning, action representation and video alignment;
it does not isolate spline benefits or match training budgets. Original pretrained
weights are loaded strictly and receive zero-output adapters solely to reuse
shared input/controller code. Both policies use the same common input adaptation,
ten Euler steps, BF16, fixed replanning prefix and verified controller clock.
Final testing uses 100 predeclared scenes disjoint from all demonstration scenes,
with identical instructions and checked initial physics states across policies.
No test results or successful simulator integration are claimed yet.

Preserve alpha 32 (alpha/rank = 0.125), dropout 0 and the existing attention
targets and fully trainable action projections. Earlier rank-16 diagnostics do
not validate rank-256 resource use or learning. Other matched settings and all
repair/replay acceptance criteria remain in force. The 500 randomized demos are
a different collection from the historical 50 clean demos; both A/C must use the
new collection. Seed files are present in the new archive, but source timing and
simulator compatibility remain unverified. Historical budgets below have ended;
no renewed budget is implied by this scope update.

A is MOTUS dense14-D actions / native16 action rows / native8 uniformly sampled
future frames. C predicts36x15 spline parameters and8 future images selected from
offline ground-truth knots. This is a combined action/video-supervision comparison
with matched LoRA, not isolated representation evidence or full fine tuning.
**Published C budget is larger than native A; compression is not demonstrated.**

Primary target: preserve success while reducing directly measured inference
latency and/or policy computation per episode. No speed scaling, asynchronous
execution, adaptive replanning, B condition, quantization or sweeps.

## Data and training

50 clean Place Can Basket demonstrations available. Exact task ID:
`place_can_basket`. Main candidates `stack_blocks_two`, `open_microwave`,
`handover_block` also each have50 episodes. Never substitute move_can_pot.
RNG0 permutation splits each task45 train /5 validation before min/max fitting.
A and C use identical manifests and eligible observations (C's unsupported video
tails and short action coverage excluded for both). Manifest stores raw SHA256,
size/mtime, conversion/spline source hashes, settings and statistics identity.
Raw data/checkpoints unchanged. Single scalar HDF5 instruction per episode is
used for both, a reduction in paraphrase diversity from upstream random sampling.

Pilot seed0,1000 optimizer updates,20 validation scenes intended; no pilot success
claims from smoke. Main3 training seeds (0,1,2), shared four-task policies,
10000 final updates, effective batch16; existing loader already samples uniformly
by task then episode. Main submission requires logged pilot cost and authorization.
No main matrix submission in this session.

One `configs/robotwin_ac.yaml` supplies both conditions. Microbatch1, accumulation16
(provisional until profile), BF16. AdamW betas(.9,.999), eps1e-8, lr1e-4,
weight_decay.01 on eligible matrices, none on biases/norms. Global clipping1.0,
linear5% warmup then cosine to10%. LoRA16,alpha32,dropout0. WAN self/cross attention
q/k/v/o and action joint fused QKV/output get adapters; exact names/counts saved.
Language/understanding expert, observation encoders, VAE and pretrained weights
frozen. Action action-input MLP and action-output MLP fully trained; observed-state
encoder frozen. No pre-existing MOTUS LoRA implementation or installed PEFT was found;
small explicit adapters support its custom [3,heads,in,head_dim] QKV tensor.

Upstream defaults differ: full WAN/action/understanding training, betas(.9,.95),
clipping.5,linear scheduler. The matched recipe deliberately changes those for both.
Original flow noise-clean targets and independent uniform scheduler timestep
sampling retained exactly (the YAML logit_normal label does not control the actual
training_step; code inspection takes precedence). C prepends zero knot input/output
weights while copying all14 original action columns/rows; deterministic positional
buffer regenerated. Reject unexplained checkpoint incompatibilities with saved lists.

## Replay gate and acceptance (before full replay)

Require matching recorded scene seed/initial physics state and native joint/gripper
semantics. Original replay must succeed before attributing loss to conversion.
Global reconstruction <=.01 maximum error in training-normalized coordinates.
Local/global error <=1e-5 on support; no nonfinite values, extrapolation or silent
repair. Paired pilot replay: at least5 recorded validation demonstrations; require
no lost success among original-success episodes, no simulator/controller error,
no gripper threshold crossing displaced by >1 source sample, and no additional
transition jump >.01 normalized units beyond the source transition. These numerical
thresholds are experiment choices, not papers' constants. Tracking acceptance in
radians/seconds is unresolved until timestamps/controller clock are recovered.
Do not declare replay accepted based on numerical fitting alone.

## Execution and evaluation gates

Baseline deployment consumes16 predictions before observing again, at source offsets
3,6,...48. C decodes at the same offsets; insufficient coverage invalidates the call,
counts it and fails the episode. Both unmerged adapters, same BF16, same10 Euler flow
steps. Native simulator `take_action(qpos)` internally TOPP-retimes waypoints;
its physics timestep is1/250s, but controller command period is not established.
This does not yet satisfy the requested fixed-frequency matched-replanning protocol.
No claim of fixed physical duration is permitted until integration is validated.

Ground-truth knots are training supervision only. Inference receives current image,
state and instruction; no future demonstration image retrieval. No clean future
observations are fed into current conditioning. Joint noisy video/action training
retains MOTUS's original attention flow.

Prepare20 fixed validation scenes and disjoint100 final scene seeds/task/trainseed
only after scene replay/reset compatibility is established. Final checkpoint at10000,
periodic resumable saves; no test-based selection. For each episode record task,
training/scene seeds, config/checkpoint, success/failure/timeout/reason, controller
steps and simulated time, wall time, policy calls, direct observation-to-command
latency (CUDA synchronization after warmup), decode time, invalid/repair/fallback/
short-coverage counts, GPU peak memory. Report all outcomes and paired-success
timing with denominators. Per task/seed success and seed spread; Wilson intervals
within seed explicitly conditional on that policy, not evidence for training-seed
uncertainty with only3 seeds. Median/p95 latency. Queue delay reported separately.

## Budget

Unattended authorization: one GPU/job, max2 concurrent, max4h/job,16 total GPU-hours
including retries. Known working single L40S/account aip-rjliao,4CPU,32GB reference
scripts; use64GB for model initialization. Diagnostics15min or shorter; conversion
30min. Training/pilot max4h only after gates pass. Existing scripts alone do not
authorize extra resources. Record job IDs in status, check squeue before submission,
inspect failed logs before retry. Full main matrix remains unsubmitted.

Adapter initialization uses a forked RNG with the training seed, independent of
representation-specific model-construction RNG consumption. Compatible adapter
shapes therefore share initial values; zero adapter output preserves checkpoint
attention initially. New C knot input/output coordinates initialize to zero.

A/C deployment reuses `utils.vlm_utils.preprocess_vlm_messages` exactly. The upstream
copy used text/image order and a generation-prompt flag different from training;
that mismatch is corrected only for the explicit LoRA experiment path. Unit
comparison requires exact input IDs, masks, image grids and pixel tensors.
Legacy deployment retains its original preprocessing.

Raw-HDF5 input adaptation is common to A/C: decode through XPolicyLab's two-format
RGB helper, build the existing head/top plus wrist/bottom mosaic, and resize to
384x320. This uses direct resizing, while the upstream MP4 reader has an
aspect-preserving padding path. The original checkpoint's precise preprocessing
provenance is not established from the local placeholder conversion configuration.
Do not describe this as an exact reproduction of the paper's entire baseline data
pipeline. The default folder/MP4 dense path remains unchanged. The current A/C
comparison preserves dense-versus-spline targets and shares these input adaptations.
