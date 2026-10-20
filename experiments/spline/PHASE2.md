# Phase 2 design (not implemented or launched)

The training gate is closed. Finish the original baseline and matched Phase 1
evaluation, resolve the physical executor timing contract, and replay
ground-truth demonstrations before changing the action model. Phase 1 currently
provides no empirical evidence supporting training. Fixed-knot prediction is a
controlled representation experiment, not the full adaptive-knot BSP method.

## Targets, timing, and K

Use **ground-truth demonstration qpos**, through the existing RobotWin sampling
indices, rather than just predictions saved by the frozen policy. Split by
episode/task before statistics or K selection. Audit true raw-sample timestamps;
do not infer physical time from the MP4's encoding fps. Preserve the condition
time, original horizon `H=48*d`, sixteen original action timestamps `3*i*d`, and
eight video timestamps `t_j=6*j*d` for the default configuration. Preserve short
episode endpoint padding and include a validity mask in target fitting/loss
assessment so repeated padding does not falsely improve the K decision.

Fit a cubic on `[0,H]` with a common, **fixed**, clamped knot vector (four copies
of each boundary), fixed K, and twelve joint coefficients per token. Include the
observed initial joint state as the t=0 target in offline fitting to constrain
the previously unobserved start; this is an explicitly new Phase 2 target
conversion choice, not Phase 1 alignment. Its first coefficient can be trained
normally initially; an exact state anchor is a later separate ablation.
Evaluate candidates K=4,6,8,10,12 on development episodes, with original-timestamp
per-channel max/RMS error, dense intermediate checks for joint-limit overshoot,
gripper transitions and actual unchanged-controller replay. Choose the smallest
K meeting predeclared error and replay criteria, including contact-heavy cases.
If no K reduces tokens acceptably, do not claim the representation has succeeded.

Keep grippers as the two existing analog open-fraction targets at all sixteen
original action timestamps. Do not fit them as joint coefficients or relabel
them binary. Avoid sixteen additional transformer tokens: flatten the 16x2
gripper sequence into **one separate gripper token**, with its own input/output
MLPs and explicit full-horizon support. This preserves exact analog timing while
keeping the total action-expert length `1 state + K coefficients + 1 gripper + 4
registers = K+6`, compared with the original 21. K must be less than 15 for a
strict reduction. Coefficient and gripper variables have separate noise tensors,
targets and loss reductions; the gripper token is not a spline control point.

Initially fit per-control-index/per-joint coefficient mean/std on **training
targets only**, with a documented variance floor and invertible affine transform.
Checkpoint the knot vector, horizon/timestamp schema, K, normalization and data
split identity. Existing raw state features remain raw. Keep gripper values in
their original physical scale initially, with their own flow variable and loss.
Measure sensitivity to changed coefficient distributions.

## Model changes and checkpoint adaptation

Change root `models/action_expert.py`, `models/motus.py`, dataset conversion and
config creation; then expose the same representation to the deployment tree.
Keep dense action length N and coefficient token count K as distinct fields.
Assert the actual input length at **every** action block and joint QKV
concatenation: `K+6`. A smaller final output tensor with 21 internal tokens would
not implement the experiment.

Initialize from the released joint Motus checkpoint with an explicit key/shape
adaptation report. Existing per-token MLPs are not flattened N*D projections:

- Retain state encoder, action block attention/QKV/output projections, norms,
  FFNs, AdaLN modulation, time embedding/projection, registers, decoder norm and
  decoder modulation. Their shapes do not depend on N or K.
- Adapt the action input MLP's first linear layer from 14 to 12 inputs by copying
  joint columns 0:6 and 7:13; retain its hidden layers. Copy the output head's
  corresponding 12 rows and biases. This is only an initialization because
  normalized coefficients have different statistics from raw qpos. Account for
  coefficient normalization in initialization or allow those layers to adapt.
- Initialize the gripper token's 32-input/32-output projections separately;
  flattened sixteen-step grippers have no direct released weight equivalent.
  Reuse matching hidden blocks where justified; report every new parameter.
- Rebuild positional buffers deliberately. `strict=False` alone does not adapt
  mismatched tensor shapes; do not quietly discard arbitrary checkpoint keys.
- Freeze WAN/video, VAE, Qwen and understanding expert parameters initially.
  Train action expert plus coefficient/gripper input/output and temporal-support
  components. Print optimizer parameter names and assert frozen parameter
  gradients remain absent. This is an initial strategy, not a capacity guarantee.

Audit all paths with `action_chunk_size`, `chunk_size`, action shape construction,
state/register slicing (including zero-register pretraining), output heads,
time embeddings, training/sampling/alternative solvers, evaluation caches,
losses, serialization and duplicated deployment implementations. Begin with the
actual `inference_step` used by RoboTwin; disable other solvers for spline mode
until their coefficient shapes and schedules are tested.

## Flow matching and decoded errors

Root `Motus.training_step` samples separate video/action scheduler IDs. The
action scheduler uses `shift=5`, 1000 timesteps and `extra_one_step=True` from
`wan/utils/fm.py`. It uses the scheduler's **sigma** for mixing and its scaled
timestep (0..1000) for time embedding. These are distinct quantities.

For normalized clean coefficients C and Gaussian noise E, use:

```text
X_sigma = (1 - sigma) * C + sigma * E
V_target = E - C
L_coeff = mean((V_pred - V_target)^2)
```

Use noise shape `[B,K,12]`, separate gripper noise `[B,16,2]` flattened only at
its token interface, and separate loss normalization. Inference used by this
wrapper currently takes a linear grid from t=1 to t=0, uses t*1000 embeddings,
and integrates `X_next = X + V_pred * (t_next - t)` with **negative** steps.
Preserve that existing convention in the controlled comparison rather than
copying a sign or introducing a scheduler change along with the representation.

Begin with coefficient and gripper flow losses plus the existing video loss.
Evaluate inverse-normalized decoded trajectory error `B(t_i) C_pred - a_i` and
video-time action error `B(t_j) C_pred - a(t_j)`. Use a float32 differentiable
precomputed basis matrix in training. Never compare a noisy coefficient directly
to a clean controller trajectory. If needed, derive the clean estimate from the
mixture: `C_hat = X_sigma - sigma * V_pred`, since
`X_sigma = C + sigma * (E-C)`. An optional auxiliary loss is then
`||B(t) denorm(C_hat) - a(t)||^2`, with explicit physical-channel scaling,
validity masks and a separately evaluated weight. Do not enable it by default.

## Video/action alignment, attention, and gradient flow

A coefficient's support is `[u_i,u_(i+4)]`; its Greville abscissa is
`(u_(i+1)+u_(i+2)+u_(i+3))/3`. Map this to the original action-position coordinate
`t/(3*d)` (state at zero) and reuse the current sine/cosine embedding function
with fractional positions. Add a small zero-initialized projection of support
start/end/width in normalized physical time. Give the gripper token full-horizon
support and a type embedding; registers remain global with explicit positions.
This conveys basis support without pretending a coefficient occurs at a frame.
Keep noise-time AdaLN separate from these trajectory-time features.

Retain the existing **bidirectional** denoising attention in the first version.
There is no existing physical-time mask to resize. Build dynamic concatenated
lengths for K+6 and preserve Qwen padding rules and observation-only VLM/T5
conditioning. Future video/action inputs are their scheduled noisy variables;
only the condition-frame latent and observed state are teacher-forced. Do not
feed clean future frames, clean coefficients, fitted targets, or label-derived
knot locations into conditioning. Bidirectional denoising among noisy targets
is already the joint training convention; adding causal masks would be a new
experiment. If later adding support masks, use basis support intervals and the
video latent's causal receptive field, not a nearest-frame/control index rule;
audit indirect paths through registers and understanding tokens for leakage.

Video samples remain at their original t_j, and the spline is evaluated at t_j
when an action target there is needed. VAE latents still cover groups/receptive
fields of these video samples; coefficients do not set video sample positions.
Retain the existing video noise/teacher-forcing construction and masked
condition-frame video loss. Freezing WAN **parameters** does not mean wrapping
joint video computation in `no_grad`: frozen video operations must remain in
the autograd graph so video-loss gradients can pass through shared attention to
trainable action projections/tokens. Frozen understanding operations can also
carry such gradients after joint mixing. Keep VAE target encoding and genuinely
observation-only frozen Qwen feature extraction without gradients as before.

## Validation and later extensions

Before any expensive run: verify fixed-K demonstration fit/replay, basis
partition/endpoints/support, normalization round trip, one optimizer step,
finite flow losses, exact trainable parameter set, retained video-loss gradients
to action inputs, actual K+6 attention lengths, coefficient inference/decoding,
and one RoboTwin rollout from an explicitly labeled small training smoke test.
Training and development data must keep original timing. Test speed factors
only at inference once the execution clock is resolved. Relabeling timestamps
does not create physically accurate accelerated video targets; that would need
appropriate rollout data.

Compare success, original-time decoded errors, video loss, memory and measured
end-to-end observation/action latency. Time T5/Qwen, unchanged video denoising,
action processing, decoder and controller separately with proper device
synchronization. Fewer action tokens may leave latency dominated by video;
do not infer end-to-end speedup from token counts.

Only after fixed knots work: predict nonuniform internal knots and coefficients
with a fixed output budget and fixed physical horizon. Use positive normalized
interval lengths (e.g. softplus + epsilon normalized to H), cumulative interior
knots, fixed endpoints and four repeated boundary knots for cubics. Recheck
conditioning, coefficient statistics, support positions and differentiable
basis behavior. Video targets stay at their original t_j; adaptive knots do not
require adaptive video sampling. Variable horizons are a separate extension
requiring horizon conditioning, video sampling/positions and replanning changes.
