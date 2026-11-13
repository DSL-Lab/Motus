# Stage 3: matched dense A and adaptive spline C

Implementation is prepared; **full MOTUS real-data learning and GPU reload
generation have not run**. This host has CPU-only PyTorch and no real episodes,
model assets, verified physical clocks, SAPIEN, or authorized GPU budget. Nothing
was submitted to Slurm. The full pilot is explicitly blocked in the entry point.
Stages [1](data/robotwin2/ADAPTIVE_SPLINE.md) and
[2](data/robotwin2/LOCAL_SPLINE.md) remain historical results, not replay evidence.
The adaptive fitter, 36-row representation, affine knot encoding and strict shared
decoder are unchanged.

## Implementation

* `configs/robotwin.yaml` remains the original dense entry. New
  `robotwin_research_A.yaml` inherits it; `robotwin_research_C.yaml` inherits A
  and changes only the representation. Both use the same frozen base checkpoint,
  windows, current observations, raw 14D proprioception, language index, adapter
  scope and optimizer settings. A uses original stride-3, 16-by-14 commanded
  actions and uniform future video. C uses `extract_local`'s 36-by-15 packet and
  `select_future_frames`'s knot-aligned video. No windows are refitted.
* The existing `RobotWinTaskDataset`, factory and collate function gained an
  explicit fixed-window-manifest path. The legacy random dense path remains.
  Real-data input must be original MOTUS PT commands plus the existing video,
  metadata text and T5 embeddings. Source equality, split membership, statistics,
  stage-1 residual tolerance, shared-decoder round trip, camera clocks and video
  length are checked. Errors propagate; no retries silently substitute windows.
  Windows whose nearest current image is in the future are rejected, preventing
  future-image conditioning leakage. Current-frame VLM preprocessing is reused.
* For this paired tiny diagnostic, each explicit window must cover the complete
  unchanged dense prefix (16 commands at source offsets 3,6,...,48), and contain
  at least one valid future VAE latent. Insufficient local coverage or an
  entirely padded video tail is an explicit error for **both** A and C. Nothing
  expands the knot budget or silently shortens the executed prefix. These checks
  can block contact-dense windows and are a remaining pilot coverage issue.
* `MotusConfig` separates action parameter rows from video length. C keeps 14D
  state conditioning but changes action boundary widths to 15 and noisy sequence
  length to 36. Action positional embeddings retain ordered row positions;
  knot time is an explicit target column, not positional embedding replacement.
  There is one token per row plus one state and four registers: **21 action-branch
  tokens for A; 41 for C**. Video and understanding tokens are additional and
  not included in these counts. Video RoPE/unpatchify grids now come from actual
  VAE/patch output dimensions, including evaluation batch sizes.
* Research action targets, noise, sigma arithmetic and sampled parameter latents
  stay FP32 in both A/C; transformer computation remains BF16. Trainable action
  projections are FP32 with explicit boundary casts, avoiding silent BF16 knot
  target quantization. Legacy non-research precision is retained.
* The existing flow convention remains
  `x_sigma=(1-sigma)*target+sigma*noise`, `velocity=noise-target`; inference
  integrates from sigma=1 to 0 with negative steps. Actual training samples
  uniform **scheduler indices** on the existing shifted sigma grid, independently
  for video/action, with new random Gaussian noise each call. The config records
  this rather than claiming the previously unused logit-normal setting is active.
  The tiny generation diagnostic uses existing Euler inference (10 steps).

## Padding and video masking

The same conservative causal VAE prefix validity from stage 2 reaches the model.
The current image plus eight future images produces three expected VAE latents;
the actual output length must agree before loss computation. Padded latent
positions remain queries but are excluded as keys in joint video/action/
understanding attention. Unmasked attention still uses the existing path; masks
with invalid keys use native PyTorch SDPA. [PyTorch's boolean mask convention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
is used: True means an allowed key.

C's video loss excludes invalid/current latents and retains the original dense
loss scale (the zero-loss current position's fraction remains in the denominator
convention). When all future latents are valid it equals the original dense loss;
padding does not accidentally increase the video weight by 1.5. A retains its
original repeated-tail video behavior. No mask is applied to action parameter
rows: structural/repeated coefficient rows remain supervised as specified.

CPU attention/loss isolation is verified. Causal VAE masking with actual weights,
GPU SDPA/FlashAttention and full-model gradients remain pending; these tests are
not proof of full VAE receptive-field behavior or pilot readiness.

## Adapters and optimizer audit

There was no LoRA implementation in the active MOTUS training path. The small
`utils/research.py` utility adds registered linear LoRA modules, not another
trainer. Compatible base weights are loaded **before** insertion and then frozen.
Any unexpected missing/shape-incompatible frozen base tensor is an error. A/C
both reset and fully train the complete action input MLP and output projection;
state encoders remain loaded/frozen. Name-seeded reset keeps compatible hidden
projection initialization matched despite different boundary widths.

Matched scope, rank 16 / alpha 32 / dropout 0:

* WAN block self-attention q/k/v/o and FFN linears 0/2.
* Action block `wan_action_o` and FFN linears 0/2.
* Understanding block `wan_und_o` and FFN linears 0/2 **except the last block**.

The terminal understanding outputs are discarded after final joint attention,
so their O/FFN cannot influence video/action losses. Adapting them would create
unused trainables. Raw tensor QKV projections, VLM, VAE, time/modulation/state
parameters and all other base weights remain frozen in both configurations.

Startup verifies that every trainable parameter occurs in the optimizer exactly
once and no frozen parameter occurs there. `trainables.json` records exact names,
shapes and counts for the actual checkpoint. `gradients_step_{0,1,2}.json` records
finite gradient norms; a missing gradient path on step 2 is a hard error. Zero
initial A-side LoRA gradients at the first step are expected because B starts zero.

The trainer now calls the wrapped model's forward method, uses Accelerator
accumulation, clips/advances the scheduler only at optimizer updates, and counts
updates rather than microbatches. Accumulation crosses dataloader boundaries
without flushing; this follows the
[Accelerate accumulation contract](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation).
The existing custom linear scheduler is explicitly registered for checkpointing.

Pilot defaults are recorded as lr=1e-4, effective batch 16, 1000 updates, seed 0.
**Diagnostic overrides:** 100 updates, batch 1 / accumulation 1 (effective 1),
10 warmup steps, 100-step decay, four fixed training windows maximum, and four
generation seeds [0,1,2,3]. Adapter scope/rank, loss weights, clipping and optimizer
settings match A/C. These settings are not evidence of convergence.

## Checkpoints and generation

Research checkpoints use the existing Accelerator optimizer/RNG/state save/load
path with model hooks for `research_adapter.pt` (all canonical trainable adapters
and full projections). Frozen multi-billion-parameter weights are referenced by
the selected compatible base's path/size/mtime identity instead of duplicated.
`research_config.json` stores full resolved/model configurations, adapter scope,
representation, action normalization **contents/identity**, time scaling
**contents/identity**, channel semantics, video convention and window-manifest
identity. This is in addition to the legacy `config.json`. Restore reconstructs
the same frozen base, inserts the same adapters, verifies identities/name/shape
membership, and copies every saved trainable. Changing representation,
normalization, time scaling or base rejects the bundle. Base task exposure is
still unknown until the chosen checkpoint's provenance is supplied.

`train/research_diagnostic.py` contains evaluation only; training stays in
`UniDiffuserTrainer.train`. It selects fixed dataset indices and fixed language
indices rather than a shuffled first batch. Existing `inference_sample` no longer
even reads `video_frames`; generation receives only current image, raw state,
T5 context and current-frame VLM input. Future demonstration images and packets
are scoring/training targets only.

For every window/seed the diagnostic saves raw parameters, raw knot monotonicity
and domain validity, strict full-prefix decoding outcome, failure reason and
per-channel decoded command RMSE. Invalid knots or short coverage are never
sorted, repaired or substituted. Invalid runs still get plots showing targets
and their rejection reasons. Per-channel plots distinguish joints and continuous
grippers. A new Python process reloads the saved bundle and compares raw seeded
generation against the first process (tolerance 1e-5); mismatched windows/seeds or
different results fail explicitly. This checks reproducibility, not convergence.

## Actual local checks

**8 CPU component tests** plus **16 stage-1/2 regression checks** pass. Syntax
compilation and Bash `-n` pass. CPU tests load the actual expert, norm, positional,
joint-attention, loader/collate and trainer definitions without importing absent
VLM/diffusion dependencies. Image decoding is stubbed only in the loader contract
test; unmasked GPU FlashAttention is replaced by CPU SDPA only in the component
harness. Neither substitution is used by training code.

The two-layer CPU fixture uses action hidden size 16, WAN size 12, understanding
size 8 and **rank 2 / alpha 4** (test-only overrides). Actual fixture results:

| Configuration | Trainable parameters | Trainable tensors | Adapted linears | Action-branch tokens |
|---|---:|---:|---:|---:|
| A | 2326 | 50 | 21 | 21 |
| C | 2359 | 50 | 21 | 41 |

All intended fixture trainables receive finite nonzero gradients after three
updates against action/video outputs only; terminal understanding parameters
stay frozen. The fixture additionally verifies that masked video values cannot
affect action/understanding outputs, padding has zero video-loss gradient,
conditioning matches A/C, action flow/time/noise actually resample, and eight
microbatches over a three-item loader yield exactly two optimizer/scheduler steps.

The small adapter/projection bundle reloads in a **fresh Python process** with
maximum output difference **0.0**. These are component tests, not real MOTUS
tiny-data results. No policy prediction plots or real learned validity/error
numbers are claimed. The full model's trainable count will be emitted on its
first authorized startup, not extrapolated from this fixture.

Reports are retained under `outputs/stage3-components/` (test output prints the
unique directories). Run locally from Motus:

```bash
python train/test_research.py -v
python -m unittest discover -s data/robotwin2 -p test_adaptive_spline.py -v
```

## Pending real inputs and exact allocated-node commands

Use an existing complete MOTUS environment (including repository requirements,
SciPy >=1.15 for `generate_knots`, decord and CUDA PyTorch). No job or allocation
has been requested. No use is made of the repository's placeholder 8-GPU/1.5-TB
launch template. `run_tiny_research.sh` is a thin wrapper around the **same**
trainer, then a second invocation of that trainer for reload-only evaluation;
it never invokes sbatch and checks it is on allocated compute. GPU model
execution rejects login-node environments, DeepSpeed, multiple processes and
full-pilot phases until their corresponding remote validation exists.

First supply training-only normalization and time statistics from stages 1/2,
compatible existing model weights, and a tiny fixed real-window manifest. The
manifest uses absolute artifact paths and at least one separate validation
episode/window. Example structure (replace paths with existing files):

```json
{
  "schema": "motus-fixed-real-windows-v1",
  "windows": [
    {"split":"train", "observation_index":0, "language_index":0,
     "task_name":"place_can_basket", "episode_name":"0", "data_mode":"clean",
     "qpos_path":"/data/clean/place_can_basket/qpos/0.pt",
     "video_path":"/data/clean/place_can_basket/videos/0.mp4",
     "lang_path":"/data/clean/place_can_basket/umt5_wan/0.pt",
     "cache_path":"/sidecars/episode0.npz", "timeline_path":"/replay/episode0.timeline.json"},
    {"split":"validation", "observation_index":0, "language_index":0,
     "task_name":"place_can_basket", "episode_name":"validation-id", "data_mode":"clean",
     "qpos_path":"/data/clean/place_can_basket/qpos/validation-id.pt",
     "video_path":"/data/clean/place_can_basket/videos/validation-id.mp4",
     "lang_path":"/data/clean/place_can_basket/umt5_wan/validation-id.pt",
     "cache_path":"/sidecars/validation.npz", "timeline_path":"/replay/validation.timeline.json"}
  ]
}
```

`/data`, `/sidecars` and `/replay` above are schema examples, not verified cluster
locations. The dataset root must also contain the original `metas/<id>.txt` files.
Validation episode membership must match the normalization's fixed split.

In the existing environment, from the actual Motus root, set these to real
existing paths; their values are shared by A and C:

```bash
export MOTUS_WINDOWS=/absolute/path/tiny-real-windows.json
export MOTUS_NORMALIZATION=/absolute/path/training-action-normalization.json
export MOTUS_KNOT_SCALING=/absolute/path/training-knot-scaling.json
export MOTUS_DATASET_ROOT=/absolute/path/preprocess/robotwin2
export MOTUS_BASE_CHECKPOINT=/absolute/path/compatible-original-MOTUS-checkpoint
export MOTUS_WAN=/absolute/path/Wan2.2-TI2V-5B
export MOTUS_VAE=/absolute/path/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export MOTUS_VLM=/absolute/path/Qwen3-VL-2B-Instruct
```

**Only after an explicit compute budget has been approved and its allocation
already exists**, run within that allocation (these commands do not allocate or
submit a new job):

```bash
export MOTUS_DIAGNOSTIC_OUTPUT="$PWD/outputs/tiny-A-unique-run"
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=4 bash scripts/slurm/run_tiny_research.sh A

export MOTUS_DIAGNOSTIC_OUTPUT="$PWD/outputs/tiny-C-unique-run"
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=4 bash scripts/slurm/run_tiny_research.sh C
```

Each variant performs 100 updates and 4-seed generation on at most four windows,
then launches a fresh Python reload/generation process. Required GPU memory and
wall time have not been measured; the approved allocation must cover the run.
Never execute the Python training command on a login node. Fresh output paths
are mandatory to preserve prior results.

For variant C, expected result locations under `$MOTUS_DIAGNOSTIC_OUTPUT`:

* `train/robotwin_research_C/tiny/trainables.json` and first-three-step gradients.
* `train/robotwin_research_C/tiny/checkpoint_step_100/` with adapters, projections,
  embedded inference contract and Accelerator optimizer/scheduler/RNG state.
* `train/robotwin_research_C/tiny/generation/{results.json,predictions.npz,window-0.png,...}`.
* `reload/robotwin_research_C/tiny/fresh_reload/{results.json,predictions.npz,window-0.png,...}`.

Inspect all validity failures and per-channel decoded errors before interpreting
loss. A low training loss with invalid generated knots is a failed diagnostic.
Fresh-process reproducibility with inaccurate predictions is not a learning pass.

**Remaining pilot blockers:** real conversion/local/frame checks, ground-truth
matched replay (not yet run), training-only statistics and clocks, compatible
checkpoint/provenance, full-model VAE/mask/gradient audit on GPU, multi-noise real
generation/plots, full-model fresh reload, and resolution of short-coverage/tail
windows under the unchanged matched execution prefix. The 1000-update pilot has
not started and remains blocked even if a config requests it.
