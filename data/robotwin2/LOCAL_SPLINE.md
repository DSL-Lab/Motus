# Stage 2: local targets, decoder, video inspection and replay

No model, training loop, deployment policy wrapper, simulator source or reference
repository was modified. The adaptive episode fitting algorithm is unchanged.
All stage-1 files/results and the pre-existing WAM-RoboTwin submodule change
remain in place. `adaptive_spline.py` is the shared extraction/decoding API;
`visualize_episode_spline.py` and `replay_episode_spline.py` both call it.
The dense deployment wrapper remains dense; future spline deployment must import
`decode_local`, rather than implementing another decoder.

## Stage-1 recheck and reference boundary

The stored synthetic fit was reopened and its normalized error recomputed:
**0.009547563666154613**, exactly matching metadata. The gripper plot was opened
again; its between-sample overshoot persists. No real demo has become available.
SAPIEN, embodiment assets, local demonstrations and saved expert trajectories
are absent. There is no claim of real-image inspection or successful replay.

The 36-row allocation and observation-relative absolute knot positions were
verified against [SplineWAM Appendix D, printed pp. 17–18](https://arxiv.org/pdf/2609.39873).
The installed read-only B-spline Policy functions `extract_unique_knots` and
`chunk_bspline_trajectory` were also inspected. Tests execute those two pure
functions directly, avoiding their heavy dependency tree. The paper's codebase
is described as Fast-WAM-derived; WAM-RoboTwin is not identified as that codebase.

The 36-by-15 contract is retained for 14 channels: 540 scalars, **not a verified
transformer token count**. With 36 knots the cubic domain is [u3,u32] and only
32 coefficients affect it. The final four coefficient columns are structural;
the knot column in those same rows remains significant. This is a budget, not
a relationship to MOTUS's current 8 future video frames / 16 dense actions.

## Shared numerical contract

`extract_local(spline, observation_time, scaling)` selects the containing span
(right-hand span at an interior knot, final left-hand span at episode end),
starts three support knots before it, and copies up to 36 contiguous knots and
36 coefficient rows. There is no refitting. Missing tail entries repeat the
last source entry, exactly as in the installed reference. Original future
coefficients in the structural rows are retained when available. Source indices,
structural rows, repeats, effective coefficient count, span, observation and
valid/future intervals are returned. Negative support offsets are retained.

Knot coordinate = `(global_knot - observation - offset) / scale`. The packet
is float32, matching a prospective model target. Coefficients remain in the
stage-1 normalized action space and are never clipped. Every parameter row is
supervised under the reference convention; validity metadata is not an action
loss mask. In a short tail, fewer than 32 coefficients are effective.

`decode_local(parameters, original_timestamp_grid_in_source_steps, observation,
scaling, normalization)` is the sole deployment decoder. It inverts the affine
time map; checks shape, finiteness, monotonicity, multiplicity, domain and requested
coverage; discards the four structural coefficient rows; and removes surplus
right-end knot repetitions together with the corresponding inactive coefficient
rows. This trimming is explicitly reported. No running-maximum repair or sorting
is performed (a deliberate strict alternative to the paper's prediction repair).
Interior inversions and unsupported domains are errors.

Only floating-point boundary snaps within a recorded float32 roundoff allowance
are permitted, and their count is returned. Evaluation never extrapolates or
redistributes the requested timestamps. The caller supplies the original grid;
there is no decoder `linspace`. Normalization is inverted to absolute joint
radians and continuous gripper fractions. Out-of-range grippers are reported,
passed through to the existing controller, and never thresholded. This differs
from the paper's binary gripper embodiments and matches audited RoboTwin semantics.

Use `fit_knot_time_scaling(training_cache_paths)` once to establish a shared
training-only affine map. It requires exactly every episode in stage-1's fixed
training manifest, checks membership/hashes, rejects validation/duplicates/failed
fits, and records its identity, bounds and normalization identity. It examines
integer observation windows without refitting. Save its returned JSON and reuse
for validation/deployment. No trained time statistics can be established from
the retained single synthetic episode. Local inspections explicitly use
diagnostic-only offset=0, scale=100 source steps; do not use this silently for
training. This function does not batch-convert or schedule any work.

## Deterministic camera mapping and VAE audit

The recorded pilot selection is **rows 4,8,12,16,20,24,28,32**: one knot after
the left supported boundary, then stride four, giving eight future images.
The paper verifies stepping in knot index; it does not supply a precise offset
for this MOTUS adaptation. Offset=1/stride=4 are therefore explicit pilot choices,
not claimed recovered SplineWAM implementation details. They are stored per sample.

An external timeline must provide `action_times` (one per original command row),
`camera_times` (one per video frame), `units: "seconds"`, `status: "verified"`,
and the stage-1 `action_sha256`. Knot source coordinates interpolate on the
action clock, then choose the nearest camera timestamp, resolving ties toward
the earlier index. Each record stores requested knot row/global index, source
coordinate, target time, actual camera time, frame index and rounding error.
Current image timing is recorded separately from the observation time.
Support-only or repeated tail requests are explicitly masked, as are duplicate
camera selections (including the current image) and camera-clock overruns.
The raw converter writes one image per row when all cameras decode successfully;
the visualization checks video length against the timeline and reuses MOTUS's
existing `load_video_frames`/`get_video_frame_count` helpers (decord RGB).
It does not derive physical time from the encoded 30 FPS setting.

`models/motus.py:798–805` prepends the current image and encodes the full clip.
`bak/wan/modules/vae2_2.py:784–810` processes one current frame, then groups of
four future frames with causal feature caches. Hence eight future frames yield
three latents, not nine. A raw frame invalidates its own group and can contaminate
later latents through those caches. Inspection emits conservative **prefix-based**
latent validity, a proposed latent loss mask (current=False) and key mask. These
are conservative diagnostics, not a claim of exact VAE receptive-field tracing.
Counts not divisible by four are rejected to avoid silently dropped frames.

The existing MOTUS loss (`models/motus.py:889–892`) zeros the current latent but
has no per-sample video padding-mask argument; these proposed masks are consumed
by neither its loss nor attention. `unchanged_motus_training_compatible` describes
**video padding compatibility only**, and is false if any raw frame is invalid.
Padded samples are explicitly blocked as prospective targets for the unchanged
video path; no training integration was attempted. A real masked VAE/loss/attention
integration remains future work and must be checked with the actual VAE weights.

## Available validation and visualization

All **16 lightweight tests pass**, including all seven stage-1 checks, reference
packing equivalence, serialized float32 packet-to-deployment-decoder round trips,
negative support, both episode boundaries, sparse/short tails, explicit failures,
whole-grid decoding, training-only time statistics and video/latent flags.
A simulator-free replay-hook test checks unchanged step counts, original velocity
feedforward and original gripper eps. It is not simulator replay.

An additional sweep checked all 120 observation indices in stage-1's synthetic
episode, comparing 101 points across each whole common domain. Maximum decoded
local/global difference: **1.7186008506864248e-6**; effective coefficients: 4–32.

Retained artifacts in `outputs/spline-stage2/`:

* `verified-beginning.{npz,json,actions.png,frames.png}`: observation 0, valid [0,98],
  32 effective coefficients; local/global error **7.111789521729861e-7**.
* `verified-interior.{npz,json,actions.png,frames.png}`: observation 37.5, valid [37,119],
  26 effective coefficients; error **7.118365166203233e-7**.
* `verified-tail.{npz,json,actions.png,frames.png}`: observation 119, valid [105,119],
  four effective coefficients; error **2.5937535008679902e-8**.

Plots were opened and inspected. The beginning sample has no video repeats; the
interior and tail have invalid video positions. Their camera panels explicitly
show missing images, rather than fabricated frames. These are **synthetic
diagnostic samples**, not actual training samples. Real episode/current/future
image inspection is pending source data and a verified clock.

From the Motus root (choose a fresh output prefix):

```bash
python data/robotwin2/visualize_episode_spline.py outputs/spline-smoke/verified.npz --observation-time 0 --diagnostic-only --output-prefix outputs/spline-stage2/reproduced
python -m unittest discover -s data/robotwin2 -p test_adaptive_spline.py -v
```

## Exact simulator-host replay command (pending; no jobs submitted)

RoboTwin's policy `take_action` performs TOPP retiming, so replaying recorded
rows through that method would not preserve the demonstration clock. The replay
entry instead reuses collection's **setup_demo → load_tran_data → set_path_lst →
play_once** route with the original `seed.txt`, `_traj_data/episodeN.pkl`, config
and assets. Set the three variables to the actual existing episode artifacts:

```bash
cd WAM-RoboTwin
export RECORDING_DIR=/absolute/path/to/original/demo_clean/place_can_basket/aloha_agilex
export SPLINE_CACHE=/absolute/path/to/Motus/outputs/place_can_basket/episode.npz
export EPISODE_INDEX=0
python ../Motus/data/robotwin2/replay_episode_spline.py --robotwin-root "$PWD" --recording-dir "$RECORDING_DIR" --cache "$SPLINE_CACHE" --task-config demo_clean --episode-index "$EPISODE_INDEX" --diagnostic-only --output ../Motus/outputs/place_can_basket/paired-replay.json
```

These variables identify required inputs, not verified cluster paths. Use the
**original** collection config; do not substitute `demo_clean` if it differs.
Diagnostic scaling is explicitly permitted for a ground-truth replay; it does
not make the normalization suitable for training. Once shared training scaling
exists, replace `--diagnostic-only` with `--knot-scaling /absolute/path/time.json`
(retain the diagnostic flag if the action bounds are still provisional).

The baseline intercepts picture call sites and scene steps without writing to
the recording. It must reproduce every source command row within 2e-6 with the
same count. It rejects incomplete coverage or conflicting commands at duplicate
timestamps. Identical duplicated boundary rows use the last row when inverting
the clock; this rule is recorded. A second fresh reset must match all initial
robot/actor poses before execution. Commands are decoded through
`decode_episode_grid` → **`decode_local`** at the original scene-step grid;
existing position setters, velocity feedforward, gripper eps and stepping remain
in use. The clock uses the live scene's
[`get_timestep()`](https://sapien.ucsd.edu/docs/latest/apidoc/sapien.core.html),
not video FPS or an assumed config value. This is a ground-truth representation comparison, not deployment-policy
evaluation. The script rejects event-schedule or duration divergence.

The JSON/NPZ reports task outcomes, measured joint differences, Cartesian
end-effector position RMSE, executed command differences, gripper excursions and
midpoint-crossing times (diagnostic only; commands are not thresholded), and
decoder snaps. `replay_passed` additionally requires both task outcomes to succeed.
A source-bound `.timeline.json` records the reconstructed original clock after
matching the source rows. This assumes the supplied saved paths/config/assets
are those of collection; missing original metadata prevents that claim.

Then inspect a real prospective sample using the preprocessed three-camera video:

```bash
cd ../Motus
python data/robotwin2/visualize_episode_spline.py "$SPLINE_CACHE" --observation-time 0 --diagnostic-only --timeline outputs/place_can_basket/paired-replay.timeline.json --video /absolute/path/to/corresponding/videos/0.mp4 --output-prefix outputs/place_can_basket/sample-0
```

PT and raw HDF5 inputs bind to the same source action values via their cache
identity/hash, but the original saved seed/path and config are still needed.
Downloaded HDF5/PT alone cannot recover an exact controller timeline or scene
reset. Without those inputs replay fails explicitly; no guessed 30 Hz fallback
is provided. Original observation images are read only, never copied back.

**Pending:** real demonstration/image availability, original saved paths/seeds
and controller config, physical camera-clock validation, SAPIEN replay, trained
shared time statistics, and numerical VAE mask testing with weights. No model
training, video-loss integration, deployment changes or cluster submissions
were performed.
