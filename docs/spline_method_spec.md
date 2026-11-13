# Verified method and explicit adaptations

Evidence: local `references/2609.39873v1.pdf` (SplineWAM), sections 3.1–3.4,
Eq. 1–2, Appendix D pp.17–19; `2607.09648v1.pdf` (B-spline Policy);
`2512.13030v2.pdf` (MOTUS). PDF text extracted once to `/tmp` for inspection.
Official https://splinewam.github.io/ checked 2026-11-11: **Code coming soon**.
WAM-RoboTwin is not official SplineWAM. Exact unpublished implementation
choices cannot be represented as verified reproduction.

## Global fit

Cubic degree 3, shared knots for all 14 channels, open/clamped boundaries with
multiplicity four. Source coordinate is integer episode sample index (not
verified seconds). Actions are min/max mapped to [-1,1] using training episodes
only, before fitting. Continuous grippers are fitted jointly. Reference BSP
`common/bspline_action.py:ScipyBSplineCompression.compress` at
`61ed5f42fced971d50a89b46417493790876ccd1` uses SciPy generate_knots(s=1e-12),
then make_lsq_spline, stops on maximum absolute element error <0.01. It defaults
to cubic implicitly and falls back to its final fit even if tolerance is unmet.
SplineWAM instead describes insertion at largest residual and no silent unmet
bound. We implement its documented residual-driven least-squares loop, rejecting
an unmet tolerance. This is a paper adaptation, not a copy of the BSP solver.
Start with endpoints, insert highest-error eligible integer sample from2..N-3
(the admissible set preserves the not-a-knot interpolation limit). At mostN
control points; reject if the bound remains unmet or no candidate remains.
No endpoint value constraints beyond least squares; no extrapolation. Default 0.01 is
SplineWAM's LIBERO tolerance; 0.08 is RoboCasa's, neither is verified for RoboTwin.
0.01 is a conservative initial RoboTwin adaptation pending replay.

## Local window and packing

Paper Appendix D: 36 rows, 15 columns for this 14-D embodiment. 36 knot times,
32 effective control points, 4 trailing control-point padding rows. First three
and last three knots are support; valid interval [u3,u32]. Keep 36 rows and
stride 4 as the concrete published 8-future-frame example. **This is more action
tokens than MOTUS's native 16-row chunk**, not token compression. No latency
benefit is assumed. The paper's budget relation is a choice, not a spline identity.

Observation t -> containing global knot span j -> start=j-3 -> contiguous
36 knots and corresponding control points -> repeat final knots/control points
at episode tail -> subtract t from all knot times. Use the containing span as
specified by SplineWAM, not BSP sampler's next-span assignment (its while loop
assigns observations through each chunk's start knot). Independent tests must
compare local and global curves only on supported times. Padding control points
are supervised, as Appendix D states. Parameter column 0 is absolute local time,
not differences; remaining columns are normalized control values. Fit an affine
per-column map on training local windows only; do not clip validation targets.

## Executable alignment rule (explicit adaptation where unpublished)

```
j = searchsorted(global_knots, observation_index, side='right') - 1
start = clip(j - 3, 0, len(global_knots) - 8)
local = extract(global, start, rows=36); local.knots -= observation_index
selected_rows = 4 + 4 * arange(8)  # 4,8,...32; excludes exterior support
image_indices = floor(observation_index + local.knots[selected_rows] + 0.5)
condition_image = images[observation_index]
```

The paper verifies fixed knot stride, prepended current frame, repeated tail
padding and 8 future frames. It does **not** specify numeric offset or rounding.
The offset 4 (first strictly future knot after the containing span), round-half-up
and rejecting duplicate indices are explicit conservative adaptations. Never
claim these are exact released-code choices. Current image/state stay at t;
training targets use offline ground-truth knots only.

Wan2.2 VAE uses causal temporal convolutions and 4x temporal compression; nine
input frames produce three latents (see `bak/wan/modules/vae2_2.py`). A frame mask
cannot be applied directly to these mixed latents. Initial C excludes windows
with duplicate/out-of-range target indices before VAE encoding; this avoids
inventing a latent padding-mask rule. A uses the same eligible observation set
when the experiment manifest is active, but retains its original uniform target
sampling. This common sampling restriction is an adaptation that must be reported.
No tail padding is sent to the VAE in this experiment.

## Inference

Invert parameter map; evaluate at baseline source offsets 3,6,...48, executing
the same prefix as A. No variable-duration replanning or speed scaling. Validate
finite shape, monotonic knots, valid domain and full prefix coverage. Reject
invalid predictions explicitly; no sorting, extrapolation, silent repair or hold.
Predicted knots receive Appendix D's explicitly counted running maximum before
strict domain/multiplicity/coverage validation. This does not apply to offline fits.
The initial strict-only diagnostic and subsequent guard update are recorded below.
RoboTwin grippers are continuous opening fractions, unlike paper's binary
single-arm commands: do not threshold until actual controller semantics are
established. No cross-call smoothing is assumed.

## Known validation gates

Downloaded task HDF5 files lack recorded seed.txt and verified physical timestamps.
Do not infer seeds or Hz from episode IDs or additional_info/frequency. Original
seeded replay and matched physical controller timing remain blocked until recovered.

## Native controller saturation

`motus_actions.spline.controller_commands` rejects nonfinite commands and arm
values outside limits read from the live robot. It explicitly applies/counts the
native gripper saturation in `WAM-RoboTwin/envs/robot/robot.py:set_gripper` (line628),
which clips opening fractions to[0,1]. Apply equally to A/C. Count each saturated
scalar as a repair; no arm clipping, knot repair, extrapolation or fallback.
The earlier "no repairs" statement applies to spline parameter decoding itself.

## Evidence-driven decoder update (after strict tiny check)

The50-update C diagnostic passed fixed-loss learning but its fresh inference failed
strict knot ordering (record remains invalid). Appendix D explicitly defines a
running maximum. It is now implemented for **predicted knots only** in
`repair_prediction_knots`; each changed scalar is counted before strict support,
multiplicity and coverage checks. Control-point rows never reorder. This supersedes
the earlier temporary strict-only inference choice; offline evaluation/extraction
remains strict. No replay acceptance threshold or original diagnostic result changed.
Degenerate domains, unsupported prefixes and excessive interior multiplicity still
fail; there is no command fallback. Native gripper clipping is counted separately.
