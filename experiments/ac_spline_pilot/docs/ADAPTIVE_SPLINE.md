# Stage 1: episode conversion and inspection

Stage 2's local-target/decoder audit and pending replay commands are documented
in [LOCAL_SPLINE.md](LOCAL_SPLINE.md). Stage-1 results below remain unchanged.

This stage adds an offline sidecar only. Training, video supervision, deployment,
simulator execution and original episode/image files are unchanged.

## Repository audit (2026-11-12)

Workspace resolved to `C:/Users/Kevin/Personal/Research/Robotics_Research`.
Its three children `Motus`, `WAM-RoboTwin`, and `bspline-policy` are separate Git
repositories. Motus started clean at `88f6fe30c62af033874b0a2e48ad1cbdaf12cfdf`;
bspline-policy started clean at `61ed5f42fced971d50a89b46417493790876ccd1` and was
read only. WAM-RoboTwin already had a changed XPolicyLab submodule pointer
(`b1e2d3a...` to checked-out `c07a096...`), with a clean submodule worktree. This
existing change is preserved. WAM-RoboTwin's origin is DSL-Lab/WAM-RoboTwin;
it has not been identified as the official SplineWAM implementation.

No applicable AGENTS.md was found in the workspace or its ancestors or in Motus
or bspline-policy. The XPolicyLab AGENTS.md was read; no XPolicyLab files changed.
The workspace research plan was read.

## Verified action contract

* `robotwin_data_convert/robotwin_converter.py:386` reads
  `joint_action/vector`, requires 14 columns, casts to float32 and saves the
  tensor as `qpos/<id>.pt`. Despite this filename these are commands.
* `robotwin_agilex_dataset.py:301` uses that tensor for state and action targets.
  Its normalization calls are commented out. Generic `data/utils/norm.py`
  helpers do not establish an active RoboTwin normalization.
* `WAM-RoboTwin/envs/_base_task.py:484` concatenates left joint targets plus
  left gripper, then right joint targets plus right gripper.
  `envs/robot/robot.py:494` reads **drive targets**, whereas the separate
  `get_*_real_jointState` methods read measured qpos.
* `env_cfg/robot/_robot_info.json` registers aloha_agilex with two 6-joint arms
  and one gripper per arm. Channels 0..5 and 7..12 are absolute revolute joint
  positions in radians, in each embodiment's configured `arm_joints_name` order;
  indices 6 and 13 are continuous gripper fractions. `set_gripper` clips to
  [0,1], with 0 closed and 1 open (`_base_task.py:715,725`). There is no binary
  threshold in this converter. The local embodiment assets are missing, so
  anatomical joint names and asset-level verification remain unresolved. The
  ordinal names in plots describe channel order, not guessed joint anatomy.
* `configs/robotwin.yaml` specifies 14 actions, downsampling stride 3, eight
  future video frames and ratio 2: 16 future action targets. The sampling code
  uses `condition_index + 3*(i+1)`, with tail clamping. The offline spline fits
  all original recorded rows, so these training grids can later be decoded
  without changing their coordinate system. It does not fit model windows.
* Video conversion defaults to 30 FPS; this is an encoding setting, not verified
  controller timing. Installed simulation defaults to 1/250 s and collection
  configs specify 15 steps between saves, but movement boundaries can add
  recordings. Neither these defaults nor misleading loader comments prove
  a constant episode interval. Sidecars use source-row indices, retain the
  one-to-one recorded image-row association, and mark physical timestamps
  unknown unless `--seconds-per-step` supplies an independently verified value.
  No physical timestamp array or controller execution grid was available.

Installed task identifier: **`place_can_basket`**, verified by the module and
class in `WAM-RoboTwin/envs/place_can_basket.py` and its instruction JSON.

## Method and artifact contract

The reference at
`bspline-policy/bspline_policy/bspline_policy/common/bspline_action.py:26`
uses `generate_knots(t,data,s=1e-12)` and `make_lsq_spline`, testing maximum
elementwise residual. This implementation uses the same rule and explicit QR
solver. SciPy 1.18.0 source inspection verified that FITPACK insertion selects
the interval with largest summed squared residual, then an interior data site
near that interval's midpoint; it can batch insertions. This is **not** literal
insertion at the single worst sample. At interpolation capacity SciPy uses
not-a-knot interior locations, still with repeated endpoint knots. The generator
is finite, starts with four copies of each endpoint, and never exceeds T
coefficients. Endpoints are not constrained to the observed values. Public
contracts: [generate_knots](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.generate_knots.html)
and [make_lsq_spline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_lsq_spline.html).

Differences from the reference wrapper: threshold accepts `<=`; cap/exhaustion
is a saved failure and exit code 2, never a silently accepted fallback.
`--max-control-points` is optional; a batched generator step that exceeds the
cap is rejected and the last admissible curve saved. A valid cap may therefore
stop before exactly reaching the cap. The generator smoothing stopping value
is independent of the maximum residual acceptance test; settings are saved.

No verified active normalization exists. Before reading values, the converter
sorts episode paths and fixes a seed-0 permutation; ceil(10%) are validation,
the rest training (50 gives 45/5). Streaming bounds use only training episodes,
and are shared across conversions using `--normalization <saved.json>`.
Statistics include full training source identities/hashes and the validation
manifest. Bounds map to [-1,1] without clipping. Constant training channels map
to zero and decode to the constant; a different validation value is rejected
explicitly because it cannot be represented. With one available episode,
`--diagnostic-only` is mandatory and the normalization remains diagnostic-only
when reused. Freeze/reuse this manifest rather than rescanning a changing
directory. For each conversion pass only one task, embodiment and data mode.

The NPZ stores knots, coefficients, degree, original command values, source row
indices and JSON metadata. Metadata contains normalization and its identity,
task/channel semantics, source action hash/path/size/mtime, converter code hash,
fitting settings, per-channel sampled errors, history, and analytic decoded
curve extrema/overshoot. The cache identity depends on this metadata; there is
no automatic cache reuse or image duplication. Outputs refuse overwrite.
HDF5 images are referenced through the source episode and row indices;
preprocessed PT inputs reference the sibling videos directory.

## Run on the cluster (CPU; no job submission)

Use an existing environment with NumPy, SciPy >=1.15 and Matplotlib; PT input
also needs PyTorch, raw HDF5 input needs h5py. No GPU, VLM weights, simulator or
whole-dataset download is required. From the **Motus repository root**:

```bash
python data/robotwin2/convert_episode_spline.py --episode-dir /share/dataset/preprocess/robotwin2/clean/place_can_basket/qpos --output outputs/place_can_basket/episode.npz --plot
```

This is the expected path from the installed MOTUS config, **not a verified
cluster data location**. It selects one training episode after fixing the split;
prints its output path, tolerance result and parameter count; and creates
`.arms.png`, `.grippers.png`, `.ranges.png`, `.ranges.json` and
`.normalization.json` next to the sidecar. Supply `--episode /absolute/path.pt`
to choose an episode. If there is only one episode add `--diagnostic-only`.
Choose a fresh output filename for each run. No Slurm job was submitted.

For raw data, replace only `--episode-dir` with the existing
`<raw-root>/place_can_basket/aloha-agilex_clean_50/data` directory. This reads
only command vectors for statistics and fitting, leaving images untouched.
Physical timing should remain unknown until verified from that recording.

Standalone visualization:

```bash
python data/robotwin2/visualize_episode_spline.py outputs/place_can_basket/episode.npz
```

## Actual local validation and outstanding evidence

No HDF5 demonstrations or converted action tensors were found in local data
locations, including ignored files. `WAM-RoboTwin/data` contains only a utility
script; the configured `/share/dataset/preprocess/robotwin2` mount is absent.
Restricted tooling-cache directories could not be searched. No claim is made
that an available cluster demonstration has been located or converted.

Seven synthetic numerical/CLI tests passed with Python 3.13, SciPy 1.18.0.
They cover independent least-squares agreement/rank, cubic initialization,
switch refinement, cap failure, constant-channel rejection, no clipping,
exact overshoot extrema, split isolation, cache decoding, normalization reuse,
overwrite rejection, persisted failure status/exit code, and single-episode opt-in. Run:

```bash
python -m unittest discover -s data/robotwin2 -p test_adaptive_spline.py -v
```

The retained synthetic smoke fixture at
`outputs/spline-smoke/episodes/synthetic_commands.npy` has 120 rows and 14
channels, including a constant joint and an abrupt gripper switch. Its
diagnostic-only fit used 34 coefficients / 38 knots and achieved maximum
normalized sampled error **0.009547563666154613**. The three plots were opened
and inspected. Its discontinuous gripper exhibits decoded extrema about
**-0.108332 and 1.10784** between samples despite meeting tolerance. This is
evidence for inspecting decoded curves and executing replay before training,
not a real demonstration result or a manipulation success claim.

Final local artifacts: `outputs/spline-smoke/verified.npz` and
`verified.{arms,grippers,ranges}.png`, plus range JSON and normalization JSON.
To reproduce that conversion using the retained fixture, choose a fresh path:

```bash
python data/robotwin2/convert_episode_spline.py --episode-dir outputs/spline-smoke/episodes --diagnostic-only --output outputs/spline-smoke/reproduced.npz --plot
```

Unresolved: actual cluster data availability/location, episode/embodiment asset
identity and joint names, physical timestamps/controller sampling, and decoded
trajectory execution. The PT/HDF5 reading branches follow the audited source
format but have not been exercised against real recordings locally. This stage
stops at conversion/inspection; no training, video supervision or deployment
changes are included.
