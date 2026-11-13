# Place Can Basket pilot — prerequisite audit

**Status: blocked before deployment/training; no jobs submitted or running.**
Task identifier `place_can_basket` is verified in the installed module/class and
instruction file. Scope remains original checkpoint, dense A, spline C, seed 0.
Audit artifacts were captured 2026-11-13 UTC (workspace session date 2026-11-12).

Previous-stage evidence was inspected in `data/robotwin2/ADAPTIVE_SPLINE.md`,
`data/robotwin2/LOCAL_SPLINE.md`, `RESEARCH_TRAINING.md` and their retained outputs.
Conversion/local extraction and checkpoint reload results are **synthetic/CPU
component evidence**. Real conversion, simulator replay, tiny-data learning and
full-model fresh reload were never executed. The pilot readiness gate remains.

| Requested measurement | Original | A | C |
|---|---|---|---|
| Completed deployment trials; successes | 0; unmeasured | 0; unmeasured | 0; unmeasured |
| Failure reasons | Runtime/data/controller prerequisites below; no episode failures observed | Same | Same |
| Warm observation-to-command median/p95; decoder cost | Unmeasured | Unmeasured | Unmeasured |
| Invalid prediction frequency | Unmeasured | Unmeasured | Unmeasured |
| Actual full-model action token count | Unmeasured | Unmeasured | Unmeasured |
| Policy calls, simulated duration, wall time, peak memory | Unmeasured | Unmeasured | Unmeasured |
| Training time; pilot checkpoints | N/A | Not trained; none | Not trained; none |

A/C paired successes: **0 observed pairs, no deployed trials**; paired-success
timing is unavailable. There is no success/efficiency comparison, latency gain,
or policy-call reduction to claim. Unexecuted trials are not scored as failures.

Completed local work:

* Re-ran 16 spline/replay-hook regressions and 8 training component checks: pass.
  New CPU fixture artifacts: `outputs/stage3-components/593f47b8e4b64635824e8a6fbae36948/`
  and `outputs/stage3-components/ab1e2535e9634c3696bc27ce10eb3af2/`.
  Fresh-process **component** reload maximum difference is 0.0.
* Fixed original `inference/robotwin/Motus/deploy_policy.py`: checkpoint loading
  now requires all tensors; three-view geometry follows actual head resolution;
  already-sized uint8 RGB now receives the same [0,1] conversion as resized RGB.
  Raw 14D commanded-state conditioning and absolute command units remain.
  Five direct-method tests pass. Model loading and OpenCV resizing are stubbed
  in these tests; they do not validate weights, interpolation, VAE or simulation.
* Added CPU preparation utility and frozen source/configuration checksums.
  Reusing the preparation output refuses overwrite. Submission is disabled;
  there are no job IDs, queued jobs or automatic retries to duplicate.

Unresolved prerequisites, in execution order:

1. **Runtime/assets:** Python 3.13 has CPU PyTorch 2.12.1, SciPy 1.18.0 and
   Accelerate 1.14.0; no CUDA devices, SAPIEN, decord, Transformers, OmegaConf
   or OpenCV. The workspace `.venv` uses Python 3.14 and lacks those scientific
   dependencies. Use `py -3.13` for the recorded CPU checks. No Slurm commands
   or verified cluster environment were supplied; SSH resolves to a deny wrapper.
   No real weights/demonstrations were found in accessible local model/data
   paths; a restricted inference cache was not searchable. The configured
   `/share` data mount and aloha-agilex embodiment assets are absent.
2. **Original deployment:** the bundled `eval.sh` expects
   `script/eval_policy.py`; installed WAM-RoboTwin routes through XPolicyLab in
   `scripts/eval_policy.sh`. The original bridge is not integrated there.
   Select the compatible RoboTwin checkpoint, record prior task exposure, and
   execute strict loading before interpreting prediction quality. No checkpoint
   provenance or actual camera/T5/VLM/state parity measurement exists yet.
   Source inspection shows the converter's language prefix matches the wrapper.
3. **Controller:** `envs/_base_task.py:1486` runs TOPP per qpos target with a
   variable number of scene steps, and silently falls back to 50 steps on planner
   failure. Its default scene timestep is 1/250 s, configurable independently of
   TOPP's hard-coded 1/250 argument. Sixteen commands are not sixteen simulator
   control steps. Establish the actual clock, log planner failures, and verify
   one fixed executed-step contract shared by A/C before rollout comparison.
   No controller frequency or step budget is claimed verified by execution.
4. **Real-data/learning gates:** locate 50 complete task demonstrations, freeze
   45/5 membership before statistics, convert and check every local prefix/video
   target, execute matched replay, then run the existing varied-noise tiny A/C
   diagnostic and full-model fresh reload. Resolve insufficient 36-row coverage
   and VAE masking failures without silently changing the execution horizon.
   `train/train.py` still rejects pilot phase/>100 updates; `load_contract` still
   permits only four tiny training windows. No bypass was introduced.
5. **Research deployment/metrics:** adapter bundle loading, shared spline decoder
   integration, fixed-step execution, warmed CUDA-synchronized latency/memory,
   scene-step/policy-call accounting and episode failure logging still need actual
   end-to-end implementation/validation. Measure latency from observation ingress
   through command availability, include decoding, exclude warmup explicitly,
   and retain all episodes plus timing for shared successful scene pairs only.

Candidate configuration, **not launch-ready**:

* Proposed 50 demonstrations, 45/5 split; training seed 0; LoRA 16/32/0;
  lr 1e-4; microbatch 1 × accumulation 16 on one process; 1,000 optimizer updates.
  Full-model memory feasibility is unknown. Same AdamW scope/betas (0.9,0.95),
  weight decay .01 and clipping .5 in A/C. Candidates explicitly retain the
  original scheduler: warmup 200, cycle 5,000,000, f_max .99, f_min .4; this is
  not the tiny diagnostic's 10/100 schedule or a convergence result.
* A retains dense 16×14 actions and original uniform video sampling. C uses
  36×15 parameters and knot rows 4,8,…,32. Both retain raw 14D proprioception.
  Action input boundary weights are 1024×14 versus 1024×15; output weights
  14×1024 versus 15×1024, with corresponding biases. Complete action input MLP
  and output projections are reset and fully trained in both; compatible hidden
  layers use name-seeded matched initialization. Other compatible weights and
  LoRA scope match. Actual checkpoint compatibility remains unknown.
* CPU/source contract: 16/36 predicted action rows; **21/41 action-branch tokens**
  including one state and four registers. These are not full-model runtime token
  measurements. Video/understanding tokens are additional.
* Twenty shared development scenes are proposed. Seeds remain unassigned until
  reset semantics and training-scene overlap can be checked. No final evaluation
  scenes are consumed or mixed into development. No extra tasks/seeds, physical
  speed scaling or adaptive replanning were added.

Commands actually executed from `Motus` (all listed tests passed):

```powershell
py -3.13 train/test_research.py -v
py -3.13 -m unittest discover -s data/robotwin2 -p test_adaptive_spline.py -v
py -3.13 -m unittest discover -s tests -p test_robotwin_deployment_contract.py -v
py -3.13 scripts/prepare_single_task_pilot.py --task place_can_basket --representation both --seed 0 --updates 1000 --output outputs/stage4-preparation-20261112
git diff --check
```

Artifacts: `outputs/stage4-preparation-20261112/{readiness.json,pending-jobs.json,
source-identities.json,validation.json}` and `source/{Motus,WAM-RoboTwin}/`.
Candidate training commands/config paths and exclusive future output directories
are recorded in `pending-jobs.json`. They were **not executed** and the current
trainer deliberately refuses them. Existing `scripts/slurm/run_tiny_research.sh`
is retained for approved allocated-node prerequisite diagnostics; the existing
8-GPU/1.5-TB Slurm template has placeholder settings and is not reused as verified
resource evidence. Frozen snapshots must be regenerated after necessary fixes
and after real paths/configuration are established, before any submission.

**Compute:** no explicit budget authorization is present in the available
session. GPU-hours used by this work: 0; total pilot GPU-hours **undetermined**,
not zero. Exact sbatch jobs/budget cannot be honestly prepared without a verified
cluster account/partition, GPU type/memory, environment, asset paths and timings.
Those details were requested; no budget or submission approval has been assumed.
Once established, prepare one capped budget covering original deployment,
prerequisite diagnostics, A/C training and development evaluation, and ask once
before submission. Job IDs must be durably recorded; ambiguous submissions must
be reconciled through Slurm rather than retried.

**Next diagnostic supported by evidence:** on the verified simulator host,
strict-load the original compatible checkpoint, compare its actual current
observation conditioning with the training loader, and trace one ground-truth
command prefix through `take_action` with scene-step and TOPP-status logging.
The demonstrated entry-point mismatch and variable controller timing must be
resolved before an expensive training result can answer the research question.
