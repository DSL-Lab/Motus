#!/usr/bin/env bash
# One task, matched rank-16 LoRA, using the existing A/C trainer.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"

usage() {
    cat <<'EOF'
Usage: bash scripts/experiments/finetune_one_task.sh [TASK] [A|C|both] [--check]

Defaults: place_can_basket, both, rank 16, 50 episodes (45 train / 5 validation),
seed 0, 1000 optimizer updates, effective batch 16. A=dense; C=spline.
Submits to Slurm: one L40S, 4 CPUs, 64 GB, up to 4 hours per condition.
The two conditions run sequentially (up to 8 GPU-hours for both).
--check checks prerequisites without creating outputs or submitting jobs.

Prepared data defaults to outputs/repair-cache-20261111/manifest.json. Override with
MOTUS_EPISODE_MANIFEST. MOTUS_REPLAY_REPORT and MOTUS_REPAIR_REPORT must identify
the existing validation reports required by scripts/train.sh.
Optional: MOTUS_WALLTIME (at most 04:00:00), MOTUS_SEED (0),
MOTUS_UPDATES (1000), MOTUS_CHECKPOINT (initial weights), MOTUS_OUTPUT_ROOT.
The manifest selects data/split; both conditions use it unchanged. Submission
requires a current user-authorized budget; resource requests do not grant one.
Raw demonstrations and pretrained weights are never overwritten.
EOF
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi

# This branch executes inside the existing Slurm allocation/environment wrapper.
if [[ "${1:-}" == --allocated ]]; then
    shift
    task="$1"; condition="$2"; manifest="$3"; output="$4"; seed="$5"
    resume="${6:-}"; test_plan="${7:-}"
    : "${SLURM_JOB_ID:?Use the submission entry point}"
    source "$ROOT/scripts/experiments/environment.sh"
    if [[ -z "$resume" ]]; then
    motus_python - "$ROOT/configs/robotwin_ac.yaml" "$task" "$manifest" "$output" "$seed" "$test_plan" "${MOTUS_UPDATES:-1000}" "${MOTUS_CHECKPOINT:-}" <<'PY'
from pathlib import Path
import json
import sys
from omegaconf import OmegaConf
base, task, manifest, output, seed, test_plan, updates, checkpoint = sys.argv[1:]
config = OmegaConf.load(base)
config.dataset.task_mode = 'single'
config.dataset.task_name = task
config.dataset.data_mode = 'clean'
config.dataset.episode_manifest = manifest
episodes = json.loads(Path(manifest).read_text())['episodes']
config.dataset.dataset_dir = str(Path(episodes[0]['hdf5_path']).parents[3])
config.training.lora = True
config.training.lora_rank = 16
config.training.seed = int(seed)
config.training.batch_size = 1
config.training.gradient_accumulation_steps = 16
updates = int(updates)
config.training.max_steps = updates
config.training.warmup_steps = max(1, int(updates * .05))
if checkpoint:
    config.finetune.checkpoint_path = checkpoint
config.system.checkpoint_dir = output
if test_plan:
    plan = json.loads(Path(test_plan).read_text())
    config.execution.control_dt_s = plan['control_dt_s']
    config.execution.clock_provenance = plan['clock_provenance']
path = Path(output) / 'experiment.yaml'
path.parent.mkdir(parents=True, exist_ok=False)
OmegaConf.save(config, path)
PY
    fi
    export MOTUS_TRAIN_CONFIG="$output/experiment.yaml"
    export MOTUS_EPISODE_MANIFEST="$manifest"
    exec bash "$ROOT/scripts/train.sh" "$condition" "$seed" "$output" "$resume"
fi

task="${1:-place_can_basket}"
condition="${2:-both}"
check="${3:-}"
[[ "$task" =~ ^[a-z][a-z0-9_]*$ ]] || { echo 'Invalid task name' >&2; exit 2; }
[[ "$condition" == A || "$condition" == C || "$condition" == both ]] || { usage >&2; exit 2; }
[[ -z "$check" || "$check" == --check ]] || { usage >&2; exit 2; }
[[ $# -le 3 ]] || { usage >&2; exit 2; }
manifest="${MOTUS_EPISODE_MANIFEST:-$ROOT/outputs/repair-cache-20261111/manifest.json}"
walltime="${MOTUS_WALLTIME:-04:00:00}"
seed="${MOTUS_SEED:-0}"
updates="${MOTUS_UPDATES:-1000}"
[[ "$updates" =~ ^[1-9][0-9]*$ ]] || { echo 'MOTUS_UPDATES must be positive' >&2; exit 2; }
export MOTUS_UPDATES="$updates"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'MOTUS_SEED must be a nonnegative integer' >&2; exit 2; }

# Cheap metadata checks on the login node; model work stays in Slurm.
python3 - "$task" "$manifest" "${MOTUS_REPLAY_REPORT:-}" "${MOTUS_REPAIR_REPORT:-}" "$walltime" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys
task, source, replay, repair, walltime = sys.argv[1:]
match = re.fullmatch(r'(\d{2}):([0-5]\d):([0-5]\d)', walltime)
if not match or not 0 < sum(int(x)*scale for x,scale in zip(match.groups(), (3600,60,1))) <= 14400:
    raise SystemExit('MOTUS_WALLTIME must be HH:MM:SS, greater than zero and at most 04:00:00')
source = Path(source).resolve()
if not source.is_file():
    raise SystemExit(f'Episode cache is not prepared: {source}\n'
        'Prepare scene-prefixed language for the existing 50-demo cache. See docs/runbook.md.')
manifest = json.loads(source.read_text())
identity = {k:v for k,v in manifest.items() if k != 'identity'}
if manifest.get('identity') != hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest():
    raise SystemExit('Manifest identity mismatch; regenerate the cache')
episodes = manifest['episodes']
ids = {e['id'] for e in episodes}
train, validation = (manifest['split'][s] for s in ('train', 'validation'))
if len(episodes) != 50 or len(ids) != 50 or {e['task_name'] for e in episodes} != {task}:
    raise SystemExit('Expected exactly 50 distinct episodes of the selected task')
if (len(train) != 45 or len(set(train)) != 45 or len(validation) != 5 or len(set(validation)) != 5
        or set(train) & set(validation) or set(train) | set(validation) != ids):
    raise SystemExit('Expected a disjoint 45/5 episode split covering all 50 demos')
if manifest.get('conditioning') != 'robotwin_scene_prefix_v1':
    raise SystemExit('Scene-prefixed language cache is not prepared')
for episode in episodes:
    for key in ('hdf5_path', 'cache_path'):
        if not Path(episode[key]).is_file():
            raise SystemExit(f'Missing {key}: {episode[key]}')
# The dataset checks the shared prompt helper at runtime. Require cached language
# files before allocating a GPU.
if not manifest.get('language_records'):
    raise SystemExit('Missing language cache records; run prepare --language-from-manifest')
for record in manifest['language_records']:
    if not (source.parent/'language'/f"{record['sha256']}.pt").is_file():
        raise SystemExit('Missing cached language embedding')
for path, gates in [(replay, ('original_replay','spline_replay','matched_controller')),
                    (repair, ('conditioning','accumulation','ground_truth_decoder','tiny_generation','fresh_resume'))]:
    if not path or not Path(path).is_file():
        raise SystemExit('Set MOTUS_REPLAY_REPORT and MOTUS_REPAIR_REPORT to completed validation reports')
    report = json.loads(Path(path).read_text())
    for gate in gates:
        if report.get(gate) != 'passed':
            raise SystemExit(f'Validation prerequisite is not passed: {gate}')
print(f'Prepared: {task}; 50 demos = 45 train + 5 validation; LoRA rank 16')
PY

printf 'Conditions: %s; seed: %s; %s updates; one L40S per job; limit %s per condition\n' "$condition" "$seed" "$updates" "$walltime"
if [[ "$check" == --check ]]; then exit 0; fi
cd "$ROOT"
conditions=("$condition")
if [[ "$condition" == both ]]; then conditions=(A C); fi
for representation in "${conditions[@]}"; do
    existing="$(squeue --noheader --user "$(id -un)" --name "motus-$task-r16-$representation" --format '%i')"
    [[ -z "$existing" ]] || { echo "Existing $representation job: $existing; refusing duplicate submission" >&2; exit 2; }
done
mkdir -p logs "${MOTUS_OUTPUT_ROOT:-$ROOT/outputs}"
run="$(mktemp -d "${MOTUS_OUTPUT_ROOT:-$ROOT/outputs}/${task}-50-r16-seed${seed}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
manifest="$(realpath "$manifest")"
export MOTUS_REPLAY_REPORT="$(realpath "$MOTUS_REPLAY_REPORT")"
export MOTUS_REPAIR_REPORT="$(realpath "$MOTUS_REPAIR_REPORT")"
cp -- "$manifest" "$run/dataset_manifest.json"
cp -- "$MOTUS_REPLAY_REPORT" "$run/replay_report.json"
cp -- "$MOTUS_REPAIR_REPORT" "$run/repair_report.json"
export MOTUS_REPLAY_REPORT="$run/replay_report.json" MOTUS_REPAIR_REPORT="$run/repair_report.json"
dependency=()
for representation in "${conditions[@]}"; do
    job="$(sbatch --parsable --export=ALL --time="$walltime" \
        --job-name="motus-$task-r16-$representation" "${dependency[@]}" \
        scripts/slurm/slurm_single_node.sh scripts/experiments/finetune_one_task.sh \
        --allocated "$task" "$representation" "$manifest" "$run/$representation" "$seed" "" "${MOTUS_TEST_PLAN:-}")"
    job="${job%%;*}"
    printf '%s\t%s\tsubmitted\t%s\n' "$representation" "$job" "$run/$representation" | tee -a "$run/jobs.tsv"
    dependency=(--dependency="afterok:$job" --kill-on-invalid-dep=yes)
done
printf 'Run records: %s\n' "$run"
