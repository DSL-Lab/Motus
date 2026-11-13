#!/usr/bin/env bash
# Shared A/C entry point; uses the existing trainer, not a parallel training loop.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/experiments/environment.sh"
: "${SLURM_JOB_ID:?Submit through scripts/slurm/slurm_single_node.sh}"
cd "$MOTUS_ROOT"
representation="${1:?Usage: train.sh A|C SEED OUTPUT [RESUME]}"
seed="${2:?training seed required}"
output="${3:?unique persistent output directory required}"
resume="${4:-}"
[[ "$representation" == A || "$representation" == C ]]
[[ "$seed" =~ ^[0-9]+$ ]]
# These are scientific validation gates, not an approval prompt.
: "${MOTUS_REPLAY_REPORT:?Original and spline recorded-scene replay must pass first}"
: "${MOTUS_REPAIR_REPORT:?Conditioning, batch, decoder, generation and fresh resume gates must pass first}"
python3 - "$MOTUS_REPLAY_REPORT" "$MOTUS_REPAIR_REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
for gate in ('original_replay', 'spline_replay', 'matched_controller'):
    if report.get(gate) != 'passed':
        raise SystemExit(f'Unpassed training gate: {gate}')
repair = json.load(open(sys.argv[2]))
for gate in ('conditioning', 'accumulation', 'ground_truth_decoder', 'tiny_generation', 'fresh_resume'):
    if repair.get(gate) != 'passed':
        raise SystemExit(f'Unpassed repair gate: {gate}')
PY
mkdir -p "$output"
exec 9>"$output/.train.lock"
flock -n 9 || { echo 'Training already active for this output' >&2; exit 1; }
if [[ -z "$resume" && -e "$output/launch.txt" ]]; then
    echo 'Existing run; provide its resume checkpoint or use a new output' >&2
    exit 1
fi
printf '%q ' "$0" "$@" >> "$output/launch.txt"
printf '\n' >> "$output/launch.txt"
args=(--config "${MOTUS_TRAIN_CONFIG:-$MOTUS_ROOT/configs/robotwin_ac.yaml}" --representation "$representation"
      --seed "$seed" --checkpoint_dir "$output" --run_name "${representation}-seed${seed}" --report_to none)
if [[ -n "$resume" ]]; then args+=(--resume "$resume"); fi
if [[ -n "${MOTUS_EPISODE_MANIFEST:-}" ]]; then args+=(--episode_manifest "$MOTUS_EPISODE_MANIFEST"); fi
motus_python "$MOTUS_ROOT/train/train.py" "${args[@]}"
