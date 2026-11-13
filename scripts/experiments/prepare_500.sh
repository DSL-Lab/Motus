#!/usr/bin/env bash
# Stage the pinned public archive on the login node; convert/fit/encode in Slurm.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
revision=3dc3b798668feb99ac61cc9086d84cbcc3d79186
directory="$ROOT/.runtime/datasets/place_can_basket-500-$revision"
archive="$directory/aloha-agilex_randomized_500.zip"
output="${MOTUS_500_CACHE:-$ROOT/outputs/place_can_basket-500}"
mode="${1:-all}"
if [[ "$mode" == --help ]]; then
    cat <<'EOF'
Usage: bash scripts/experiments/prepare_500.sh [all|download|submit|check]
Default: download the pinned 9.86 GB Place Can Basket archive, then submit one
30-minute Slurm preparation job (1 L40S, 4 CPUs, 64 GB). No training is submitted.
download: network staging only; submit: use an already staged archive.
check: inspect local preparation state without downloading or submitting.
MOTUS_500_CACHE overrides the final cache directory.
EOF
    exit 0
fi
if [[ "$mode" == --allocated ]]; then
    : "${SLURM_JOB_ID:?Requires Slurm}"
    source "$ROOT/scripts/experiments/environment.sh"
    motus_python -m motus_actions.prepare_500 --archive "$archive" \
        --work-dir "$directory" --output "$output"
    exit 0
fi
[[ "$mode" == all || "$mode" == download || "$mode" == submit || "$mode" == check ]] || { echo 'Use --help' >&2; exit 2; }
if [[ "$mode" == check ]]; then
    python3 - "$archive" "$output" <<'PY'
import json
from pathlib import Path
import sys
archive, output = map(Path, sys.argv[1:])
print('Archive:', 'staged (hash checked in Slurm)' if archive.is_file() and archive.stat().st_size == 9855736739 else 'not staged')
path = output/'manifest.json'
if not path.is_file():
    raise SystemExit(f'Cache: not prepared: {path}')
manifest = json.loads(path.read_text())
if len(manifest.get('episodes', [])) != 500 or manifest.get('conditioning') != 'robotwin_scene_prefix_v1':
    raise SystemExit('Cache: incomplete or incompatible')
print('Cache: prepared:', path)
print('Next: complete repair/replay and controller/test-plan validation before training.')
PY
    exit 0
fi
mkdir -p "$directory"
exec 8>"$directory/.download.lock"
flock -n 8 || { echo 'Dataset staging/submission is already active' >&2; exit 2; }
if [[ "$mode" == all || "$mode" == download ]]; then
    if [[ ! -f "$archive" ]]; then
        df -h "$directory"
        url="https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/$revision/dataset/place_can_basket/aloha-agilex_randomized_500.zip?download=true"
        curl --fail --location --continue-at - --retry 2 --connect-timeout 30 \
            --max-time 1800 --output "$archive.part" "$url"
        python3 - "$archive.part" <<'PY'
from pathlib import Path
import sys
if Path(sys.argv[1]).stat().st_size != 9855736739:
    raise SystemExit('Download size mismatch; partial file retained for inspection/resume')
PY
        mv -- "$archive.part" "$archive"
    fi
    echo "Archive staged: $archive (SHA-256 verification occurs inside Slurm)"
    if [[ "$mode" == download ]]; then exit 0; fi
fi
[[ -f "$archive" ]] || { echo 'Run download first' >&2; exit 2; }
if [[ -e "$output/manifest.json" ]]; then
    echo "Cache already exists: $output; run check, do not submit duplicate preparation"; exit 0
fi
existing="$(squeue --noheader --user "$(id -un)" --name motus-prep-500 --format '%i')"
[[ -z "$existing" ]] || { echo "Preparation already queued/running: $existing"; exit 0; }
cd "$ROOT"
mkdir -p logs
job="$(sbatch --parsable --export=ALL --time=00:30:00 --job-name=motus-prep-500 \
    scripts/slurm/slurm_single_node.sh scripts/experiments/prepare_500.sh --allocated)"
printf '%s\tsubmitted\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job" >> "$directory/jobs.tsv"
printf 'Preparation job: %s\nLog: %s/logs/ac-%s.log\n' "$job" "$ROOT" "${job%%;*}"
