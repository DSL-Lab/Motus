#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
: "${SLURM_JOB_ID:?Conversion must run in Slurm}"
cd "$MOTUS_ROOT"
df -h "$MOTUS_ROOT"
motus_python -m motus_actions.prepare --raw-root "$MOTUS_WORKSPACE/WAM-RoboTwin/data/demo_clean" "$@"
