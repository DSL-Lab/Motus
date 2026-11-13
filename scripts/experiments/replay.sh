#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
: "${SLURM_JOB_ID:?Replay/rendering must run in Slurm}"
cd "$MOTUS_ROOT"
motus_python tests/allocated_replay.py "$@"
