#!/usr/bin/env bash
# Run ONLY inside an already approved GPU allocation. This never submits a job.
set -euo pipefail
: "${SLURM_JOB_ID:?Use an approved existing Slurm GPU allocation}"
: "${SLURMD_NODENAME:?Run via srun on its compute node, never the login node}"
variant="${1:?Supply A or C}"
[[ "$variant" == A || "$variant" == C ]] || { echo 'Expected A or C' >&2; exit 2; }
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Install robotwin-spline into this environment before launching the allocation;
# see RESEARCH_TRAINING.md. No sibling checkout or runtime path injection needed.
python -c 'import robotwin_spline; from data.robotwin2.adaptive_spline import decode_local'
out="${MOTUS_DIAGNOSTIC_OUTPUT:?Set a fresh output directory for this diagnostic}"
config="experiments/ac_spline_pilot/configs/robotwin_research_${variant}.yaml"
python -m experiments.ac_spline_pilot.trainer --config "$config" --checkpoint_dir "$out/train" --report_to none
checkpoint="$out/train/robotwin_research_${variant}/tiny/checkpoint_step_100"
test -f "$checkpoint/research_adapter.pt"
# A new Python process reconstructs the frozen base and reloads only the saved
# adapters/projections, checking all configuration/normalization identities.
python -m experiments.ac_spline_pilot.trainer --config "$config" --checkpoint_dir "$out/reload" \
    --diagnostic-eval-only "$checkpoint" --report_to none
