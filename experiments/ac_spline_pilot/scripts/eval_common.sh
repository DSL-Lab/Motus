#!/usr/bin/env bash
set -euo pipefail

variant=$1
mode=${2:-simulator}
limit=${3:-}
case "$variant" in
    A) pilot="$PWD/outputs/ac_splinewam_A_170/train/robotwin_pilot_A_170/pilot-seed-0" ;;
    C) pilot="$PWD/outputs/ac_splinewam_C_170/train/robotwin_pilot_C_splinewam_170/pilot-seed-0" ;;
    *) echo "Unknown variant $variant" >&2; exit 2 ;;
esac
checkpoint=${4:-}
if [[ -z "$checkpoint" ]]; then
    if [[ ! -f "$pilot/final_checkpoint.txt" ]]; then
        echo "Missing final checkpoint marker $pilot/final_checkpoint.txt. Pass a checkpoint directory as argument 3 to the Slurm script." >&2
        exit 2
    fi
    checkpoint="$pilot/$(<"$pilot/final_checkpoint.txt")"
fi
[[ -f "$checkpoint/research_adapter.pt" && -f "$checkpoint/research_config.json" ]] || {
    echo "Incomplete checkpoint $checkpoint" >&2; exit 2;
}

output="$PWD/outputs/evaluation/$variant/${SLURM_JOB_ID:?Submit using sbatch}-$mode"
cases=${MOTUS_EVAL_CASES:-"$PWD/outputs_backup/ac_pilot_7676974/simulator/cases.json"}
extra=()
case "$mode" in
    simulator)
        [[ -f "$cases" ]] || { echo "Missing original pilot cases $cases" >&2; exit 2; }
        extra=(--case-prefix --cases "$cases" --episodes "${limit:-100}" --spline-diagnostics)
        ;;
    offline)
        if [[ -n "$limit" ]]; then extra=(--limit-windows "$limit"); fi
        ;;
    *) echo "Usage: sbatch eval_$variant.slurm {simulator|offline} [limit] [checkpoint]" >&2; exit 2 ;;
esac

exec srun --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
    bash scripts/cluster/environment.sh --gpu python -u -m "experiments.ac_spline_pilot.$mode" \
    --variant "$variant" --checkpoint "$checkpoint" --output "$output" "${extra[@]}"
