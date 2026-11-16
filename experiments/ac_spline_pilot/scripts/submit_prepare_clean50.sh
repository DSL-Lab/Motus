#!/usr/bin/env bash
#SBATCH --account=aip-rjliao
#SBATCH --job-name=ac_prepare
#SBATCH --time=03:00:00
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/ac_prepare_%j.out
#SBATCH --chdir=/project/6101778/kding/Motus
#SBATCH --export=ALL
set -euo pipefail
dry_run=false
if [[ $# == 1 && "$1" == --dry-run ]]; then
    dry_run=true
    script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
elif (( $# != 0 )); then
    echo 'Usage: bash submit_prepare_clean50.sh --dry-run | sbatch submit_prepare_clean50.sh' >&2
    exit 2
else
    : "${SLURM_JOB_ID:?Submit preparation with sbatch, or use --dry-run}"
    script_root=$PWD
fi
cd "${MOTUS_REPO_ROOT:-$script_root}"
wam_root="${WAM_ROBOTWIN_ROOT:-$PWD/../WAM-RoboTwin}"
source_root="${MOTUS_RAW_CLEAN_ROOT:-$wam_root/data/demo_clean/place_can_basket/aloha_agilex}"
output_root="${MOTUS_CLEAN50_ROOT:-$PWD/outputs/ac_spline_pilot/clean50}"
wan_assets="${MOTUS_WAN_ASSETS:-$PWD/pretrained_models/Wan2.2-TI2V-5B}"
fresh_args=()
case "${MOTUS_PREP_FRESH:-1}" in
    1) fresh_args=(--fresh) ;;
    0) ;; # Explicit resume mode keeps existing verified outputs.
    *) echo 'MOTUS_PREP_FRESH must be 0 or 1' >&2; exit 2 ;;
esac
command=(srun --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}"
    bash scripts/cluster/environment.sh --gpu
    env MOTUS_REPO_ROOT="$PWD" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    python "$wam_root/scripts/prepare_motus_clean50.py"
    --source "$source_root" --output "$output_root" --wan-assets "$wan_assets" "${fresh_args[@]}")
if "$dry_run"; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    "${command[@]}"
fi
