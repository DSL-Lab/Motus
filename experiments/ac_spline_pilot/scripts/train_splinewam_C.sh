#!/usr/bin/env bash
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-08:00:00
#SBATCH --job-name=splinewam_C
#SBATCH --output=outputs/ac_splinewam_C/train-%j.log
#SBATCH --chdir=/project/6101778/kding/Motus

set -euo pipefail
artifacts="${MOTUS_SPLINEWAM_DATA:-$PWD/outputs/ac_splinewam_C_170/data}"
: "${SLURM_JOB_ID:?Submit this training script with sbatch}"
[[ -f "$artifacts/preparation_report.json" ]] || { echo "Missing validated SplineWAM inputs in $artifacts" >&2; exit 2; }

srun --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    bash scripts/cluster/environment.sh --gpu env \
    "MOTUS_DATASET_ROOT=$artifacts/dataset" \
    "MOTUS_WINDOWS=$artifacts/windows.json" \
    "MOTUS_NORMALIZATION=$artifacts/normalization.json" \
    "MOTUS_KNOT_SCALING=$artifacts/knot_scaling.json" \
    "MOTUS_BASE_CHECKPOINT=${MOTUS_BASE_CHECKPOINT:-$PWD/pretrained_models/Motus_robotwin2/mp_rank_00_model_states.pt}" \
    "MOTUS_WAN=${MOTUS_WAN:-$PWD/.runtime/model-metadata/Wan2.2-TI2V-5B}" \
    "MOTUS_VAE=${MOTUS_VAE:-$PWD/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}" \
    "MOTUS_VLM=${MOTUS_VLM:-$PWD/.runtime/model-metadata/Qwen3-VL-2B-Instruct}" \
    python -u -m experiments.ac_spline_pilot.trainer \
    --config experiments/ac_spline_pilot/configs/robotwin_pilot_C_splinewam_170.yaml \
    --checkpoint_dir "${MOTUS_SPLINEWAM_TRAIN:-$PWD/outputs/ac_splinewam_C_170/train}"
