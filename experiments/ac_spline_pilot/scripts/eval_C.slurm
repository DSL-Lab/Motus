#!/usr/bin/env bash
#SBATCH --account=aip-rjliao
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:50:00
#SBATCH --job-name=motus-eval-C
#SBATCH --output=outputs/eval-C-%j.log
#SBATCH --chdir=/project/6101778/kding/Motus

set -euo pipefail
exec bash experiments/ac_spline_pilot/scripts/eval_common.sh C "$@"
