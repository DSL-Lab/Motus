#!/usr/bin/env bash
#SBATCH --job-name=motus-ac
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/ac-%j.log
set -euo pipefail
# Submit from Motus; one shared allocated entry point for checks and A/C.
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from Motus}"
[[ -f models/motus.py ]] || { echo 'Submit from Motus' >&2; exit 1; }
run_dir="outputs/jobs/$SLURM_JOB_ID"
mkdir -p "$run_dir"
git rev-parse HEAD > "$run_dir/source_commit.txt"
git diff --binary > "$run_dir/source.patch"
git ls-files -co --exclude-standard -z | tar --null -T - -czf "$run_dir/source.tar.gz"
for reference in ../bspline-policy ../WAM-RoboTwin ../WAM-RoboTwin/XPolicyLab; do
    printf '%s ' "$reference" >> "$run_dir/reference_commits.txt"
    git -C "$reference" rev-parse HEAD >> "$run_dir/reference_commits.txt"
done
scontrol show job "$SLURM_JOB_ID" > "$run_dir/allocation.txt"
printf '%q ' "$@" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
# Hash immutable large assets once inside an allocation, never on login nodes.
(
    flock 8
    if [[ ! -s outputs/asset_sha256.txt ]]; then
        sha256sum ../containers/robotwin-cu121.sif \
            pretrained_models/Motus_robotwin2/mp_rank_00_model_states.pt \
            pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
            pretrained_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
            > "outputs/asset_sha256.$SLURM_JOB_ID.tmp"
        mv "outputs/asset_sha256.$SLURM_JOB_ID.tmp" outputs/asset_sha256.txt
    fi
) 8>outputs/asset_sha256.lock
bash "${1:-scripts/experiments/smoke_test.sh}" "${@:2}"
