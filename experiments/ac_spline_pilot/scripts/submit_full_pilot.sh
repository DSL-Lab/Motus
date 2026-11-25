#!/usr/bin/env bash
#SBATCH --account=aip-rjliao
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=ac_pilot
#SBATCH --output=outputs/ac_pilot_%j.out
#SBATCH --chdir=/project/6101778/kding/Motus
#SBATCH --export=ALL

# Submit from Motus/: mkdir -p outputs && sbatch experiments/ac_spline_pilot/scripts/submit_full_pilot.sh
# Check paths first: bash experiments/ac_spline_pilot/scripts/submit_full_pilot.sh --check
# Export any MOTUS_* overrides before sbatch. Both variants share the same assets.
set -euo pipefail
check_only=false
dry_run=false
submit=false
task=place_can_basket
while (( $# )); do
    case "$1" in
        --task) task="${2:?--task needs a task name}"; shift 2 ;;
        --check|--dry-run|--submit)
            if "$check_only"; then echo 'Choose one of --check, --dry-run or --submit' >&2; exit 2; fi
            check_only=true
            case "$1" in --dry-run) dry_run=true ;; --submit) submit=true ;; esac
            shift ;;
        *) echo 'Usage: submit_full_pilot.sh [--submit|--check|--dry-run] [--task place_can_basket]' >&2; exit 2 ;;
    esac
done
[[ "$task" == place_can_basket ]] || { echo 'Current A/C experiment supports place_can_basket only' >&2; exit 2; }
episodes="${MOTUS_EVAL_EPISODES:-100}"
eval_seed="${MOTUS_EVAL_SEED:-0}"
[[ "$episodes" =~ ^[1-9][0-9]*$ && "$eval_seed" =~ ^[0-9]+$ ]] || { echo 'Invalid evaluation episode count or seed' >&2; exit 2; }
if "$check_only"; then
    # Direct execution keeps the real script path; Slurm copies batch scripts.
    script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
    repo_root="${MOTUS_REPO_ROOT:-$script_root}"
else
    : "${SLURM_JOB_ID:?Submit this script with sbatch, or use --check to check artifact paths}"
    repo_root="${MOTUS_REPO_ROOT:-$PWD}"
fi
cd "$repo_root"
[[ -f train/train.py && -f scripts/cluster/environment.sh ]] || {
    echo 'MOTUS_REPO_ROOT must point to the Motus repository' >&2
    exit 2
}

# Prepared clean-50 inputs only. These defaults are destinations for the required
# preprocessing outputs, not a claim that those artifacts already exist.
# Supply overrides for your actual paths; randomized-500 caches are rejected.
export MOTUS_CLEAN50_ROOT="${MOTUS_CLEAN50_ROOT:-$PWD/outputs/ac_spline_pilot/clean50}"
export MOTUS_DATASET_ROOT="${MOTUS_DATASET_ROOT:-$MOTUS_CLEAN50_ROOT/dataset}"
export MOTUS_WINDOWS="${MOTUS_WINDOWS:-$MOTUS_CLEAN50_ROOT/windows.json}"
export MOTUS_NORMALIZATION="${MOTUS_NORMALIZATION:-$MOTUS_CLEAN50_ROOT/normalization.json}"
export MOTUS_KNOT_SCALING="${MOTUS_KNOT_SCALING:-$MOTUS_CLEAN50_ROOT/knot_scaling.json}"
export MOTUS_BASE_CHECKPOINT="${MOTUS_BASE_CHECKPOINT:-$PWD/pretrained_models/Motus_robotwin2/mp_rank_00_model_states.pt}"
export MOTUS_WAN="${MOTUS_WAN:-$PWD/.runtime/model-metadata/Wan2.2-TI2V-5B}"
export MOTUS_VAE="${MOTUS_VAE:-$PWD/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
export MOTUS_VLM="${MOTUS_VLM:-$PWD/.runtime/model-metadata/Qwen3-VL-2B-Instruct}"

container_env=("OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}")
missing_assets=0
for name in MOTUS_DATASET_ROOT MOTUS_WINDOWS MOTUS_NORMALIZATION MOTUS_KNOT_SCALING \
            MOTUS_BASE_CHECKPOINT MOTUS_WAN MOTUS_VAE MOTUS_VLM; do
    # Apptainer --cleanenv needs the same checked asset values explicitly.
    container_env+=("$name=${!name}")
    # Preparation produces these four paths before the dependent training job.
    if { "$submit" || "$dry_run"; } && [[ "$name" == MOTUS_DATASET_ROOT || "$name" == MOTUS_WINDOWS || "$name" == MOTUS_NORMALIZATION || "$name" == MOTUS_KNOT_SCALING ]]; then
        continue
    fi
    if [[ ! -e "${!name}" ]]; then
        echo "Missing $name: ${!name}. Set this variable to the prepared artifact before sbatch." >&2
        missing_assets=$((missing_assets + 1))
    fi
done
if (( missing_assets )); then
    echo "$missing_assets required artifact paths are missing. Raw HDF5 demos require preprocessing; they cannot be used as MOTUS_DATASET_ROOT." >&2
    exit 2
fi
if "$submit"; then
    mkdir -p outputs
    prep_job=$(MOTUS_PREP_FRESH=0 sbatch --parsable experiments/ac_spline_pilot/scripts/submit_prepare_clean50.sh)
    echo "Preparation job: $prep_job (reuse enabled); training/evaluation depends on its success."
    exec sbatch --dependency="afterok:${prep_job%%;*}" \
        experiments/ac_spline_pilot/scripts/submit_full_pilot.sh --task "$task"
fi
if ! "$dry_run" && [[ ! -d "$MOTUS_DATASET_ROOT/clean/place_can_basket/qpos" ]]; then
    echo 'MOTUS_DATASET_ROOT must contain clean/place_can_basket/qpos (50 complete episodes)' >&2
    exit 2
fi

if "$check_only" && ! "$dry_run"; then
    echo 'Artifact paths exist. This path-only check does not load models or submit a job.'
    echo 'The trainer will validate the complete clean-50 corpus, 45/5 split, caches and clocks at startup.'
    exit 0
fi

run_root="${MOTUS_PILOT_OUTPUT:-$PWD/outputs/ac_pilot_${SLURM_JOB_ID:-DRY_RUN}}"
if ! "$dry_run"; then
    mkdir -p "$(dirname "$run_root")"
    mkdir "$run_root"
    mkdir "$run_root/logs"
fi

if "$dry_run"; then
    echo 'Dry run: no jobs or commands will execute. --submit schedules preparation, then this job afterok.'
    echo 'MOTUS_PREP_FRESH=0 sbatch --parsable experiments/ac_spline_pilot/scripts/submit_prepare_clean50.sh'
    echo 'sbatch --dependency=afterok:<PREPARATION_JOB_ID> experiments/ac_spline_pilot/scripts/submit_full_pilot.sh --task place_can_basket'
else
    echo "Starting matched A/C pilot job $SLURM_JOB_ID at $(date -Is)"
fi
echo "Dataset: $MOTUS_DATASET_ROOT (clean place_can_basket, seed-0 45/5 split)"
echo "Outputs: $run_root"
declare -A checkpoints
for variant in A C; do
    echo "Variant $variant: 15 epochs"
    variant_dir="$run_root/$variant/robotwin_pilot_$variant/pilot-seed-0"
    command=(srun --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}"
        bash scripts/cluster/environment.sh --gpu env "${container_env[@]}"
        python -u -m experiments.ac_spline_pilot.trainer
        --config "experiments/ac_spline_pilot/configs/robotwin_pilot_${variant}.yaml"
        --checkpoint_dir "$run_root/$variant" --report_to none)
    if "$dry_run"; then
        printf '%q ' "${command[@]}"
        printf '\n'
        checkpoint="$variant_dir/<final_checkpoint>"
    else
        "${command[@]}" 2>&1 | tee "$run_root/logs/train_${variant}.log"
        if [[ ! -f "$variant_dir/final_checkpoint.txt" ]]; then
            echo "Training did not publish its final checkpoint: $variant_dir/final_checkpoint.txt" >&2; exit 2;
        fi
        read -r checkpoint_name < "$variant_dir/final_checkpoint.txt"
        [[ "$checkpoint_name" =~ ^checkpoint_step_[0-9]+$ ]] || { echo 'Invalid final checkpoint name' >&2; exit 2; }
        checkpoint="$variant_dir/$checkpoint_name"
        [[ -f "$checkpoint/research_adapter.pt" && -f "$checkpoint/research_config.json" ]] || {
            echo "Training did not produce its final checkpoint: $checkpoint" >&2; exit 2;
        }
    fi
    checkpoints[$variant]="$checkpoint"
done
# These are real closed-loop rollouts. generation/ above remains offline diagnostics.
eval_status=0
for variant in A C; do
    checkpoint="${checkpoints[$variant]}"
    command=(srun --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}"
        bash scripts/cluster/environment.sh --gpu env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}"
        python -u -m experiments.ac_spline_pilot.simulator --task "$task"
        --checkpoint "$checkpoint" --cases "$run_root/simulator/cases.json"
        --episodes "$episodes" --seed "$eval_seed" --output "$run_root/simulator/$variant")
    if "$dry_run"; then
        printf '%q ' "${command[@]}"; printf '\n'
    elif ! "${command[@]}" 2>&1 | tee "$run_root/logs/simulator_${variant}.log"; then
        # Preserve the error and still attempt the matched other variant.
        eval_status=1
    fi
done
if ! "$dry_run"; then echo "A/C training and simulator evaluation finished at $(date -Is), status=$eval_status"; fi
exit "$eval_status"
