#!/bin/bash
#SBATCH --job-name=motus-spline
#SBATCH --account=aip-rjliao
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/project/6101778/kding
#SBATCH --output=motus-spline-%j.out

set -euo pipefail

# Export these three real asset directories before sbatch (or set them here).
: "${MOTUS_CHECKPOINT:?Set MOTUS_CHECKPOINT to the directory containing mp_rank_00_model_states.pt}"
: "${WAN_PATH:?Set WAN_PATH to the Wan2.2-TI2V-5B directory, including VAE and T5 assets}"
: "${VLM_PATH:?Set VLM_PATH to the Qwen3-VL-2B-Instruct processor directory}"

WORKSPACE=${MOTUS_WORKSPACE:-/project/6101778/kding}
REPO=${ROBOTWIN_REPO:-$WORKSPACE/WAM-RoboTwin}
MOTUS_REPO=${MOTUS_ROOT:-$WORKSPACE/Motus}
SIF=${MOTUS_SIF:-$WORKSPACE/containers/robotwin-cu121.sif}
CONTAINER_PYTHON=${MOTUS_CONTAINER_PYTHON:-/usr/bin/python}
TASK_NAME=${TASK_NAME:-click_alarmclock}
TASK_CONFIG=${TASK_CONFIG:-demo_randomized}
SCENE_SEEDS=${SCENE_SEEDS:-4300000}
POLICY_SEED=${POLICY_SEED:-42}

if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'ERROR: Submit this script with sbatch so Slurm allocates a GPU.' >&2
    exit 1
fi
for asset_dir in "$REPO" "$MOTUS_REPO" "$MOTUS_CHECKPOINT" "$WAN_PATH" "$VLM_PATH"; do
    if [[ ! -d "$asset_dir" ]]; then
        echo "ERROR: Directory does not exist: $asset_dir" >&2
        exit 1
    fi
done
if [[ ! -f "$SIF" || ! -f "$MOTUS_CHECKPOINT/mp_rank_00_model_states.pt" ]]; then
    echo 'ERROR: Container image or mp_rank_00_model_states.pt is missing.' >&2
    exit 1
fi

# Resolve relative paths before changing directories or building Apptainer binds.
REPO=$(realpath "$REPO")
MOTUS_REPO=$(realpath "$MOTUS_REPO")
MOTUS_CHECKPOINT=$(realpath "$MOTUS_CHECKPOINT")
WAN_PATH=$(realpath "$WAN_PATH")
VLM_PATH=$(realpath "$VLM_PATH")
SIF=$(realpath "$SIF")
RESULTS_ROOT=${MOTUS_RESULTS_ROOT:-$MOTUS_REPO/experiments/spline/runs}
mkdir -p "$RESULTS_ROOT"
RUN_OUTPUT="$(realpath "$RESULTS_ROOT")/slurm-$SLURM_JOB_ID"
mkdir "$RUN_OUTPUT" # Refuse to overwrite an earlier run with the same job ID.

module load apptainer/1.4.5
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

echo "Job: $SLURM_JOB_ID; node: ${SLURMD_NODENAME:-unknown}"
echo "Task: $TASK_NAME / $TASK_CONFIG; scenes: $SCENE_SEEDS; policy seed: $POLICY_SEED"
echo "Results: $RUN_OUTPUT"
echo "Allocated CUDA device(s): $CUDA_VISIBLE_DEVICES; container CUDA ordinal: 0"
nvidia-smi -L
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
    --format=csv -l 5 > "$RUN_OUTPUT/gpu-utilization.csv" 2>&1 &
gpu_monitor_pid=$!
cleanup() {
    kill "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

container_pythonpath=/workspace/Motus:/workspace/RoboTwin:/workspace/RoboTwin/XPolicyLab
extra_binds=()
# Optional, already-installed Python 3.10 packages supplementing the immutable SIF.
# Nothing is installed or downloaded during a GPU job.
if [[ -n "${MOTUS_EXTRA_PACKAGES:-}" ]]; then
    if [[ ! -d "$MOTUS_EXTRA_PACKAGES" ]]; then
        echo "ERROR: MOTUS_EXTRA_PACKAGES directory is missing: $MOTUS_EXTRA_PACKAGES" >&2
        exit 1
    fi
    extra_binds+=(--bind "$(realpath "$MOTUS_EXTRA_PACKAGES"):/opt/motus-python:ro")
    container_pythonpath="/opt/motus-python:$container_pythonpath"
fi

# Keep the existing graphics/GPU setup; run commands instead of opening a shell.
apptainer exec \
    --nvccli --writable-tmpfs --cleanenv \
    --bind "$REPO:/workspace/RoboTwin" \
    --bind "$MOTUS_REPO:/workspace/Motus" \
    --bind "$MOTUS_CHECKPOINT:/models/motus:ro" \
    --bind "$WAN_PATH:/models/wan:ro" \
    --bind "$VLM_PATH:/models/qwen:ro" \
    --bind "$RUN_OUTPUT:/results" \
    "${extra_binds[@]}" \
    --bind /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro \
    --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro \
    --env "PYTHONPATH=$container_pythonpath" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
    --env "OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
    --env TORCHINDUCTOR_COMPILE_THREADS=1 \
    --env PYTHONUNBUFFERED=1 \
    --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    --env VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    --env __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    "$SIF" bash -s -- "$CONTAINER_PYTHON" "$TASK_NAME" "$TASK_CONFIG" \
    "$POLICY_SEED" "$SCENE_SEEDS" "$@" <<'MOTUS_CONTAINER'
# BEGIN CONTAINER SCRIPT
set -euo pipefail
container_python=$1
task_name=$2
task_config=$3
policy_seed=$4
read -r -a scene_seeds <<< "$5"
shift 5
if [[ ${#scene_seeds[@]} -eq 0 ]]; then
    echo 'ERROR: SCENE_SEEDS must contain at least one integer.' >&2
    exit 1
fi
for scene_seed in "${scene_seeds[@]}"; do
    if [[ ! "$scene_seed" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid scene seed: $scene_seed" >&2
        exit 1
    fi
done
export XDG_RUNTIME_DIR="/tmp/motus-xdg-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="/tmp/motus-triton-$SLURM_JOB_ID"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
cd /workspace/Motus

eval_command=(
    "$container_python" /workspace/Motus/experiments/spline/evaluate.py
    --robotwin-root /workspace/RoboTwin --sim-python "$container_python"
    --checkpoint /models/motus --wan-path /models/wan --vlm-path /models/qwen
    --task "$task_name" --task-config "$task_config" --policy-seed "$policy_seed"
    "$@"
)
"${eval_command[@]}" --output /results/preflight --preflight

# Import real symbols: namespace directories alone do not establish compatibility.
if ! "$container_python" /workspace/Motus/experiments/spline/check_runtime.py --require-cuda; then
    echo 'ERROR: Container runtime check failed. Use a compatible MOTUS_SIF or prepared MOTUS_EXTRA_PACKAGES directory.' >&2
    exit 1
fi

echo 'Running original Motus execution on the first requested scene.'
"${eval_command[@]}" --original --scene-seeds "${scene_seeds[0]}" --output /results/original

echo 'Running dense/spline at 1x, 1.5x, and 2x.'
matrix_status=0
"${eval_command[@]}" --scene-seeds "${scene_seeds[@]}" --output /results/matrix || matrix_status=$?
if "$container_python" /workspace/Motus/experiments/spline/summarize.py \
        /results/matrix > /results/summary.json; then
    cat /results/summary.json
else
    echo 'No complete summary; inspect the per-episode console.log and process.json files.' >&2
    matrix_status=1
fi
exit "$matrix_status"
MOTUS_CONTAINER
