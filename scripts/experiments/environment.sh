#!/usr/bin/env bash
# Source on the host; execute commands inside the existing container.
set -euo pipefail
MOTUS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MOTUS_WORKSPACE="$(dirname "$MOTUS_ROOT")"
module load apptainer/1.4.5
motus_python() {
    local gpu_args=()
    if [[ -n "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" ]]; then
        export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
        export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
        export APPTAINERENV_NVIDIA_DRIVER_CAPABILITIES="$NVIDIA_DRIVER_CAPABILITIES"
        export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
        gpu_args=(--nvccli --writable-tmpfs
            --env "NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES"
            --env "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
            --bind /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro
            --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro
            --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
            --env VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
            --env __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json)
    fi
    apptainer exec "${gpu_args[@]}" --cleanenv \
        --bind "$MOTUS_WORKSPACE:$MOTUS_WORKSPACE" \
        --bind /scratch/kding:/scratch/kding:ro \
        --env "PYTHONPATH=$MOTUS_ROOT/.runtime/apptainer/lib/python3.10/site-packages:$MOTUS_ROOT" \
        --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" --env OPENBLAS_NUM_THREADS=1 \
        --env "SLURM_JOB_ID=${SLURM_JOB_ID:-}" --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
        --env "TRITON_CACHE_DIR=/tmp/motus-triton-${SLURM_JOB_ID:-cpu}" \
        --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONUNBUFFERED=1 \
        "$MOTUS_WORKSPACE/containers/robotwin-cu121.sif" python "$@"
}
