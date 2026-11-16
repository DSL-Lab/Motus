#!/usr/bin/env bash
# Noninteractive form of WAM-RoboTwin/robotwin.sh; no package installation.
set -euo pipefail
cluster_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
project_root=$(dirname "$cluster_root")
module load apptainer/1.4.5
container=${MOTUS_CONTAINER:-$project_root/containers/robotwin-cu121.sif}
overlay=${MOTUS_PYTHON_OVERLAY:-$cluster_root/.runtime/apptainer/lib/python3.10/site-packages}
python_path="$cluster_root:$overlay:$project_root/WAM-RoboTwin/scripts/spline_tools:$project_root/WAM-RoboTwin:$project_root/WAM-RoboTwin/XPolicyLab"
args=(exec --cleanenv --writable-tmpfs --bind "$project_root:$project_root"
      --bind "$project_root/WAM-RoboTwin:/workspace/RoboTwin"
      --env "PYTHONPATH=$python_path" --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1
      --env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=4,TORCHINDUCTOR_COMPILE_THREADS=1,PYTHONUNBUFFERED=1)
case "${1:-}" in
  --cpu) shift ;;  # CPU component/conversion tests only.
  --gpu)
    shift
    : "${SLURM_JOB_ID:?GPU execution needs a Slurm allocation}"
    : "${SLURMD_NODENAME:?Use srun on compute, never a login node}"
    : "${CUDA_VISIBLE_DEVICES:?No Slurm GPU allocated}"
    export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    export NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    args+=(--nvccli
      --bind /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro
      --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro
      --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
      --env VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
      --env __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
      --env "SLURM_JOB_ID=$SLURM_JOB_ID,SLURMD_NODENAME=$SLURMD_NODENAME,CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    ;;
  *) echo 'Usage: environment.sh --cpu|--gpu COMMAND [ARG...]' >&2; exit 2 ;;
esac
exec apptainer "${args[@]}" "$container" bash -c '
  set -euo pipefail
  export XDG_RUNTIME_DIR=$(mktemp -d /tmp/motus-xdg.XXXXXX)
  exec "$@"
' bash "$@"
