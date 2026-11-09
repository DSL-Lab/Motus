#!/bin/bash
# CPU-only preparation; no GPU allocation or image modification.
set -euo pipefail

case "${1:-}" in
    ""|--check|--download-models) mode=${1:-install} ;;
    *) echo "Usage: bash $0 [--check|--download-models]" >&2; exit 2 ;;
esac
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=${MOTUS_WORKSPACE:-$(cd "$script_dir/../../.." && pwd)}
motus_repo=${MOTUS_ROOT:-$workspace/Motus}
sif=${MOTUS_SIF:-$workspace/containers/robotwin-cu121.sif}
package_base=${MOTUS_PACKAGE_BASE:-$motus_repo/.runtime/apptainer}
models_root=${MOTUS_MODELS_ROOT:-$motus_repo/pretrained_models}
mkdir -p "$package_base/lib/python3.10/site-packages"
package_base=$(realpath "$package_base")
motus_repo=$(realpath "$motus_repo")
sif=$(realpath "$sif")
module load apptainer/1.4.5

apptainer exec --cleanenv \
    --bind "$motus_repo:/workspace/Motus" \
    --bind "$package_base:/opt/motus-userbase" \
    --env PYTHONUSERBASE=/opt/motus-userbase \
    --env PYTHONPATH=/opt/motus-userbase/lib/python3.10/site-packages \
    --env DS_BUILD_OPS=0 \
    --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 --env TORCHINDUCTOR_COMPILE_THREADS=1 \
    "$sif" bash -s -- "$mode" <<'MOTUS_PREPARE'
set -euo pipefail
if [[ "$1" != --check ]]; then
    unset PYTHONNOUSERSITE # The image disables --user installs by default.
    # --user reuses installed container packages, unlike pip --target. Explicit
    # PYTHONPATH also works when the image disables automatic user-site loading.
    /usr/bin/python - <<'PY'
import importlib.metadata as metadata
from pathlib import Path
names = ("torch", "torchvision", "numpy", "scipy")
Path("/opt/motus-userbase/base-constraints.txt").write_text(
    "".join(f"{name}=={metadata.version(name)}\n" for name in names)
)
PY
    /usr/bin/python -m pip install --user --no-build-isolation \
        --no-warn-script-location --disable-pip-version-check \
        --constraint /opt/motus-userbase/base-constraints.txt \
        --requirement /workspace/Motus/experiments/spline/container-requirements.txt
fi
/usr/bin/python /workspace/Motus/experiments/spline/check_runtime.py
MOTUS_PREPARE

if [[ "$mode" == --download-models ]]; then
    mkdir -p "$models_root"
    models_root=$(realpath "$models_root")
    apptainer exec --cleanenv \
        --bind "$models_root:/models-download" \
        --bind "$package_base:/opt/motus-userbase" \
        --env PYTHONPATH=/opt/motus-userbase/lib/python3.10/site-packages \
        "$sif" /usr/bin/python - <<'PY'
from huggingface_hub import snapshot_download

# ~30 GB total. Backbone weights are already in Motus; VAE and T5 load separately.
snapshot_download("motus-robotics/Motus_robotwin2",
                  local_dir="/models-download/Motus_robotwin2",
                  allow_patterns=["mp_rank_00_model_states.pt", "config.json"])
snapshot_download("Wan-AI/Wan2.2-TI2V-5B",
                  local_dir="/models-download/Wan2.2-TI2V-5B",
                  allow_patterns=["config.json", "Wan2.2_VAE.pth",
                                  "models_t5_umt5-xxl-enc-bf16.pth", "google/*"])
snapshot_download("Qwen/Qwen3-VL-2B-Instruct",
                  local_dir="/models-download/Qwen3-VL-2B-Instruct",
                  allow_patterns=["*.json", "*.txt", "*.jinja", "*.model"])
PY
fi

if [[ "$mode" == --download-models ]]; then
    checkpoint="$models_root/Motus_robotwin2"
    wan="$models_root/Wan2.2-TI2V-5B"
    qwen="$models_root/Qwen3-VL-2B-Instruct"
else
    checkpoint=${MOTUS_CHECKPOINT:-$models_root/Motus_robotwin2}
    wan=${WAN_PATH:-$models_root/Wan2.2-TI2V-5B}
    qwen=${VLM_PATH:-$models_root/Qwen3-VL-2B-Instruct}
fi
{
    printf 'export MOTUS_SIF=%q\n' "$sif"
    printf 'export MOTUS_EXTRA_PACKAGES=%q\n' "$package_base/lib/python3.10/site-packages"
    printf 'export MOTUS_CHECKPOINT=%q\n' "$checkpoint"
    printf 'export WAN_PATH=%q\n' "$wan"
    printf 'export VLM_PATH=%q\n' "$qwen"
} > "$package_base/env.sh"
echo "Dependencies checked. Load the batch-job paths with:"
printf 'source %q\n' "$package_base/env.sh"
if [[ ! -f "$checkpoint/mp_rank_00_model_states.pt" ]]; then
    echo 'Model assets still need configuring; use existing paths or rerun with --download-models.'
fi
