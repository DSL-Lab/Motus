#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
: "${SLURM_JOB_ID:?Smoke checks run in Slurm}"
cd "$MOTUS_ROOT"
case "${1:-numerical}" in
    numerical) motus_python -m unittest discover -s tests -v ;;
    model) motus_python tests/allocated_model_smoke.py "${@:2}" ;;
    overfit) motus_python tests/allocated_overfit.py "${@:2}" ;;
    decode) motus_python tests/allocated_decode.py ;;
    evaluate) motus_python inference/robotwin/evaluate_pair.py "${@:2}" ;;
    inference) motus_python tests/allocated_inference.py "${@:2}" ;;
    *) echo 'Usage: smoke_test.sh numerical|model|overfit A|C|inference|evaluate [evaluator arguments]' >&2; exit 2 ;;
esac
