#!/usr/bin/env bash
# Invoke from an activated CUDA environment: bash experiments/l1/run_gpu.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
study="artifacts/l1-study-gpu-v0.1"
mkdir -p artifacts/l1-gpu-logs
exec > >(tee -a "artifacts/l1-gpu-logs/run-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
git diff --exit-code HEAD -- src experiments tests pyproject.toml
if [[ ! -f "$study/study.json" ]]; then
    sha256sum --check --strict l1-gpu-inputs.sha256
    python -m experiments.l1.gpu_preflight --output artifacts/l1-gpu-logs/cuda-preflight.json
    python -m experiments.l1.study init \
        --dataset artifacts/l1-v0.1-final --cache artifacts/l1-embeddings-v0.1 \
        --output "$study" --code-revision "$(git rev-parse HEAD)" \
        --review experiments/reviews/l1-catalog-approval-2026-10-04.json \
        --device cuda --threads 2
fi
python -m experiments.l1.study run --output "$study" --threads 2
