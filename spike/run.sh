#!/usr/bin/env bash
# FP4 spike: go/no-go gate. Results land in results/spike/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../results/spike
STAMP=$(date +%Y%m%d_%H%M%S)

make

# 1) numerical verification vs CPU fp64 reference (small shape)
./fp4_gemm_bench --verify 2>&1 | tee ../results/spike/verify_${STAMP}.log

# 2) latency/throughput sweep
./fp4_gemm_bench 2>../results/spike/env_${STAMP}.log | tee ../results/spike/gemm_${STAMP}.csv

echo "done -> results/spike/"
