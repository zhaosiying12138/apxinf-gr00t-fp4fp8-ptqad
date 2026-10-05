#!/usr/bin/env bash
set -euxo pipefail
PROJECT=/home/zhaosiying/codebase/fp4vla
BUILD="$PROJECT/paper/_build/capture_compile_20261006/build_r3"
ADAPTER="$PROJECT/third_party/apxinf-robo/apxinf/crates/apxinf-cuda/adapters/cublaslt_fp4_adapter.cu"
test ! -e "$BUILD"
mkdir "$BUILD"
cp "$PROJECT/spike/Makefile" "$PROJECT/spike/fp4_gemm_bench.cu" "$PROJECT/spike/fp4_opbench.cu" "$PROJECT/spike/fp8_probe.cu" "$BUILD/"
/usr/local/cuda/bin/nvcc --version
make -C "$BUILD" -j1 ARCH=sm_120 APXINF_FP4_ADAPTER="$ADAPTER" fp4_gemm_bench fp4_opbench fp8_probe
sha256sum "$BUILD/Makefile" "$BUILD/"*.cu "$ADAPTER" "${ADAPTER%/*}/fp4_plan_cache.cuh"
sha256sum "$BUILD/fp4_gemm_bench" "$BUILD/fp4_opbench" "$BUILD/fp8_probe"
