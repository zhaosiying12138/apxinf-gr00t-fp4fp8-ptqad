CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_120

fp4_gemm_bench: fp4_gemm_bench.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) $< -o $@ -lcublasLt

APXINF_FP4_ADAPTER ?= ../third_party/apxinf-robo/apxinf/crates/apxinf-cuda/adapters/cublaslt_fp4_adapter.cu

fp4_opbench: fp4_opbench.cu $(APXINF_FP4_ADAPTER)
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) $^ -o $@ -lcublasLt -Xcompiler -pthread

fp8_probe: fp8_probe.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) $< -o $@ -lcublasLt

clean:
	rm -f fp4_gemm_bench fp4_opbench fp8_probe

.PHONY: clean
