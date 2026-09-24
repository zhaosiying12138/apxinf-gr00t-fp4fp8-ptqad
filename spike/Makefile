CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_120

fp4_gemm_bench: fp4_gemm_bench.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) $< -o $@ -lcublasLt

clean:
	rm -f fp4_gemm_bench

.PHONY: clean
