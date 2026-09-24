// fp8_probe.cu — isolate the engine's sm_120 cuBLAS status-15 failure.
// Engine pi05 fp8_static uses Fp8F16 GEMMs (E4M3 x E4M3) with various OUTPUT
// dtypes (F8E4M3 with per-tensor D-scale per tuning keys, F16/BF16 elsewhere).
// This probe tests, per output dtype: heuristic availability + matmul success.
//   a) E4M3 x E4M3 -> F32   (known good from spike)
//   b) E4M3 x E4M3 -> BF16
//   c) E4M3 x E4M3 -> F16
//   d) E4M3 x E4M3 -> E4M3  (+ CUBLASLT_MATMUL_DESC_D_SCALE... try without and with)
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;} } while(0)
#define CHECK_LT(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cublasLt err %d @%d\n",(int)s,__LINE__);return 1;} } while(0)

static uint8_t f2e4m3(float v){ return (uint8_t)__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3); }

int probe(const char* tag, cudaDataType cType, int M,int N,int K, bool withDScale){
  cublasLtHandle_t lt; CHECK_LT(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op; CHECK_LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t T=CUBLAS_OP_T, Nn=CUBLAS_OP_N;
  CHECK_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&T,sizeof(T)));
  CHECK_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&Nn,sizeof(Nn)));
  float s=0.01f; void* dS;
  CHECK_CUDA(cudaMalloc(&dS,4)); CHECK_CUDA(cudaMemcpy(dS,&s,4,cudaMemcpyHostToDevice));
  CHECK_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&dS,sizeof(dS)));
  CHECK_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&dS,sizeof(dS)));
  if(withDScale)
    CHECK_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,&dS,sizeof(dS)));

  cublasLtMatrixLayout_t A,B,C;
  CHECK_LT(cublasLtMatrixLayoutCreate(&A, CUDA_R_8F_E4M3, K, M, K));
  CHECK_LT(cublasLtMatrixLayoutCreate(&B, CUDA_R_8F_E4M3, K, N, K));
  CHECK_LT(cublasLtMatrixLayoutCreate(&C, cType, M, N, M));

  void *dA,*dB,*dC; size_t ws=256<<20, cb=CUDA_R_32F==cType?4:(cType==CUDA_R_8F_E4M3?1:2);
  CHECK_CUDA(cudaMalloc(&dA,(size_t)M*K)); CHECK_CUDA(cudaMalloc(&dB,(size_t)N*K));
  CHECK_CUDA(cudaMalloc(&dC,(size_t)M*N*cb)); void* wsp; CHECK_CUDA(cudaMalloc(&wsp,ws));
  std::vector<uint8_t> h(M*K); for(auto&v:h) v=f2e4m3(((float)((v*2654435761u)>>24)/255.0f-0.5f)*2.0f);
  CHECK_CUDA(cudaMemcpy(dA,h.data(),h.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB,h.data(),h.size(),cudaMemcpyHostToDevice));

  cublasLtMatmulPreference_t pref; CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));
  cublasLtMatmulHeuristicResult_t heur[4]; int nr=0;
  cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt,op,A,B,C,C,pref,4,heur,&nr);
  printf("%-28s heur=%d algos=%d", tag, (int)hs, nr);
  if(hs!=CUBLAS_STATUS_SUCCESS||nr==0){ printf("  => NO ALGO\n"); return 0; }
  float one=1.f, zero=0.f;
  cublasStatus_t es=cublasLtMatmul(lt,op,&one,dA,A,dB,B,&zero,dC,C,dC,C,&heur[0].algo,wsp,ws,0);
  cudaError_t se=cudaGetLastError();
  printf("  matmul=%d cuda=%s\n",(int)es, cudaGetErrorString(se));
  return 0;
}

int main(){
  cudaDeviceProp p; CHECK_CUDA(cudaGetDeviceProperties(&p,0));
  printf("gpu=%s sm_%d%d\n",p.name,p.major,p.minor);
  int M=512,N=2048,K=2048;
  probe("e4m3xe4m3 -> f32",       CUDA_R_32F,    M,N,K,false);
  probe("e4m3xe4m3 -> bf16",      CUDA_R_16BF,   M,N,K,false);
  probe("e4m3xe4m3 -> f16",       CUDA_R_16F,    M,N,K,false);
  probe("e4m3xe4m3 -> e4m3",      CUDA_R_8F_E4M3,M,N,K,false);
  probe("e4m3xe4m3 -> e4m3 +Dscale", CUDA_R_8F_E4M3,M,N,K,true);
  return 0;
}
