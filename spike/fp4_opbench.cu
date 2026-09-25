// fp4_opbench.cu — fp4_linear operator (quantize+GEMM) vs BF16 GEMM on pi05's
// real layer shapes. Paper evidence: per-layer engine speedup at NVFP4.
// Shapes: top-by-count from pi05.nvfp4 manifest; M swept {522, 778, 2048}
// (pi05 token counts incl. vision prefix) x real (N,K).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

extern "C" cudaError_t apxinf_nvfp4_quantize_activation(
    const void* x, void* packed, void* scale_swz, int rows, int k, cudaStream_t stream);
extern "C" cublasStatus_t apxinf_fp4_gemm_f16(
    int m, int n, int k, const void* a_packed, const void* b_packed,
    const void* a_scale, const void* b_scale, __half* c,
    void* ws, size_t ws_bytes, cudaStream_t stream);

#define CK(x) do{auto e=(x); if(e!=cudaSuccess){printf("cuda err %s@%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

struct Shape { int n, k; const char* tag; };
static const Shape SHAPES[] = {
    {1152,1152,"gemma-1152"},{3072,1024,"mlp-3072x1024"},{4096,1024,"mlp-4096x1024"},
    {16384,2048,"big-16384x2048"},{2048,2048,"attn-2048x2048"},{4304,1152,"geglu-4304x1152"},
};
static const int MS[] = {522, 778, 2048};

int main(){
  cublasLtHandle_t lt; cublasLtCreate(&lt);
  void* ws; CK(cudaMalloc(&ws, 256<<20));
  printf("shape,M,K,fp4_us(p50),bf16_us(p50),speedup\n");
  for (auto& sh : SHAPES) {
    for (int m : MS) {
      int n = sh.n, k = sh.k, kb = k/16;
      // random f16 activation (m x k) and "quantized" weight (n x k packed+scales)
      size_t asz = (size_t)m*k, bsz=(size_t)n*k;
      __half* hx; CK(cudaMalloc(&hx, asz*2));
      uint8_t *ap,*bp,*asw,*bsw; __half* c4; __nv_bfloat16* cB; __half* wB;
      CK(cudaMalloc(&ap, asz/2)); CK(cudaMalloc(&bp, bsz/2));
      CK(cudaMalloc(&asw, (size_t)512*((kb+3)/4)*((m+127)/128)));
      CK(cudaMalloc(&bsw, (size_t)512*((kb+3)/4)*((n+127)/128)));
      CK(cudaMalloc(&c4, (size_t)m*n*2)); CK(cudaMalloc(&cB,(size_t)m*n*2));
      CK(cudaMalloc(&wB, bsz*2));
      CK(cudaMemset(ap,0x22,asz/2)); CK(cudaMemset(bp,0x22,bsz/2));
      CK(cudaMemset(asw,0x38,asw?1:1)); CK(cudaMemset(asw,0x38,(size_t)512*((kb+3)/4)*((m+127)/128)));
      CK(cudaMemset(bsw,0x38,(size_t)512*((kb+3)/4)*((n+127)/128)));

      // ---- fp4 pipeline: quantize + gemm
      for (int w=0;w<10;++w){ apxinf_nvfp4_quantize_activation(hx,ap,asw,m,k,0);
        apxinf_fp4_gemm_f16(m,n,k,ap,bp,asw,bsw,c4,ws,256<<20,0); }
      CK(cudaDeviceSynchronize());
      cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      float fmin=1e9f;
      for(int r=0;r<30;++r){
        CK(cudaEventRecord(e0));
        apxinf_nvfp4_quantize_activation(hx,ap,asw,m,k,0);
        apxinf_fp4_gemm_f16(m,n,k,ap,bp,asw,bsw,c4,ws,256<<20,0);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); if(ms*1000<fmin) fmin=ms*1000;
      }
      // ---- bf16 reference gemm (same (m,k)x(k,n))
      cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la,lb,lc; cublasLtMatmulPreference_t pf;
      cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F);
      cublasOperation_t T=CUBLAS_OP_T,Nn=CUBLAS_OP_N;
      cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&T,sizeof(T));
      cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&Nn,sizeof(Nn));
      cublasLtMatrixLayoutCreate(&la,CUDA_R_16BF,k,m,k);
      cublasLtMatrixLayoutCreate(&lb,CUDA_R_16BF,k,n,k);
      cublasLtMatrixLayoutCreate(&lc,CUDA_R_16BF,m,n,m);
      cublasLtMatmulPreferenceCreate(&pf); size_t wsz=256<<20;
      cublasLtMatmulPreferenceSetAttribute(pf,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz));
      cublasLtMatmulHeuristicResult_t h; int nr=0;
      auto hst=cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,lc,lc,pf,1,&h,&nr);
      float bmin=1e9f;
      if(hst==0&&nr>0){
        float one=1,zero=0;
        for(int w=0;w<10;++w) cublasLtMatmul(lt,op,&one,hx,la,wB,lb,&zero,cB,lc,cB,lc,&h.algo,ws,wsz,0);
        CK(cudaDeviceSynchronize());
        for(int r=0;r<30;++r){
          CK(cudaEventRecord(e0));
          cublasLtMatmul(lt,op,&one,hx,la,wB,lb,&zero,cB,lc,cB,lc,&h.algo,ws,wsz,0);
          CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
          float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); if(ms*1000<bmin) bmin=ms*1000;
        }
      }
      printf("%s,%d,%d,%.1f,%.1f,%.2fx\n",sh.tag,m,k,fmin,bmin,bmin/fmin);
      cudaFree(hx);cudaFree(ap);cudaFree(bp);cudaFree(asw);cudaFree(bsw);cudaFree(c4);cudaFree(cB);cudaFree(wB);
      cublasLtMatmulPreferenceDestroy(pf);cublasLtMatrixLayoutDestroy(lc);
      cublasLtMatrixLayoutDestroy(lb);cublasLtMatrixLayoutDestroy(la);cublasLtMatmulDescDestroy(op);
    }
  }
  return 0;
}
