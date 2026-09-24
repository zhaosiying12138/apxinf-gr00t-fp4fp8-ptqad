// fp4_adapter_test.cu — functional test for cublaslt_fp4_adapter on sm_120.
// Quantizes small matrices with spike-verified helpers, calls
// apxinf_fp4_gemm_f16, compares vs CPU dequant reference.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cublasLt.h>

extern "C" cublasStatus_t apxinf_fp4_gemm_f16(
    cublasLtHandle_t, int, int, int,
    const void*, const void*, const void*, const void*, __half*,
    void*, size_t, cudaStream_t);
extern "C" size_t apxinf_fp4_scale_buffer_bytes(int rows, int k);

static uint8_t f2e4m3(float v){ return (uint8_t)__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3); }
static float e4m3f(uint8_t b){ __nv_fp8_e4m3 v; v.__x=(__nv_fp8_storage_t)b; return (float)v; }
static uint8_t f2e2m1(float lo,float hi){ __nv_fp4x2_e2m1 v(make_float2(lo,hi)); return (uint8_t)v.__x; }
static float2 e2m1f(uint8_t p){ __nv_fp4x2_e2m1 v; v.__x=(__nv_fp4x2_storage_t)p; return (float2)v; }

static void swizzle_upload(const std::vector<uint8_t>& logical, int rows,int k,int bs,
                           std::vector<uint8_t>& out){
  int KB=(k+bs-1)/bs; size_t PS=512u*((KB+3)/4);
  out.assign(PS*((rows+127)/128),0);
  for(int r=0;r<rows;++r)for(int b=0;b<KB;++b)
    out[PS*(r/128)+512u*(b/4)+16u*(r%32)+4u*((r/32)%4)+(b%4)] = logical[b+r*(size_t)KB];
}

int main(){
  setvbuf(stdout,NULL,_IONBF,0);
  printf("stage0\n");
  cublasLtHandle_t lt; if(cublasLtCreate(&lt)!=CUBLAS_STATUS_SUCCESS){printf("handle fail\n");return 1;}
  int M=384,N=256,K=512,bs=16,KB=K/bs;
  std::vector<float> A((size_t)M*K),B((size_t)N*K);
  srand(7); for(auto&v:A)v=(rand()/(float)RAND_MAX-0.5f)*0.2f; for(auto&v:B)v=(rand()/(float)RAND_MAX-0.5f)*0.2f;

  std::vector<uint8_t> ap((size_t)M*K/2),bp((size_t)N*K/2),as,bsc,as_l((size_t)M*KB),bs_l((size_t)N*KB);
  for(int r=0;r<M;++r)for(int b=0;b<KB;++b){
    float amax=0; for(int j=0;j<bs;++j)amax=std::max(amax,std::fabs(A[r*K+b*bs+j]));
    uint8_t sb=f2e4m3(amax/6.0f); float sd=e4m3f(sb); if(sd<=0)sd=1; as_l[b+r*(size_t)KB]=sb;
    for(int j=0;j<bs;j+=2) ap[(size_t)(r*K+b*bs+j)/2]=f2e2m1(A[r*K+b*bs+j]/sd,A[r*K+b*bs+j+1]/sd);
  }
  for(int r=0;r<N;++r)for(int b=0;b<KB;++b){
    float amax=0; for(int j=0;j<bs;++j)amax=std::max(amax,std::fabs(B[r*K+b*bs+j]));
    uint8_t sb=f2e4m3(amax/6.0f); float sd=e4m3f(sb); if(sd<=0)sd=1; bs_l[b+r*(size_t)KB]=sb;
    for(int j=0;j<bs;j+=2) bp[(size_t)(r*K+b*bs+j)/2]=f2e2m1(B[r*K+b*bs+j]/sd,B[r*K+b*bs+j+1]/sd);
  }
  printf("stage1-quant-done\n");
  swizzle_upload(as_l,M,K,bs,as); swizzle_upload(bs_l,N,K,bs,bsc);

  printf("stage2-swizzle-done\n");
  void *dA,*dB,*dAS,*dBS,*dC,*ws; size_t wsB=64<<20;
  cudaMalloc(&dA,ap.size()); cudaMalloc(&dB,bp.size());
  cudaMalloc(&dAS,as.size()); cudaMalloc(&dBS,bsc.size());
  cudaMalloc(&dC,(size_t)M*N*2); cudaMalloc(&ws,wsB);
  cudaMemcpy(dA,ap.data(),ap.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,bp.data(),bp.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dAS,as.data(),as.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dBS,bsc.data(),bsc.size(),cudaMemcpyHostToDevice);

  printf("stage3-copies-done\n");
  cublasStatus_t st=apxinf_fp4_gemm_f16(lt,M,N,K,dA,dB,dAS,dBS,(__half*)dC,ws,wsB,0);
  printf("gemm status=%d\n",(int)st); if(st)return 2;
  std::vector<__half> C((size_t)M*N);
  cudaMemcpy(C.data(),dC,C.size()*2,cudaMemcpyDeviceToHost);

  double maxrel=0; std::vector<float> ar(K), br(K);
  for(int m=0;m<M;++m){
    for(int b=0;b<KB;++b){float sd=e4m3f(as_l[b+m*(size_t)KB]); if(sd<=0)sd=1;
      for(int j=0;j<bs;j+=2){float2 p=e2m1f(ap[(size_t)(m*K+b*bs+j)/2]);
        ar[b*bs+j]=p.x*sd; ar[b*bs+j+1]=p.y*sd;}}
    for(int n=0;n<N;++n){
      for(int b=0;b<KB;++b){float sd=e4m3f(bs_l[b+n*(size_t)KB]); if(sd<=0)sd=1;
        for(int j=0;j<bs;j+=2){float2 p=e2m1f(bp[(size_t)(n*K+b*bs+j)/2]);
          br[b*bs+j]=p.x*sd; br[b*bs+j+1]=p.y*sd;}}
      double acc=0; for(int k=0;k<K;++k)acc+=(double)ar[k]*br[k];
      double rel=std::fabs(acc-(float)C[(size_t)m+(size_t)n*M])/(std::fabs(acc)+1e-3);
      if(rel>maxrel)maxrel=rel;
    }
  }
  printf("ADAPTER VERIFY maxrel=%.6f  (%s)\n",maxrel,maxrel<1e-3?"PASS":"FAIL");
  printf("scale buffer bytes=%zu\n",apxinf_fp4_scale_buffer_bytes(M,K));
  return maxrel<1e-3?0:3;
}
