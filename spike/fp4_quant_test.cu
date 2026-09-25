// fp4_quant_test.cu — functional test for apxinf_nvfp4_quantize_activation on sm_120.
// Random f16 matrix -> kernel -> CPU reference (spike-verified codecs) -> byte compare.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

extern "C" cudaError_t apxinf_nvfp4_quantize_activation(
    const void* x, void* packed, void* scale_swz, int rows, int k, cudaStream_t stream);

static uint8_t f2e4m3(float v){ return (uint8_t)__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3); }
static float e4m3f(uint8_t b){ __nv_fp8_e4m3 v; v.__x=(__nv_fp8_storage_t)b; return (float)v; }
static float2 e2m1f(uint8_t p){ __nv_fp4x2_e2m1 v; v.__x=(__nv_fp4x2_storage_t)p; return (float2)v; }

int main(){
  setvbuf(stdout, nullptr, _IONBF, 0);
  int rows=300, K=512, bs=16, KB=K/bs;
  srand(11);
  std::vector<__half> x((size_t)rows*K);
  for(auto&v:x) v=__float2half(((rand()/(float)RAND_MAX)-0.5f)*0.4f);

  // CPU reference (bit-exact codecs from spike)
  std::vector<uint8_t> ref_p((size_t)rows*K/2), ref_s((size_t)rows*KB);
  for(int r=0;r<rows;++r)for(int b=0;b<KB;++b){
    float amax=0;
    for(int j=0;j<bs;++j) amax=std::max(amax,std::fabs(__half2float(x[r*K+b*bs+j])));
    uint8_t sb=f2e4m3(amax/6.0f); float sd=e4m3f(sb); if(sd<=0)sd=1;
    ref_s[b+r*(size_t)KB]=sb;
    for(int j=0;j<bs;j+=2){
      uint8_t lo=0,hi=0;
      { float q=__half2float(x[r*K+b*bs+j])/sd;
        __nv_fp4x2_e2m1 dummy; (void)dummy;
        // encode via ctor pair for exactness
      }
      // use ctor path:
      float q0=__half2float(x[r*K+b*bs+j])/sd, q1=__half2float(x[r*K+b*bs+j+1])/sd;
      __nv_fp4x2_e2m1 pair(make_float2(q0,q1));
      uint8_t byte=(uint8_t)pair.__x;
      ref_p[(size_t)(r*K+b*bs+j)/2]=byte; (void)lo;(void)hi;
    }
  }
  // reference swizzle
  size_t PS=512ull*((KB+3)/4);
  std::vector<uint8_t> ref_swz(PS*((rows+127)/128),0);
  for(int r=0;r<rows;++r)for(int b=0;b<KB;++b)
    ref_swz[PS*(r/128)+512ull*(b/4)+16ull*(r%32)+4ull*((r/32)%4)+(b%4)]=ref_s[b+r*(size_t)KB];

  void *dx,*dp,*ds;
  cudaMalloc(&dx, x.size()*2); cudaMalloc(&dp, ref_p.size()); cudaMalloc(&ds, ref_swz.size());
  cudaMemcpy(dx, x.data(), x.size()*2, cudaMemcpyHostToDevice);
  cudaError_t e=apxinf_nvfp4_quantize_activation(dx,dp,ds,rows,K,0);
  printf("kernel=%s\n", cudaGetErrorString(e)); if(e) return 1;
  std::vector<uint8_t> got_p(ref_p.size()), got_s(ref_swz.size());
  cudaMemcpy(got_p.data(),dp,ref_p.size(),cudaMemcpyDeviceToHost);
  cudaMemcpy(got_s.data(),ds,ref_swz.size(),cudaMemcpyDeviceToHost);

  int pbad=0,sbad=0;
  for(size_t i=0;i<ref_p.size();++i) if(got_p[i]!=ref_p[i]) pbad++;
  for(size_t i=0;i<ref_swz.size();++i) if(got_s[i]!=ref_swz[i]) sbad++;
  printf("packed mismatches: %d/%zu   scale mismatches: %d/%zu\n",
         pbad, ref_p.size(), sbad, ref_swz.size());
  bool pass = pbad==0 && sbad==0;
  printf("QUANTIZE KERNEL %s\n", pass?"PASS":"FAIL");
  return pass?0:2;
}
