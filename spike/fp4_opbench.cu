// Deterministic real-shape NVFP4 quantize+adapter versus prepared BF16 GEMM.
// Inputs and weight preparation are outside timing. FP4 timing includes online
// F16 activation quantization and the row-major, tensor-scaled adapter call.
// CUDA event samples use the true median (average middle pair for even n).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cublasLt.h>

extern "C" cudaError_t apxinf_nvfp4_quantize_activation(
    const void*, void*, void*, int, int, cudaStream_t);
extern "C" cublasStatus_t apxinf_fp4_gemm_rowmajor_scaled_f16(
    int, int, int, const void*, const void*, const void*, const void*, float,
    __half*, void*, size_t, cudaStream_t);

#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"CUDA %s at line %d\n",cudaGetErrorString(e),__LINE__); std::exit(1); } } while(0)
#define LT(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr,"cuBLASLt status=%d at line %d\n",int(e),__LINE__); std::exit(1); } } while(0)

struct Buffer {
  void* p=nullptr;
  explicit Buffer(size_t n) { CK(cudaMalloc(&p,n)); }
  ~Buffer() { if(p) cudaFree(p); }
  Buffer(const Buffer&)=delete;
};
struct Shape { int n,k; const char* tag; };
static const Shape SHAPES[]={{1152,1152,"gemma-1152"},{3072,1024,"mlp-3072x1024"},
  {4096,1024,"mlp-4096x1024"},{16384,2048,"big-16384x2048"},
  {2048,2048,"attn-2048x2048"},{4304,1152,"geglu-4304x1152"}};
static const int MS[]={522,778,2048};
static constexpr size_t WORKSPACE=256ull<<20;

static float value(int row,int col,bool weight) {
  if((row*5+col/16)%29==0) return 0.f; // Include exact zero blocks.
  int integer=(row*17+col*13+(weight?11:3))%33-16;
  return integer/(weight?32.f:16.f);   // Exactly representable in F16 and BF16.
}
static size_t scale_size(int rows,int k) {
  return 512ull*((k/16+3)/4)*((rows+127)/128);
}
static size_t scale_offset(int row,int block,int k) {
  size_t ps=512ull*((k/16+3)/4);
  return ps*(row/128)+512ull*(block/4)+16ull*(row%32)+4ull*((row/32)%4)+block%4;
}
static float decode_scale(uint8_t bits) {
  __nv_fp8_e4m3 fp;fp.__x=bits;return float(fp);
}
struct HostQuantized {
  int rows,k;float tensor_scale;
  std::vector<uint8_t> packed,scales;
  HostQuantized(int r,int width,bool weight,float tau):rows(r),k(width),tensor_scale(tau),
      packed(size_t(r)*width/2),scales(scale_size(r,width),0) {
    for(int row=0;row<rows;++row) for(int b=0;b<k/16;++b) {
      float maximum=0;
      for(int j=0;j<16;++j)maximum=std::max(maximum,std::fabs(value(row,b*16+j,weight)/tau));
      uint8_t code=uint8_t(__nv_cvt_float_to_fp8(maximum/6.f,__NV_SATFINITE,__NV_E4M3));
      scales[scale_offset(row,b,k)]=code;
      float scale=decode_scale(code);if(scale==0)scale=1;
      for(int j=0;j<16;j+=2) {
        int col=b*16+j;
        __nv_fp4x2_e2m1 pair(make_float2(value(row,col,weight)/(tau*scale),
                                      value(row,col+1,weight)/(tau*scale)));
        packed[(size_t(row)*k+col)/2]=uint8_t(pair.__x);
      }
    }
  }
  double decoded(int row,int col) const {
    static const float grid[]={0,.5f,1,1.5f,2,3,4,6};
    unsigned byte=packed[(size_t(row)*k+col)/2];
    unsigned code=(byte>>(4*(col%2)))&15;
    double magnitude=grid[code&7]*double(decode_scale(scales[scale_offset(row,col/16,k)]))*tensor_scale;
    return code&8?-magnitude:magnitude;
  }
};

static double median(std::vector<double> values) {
  if(values.empty())throw std::runtime_error("No timing samples");
  std::sort(values.begin(),values.end());size_t mid=values.size()/2;
  return values.size()%2?values[mid]:(values[mid-1]+values[mid])/2;
}

template<class Call> static double measure(Call call,int warmup,int samples) {
  for(int i=0;i<warmup;++i)call();
  CK(cudaDeviceSynchronize());
  cudaEvent_t start,stop;CK(cudaEventCreate(&start));CK(cudaEventCreate(&stop));
  std::vector<double> times;
  for(int i=0;i<samples;++i) {
    CK(cudaEventRecord(start));call();CK(cudaEventRecord(stop));CK(cudaEventSynchronize(stop));
    float ms=0;CK(cudaEventElapsedTime(&ms,start,stop));
    if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("Invalid CUDA elapsed time");
    times.push_back(double(ms)*1000);
  }
  CK(cudaEventDestroy(start));CK(cudaEventDestroy(stop));return median(times);
}

struct Bf16Gemm {
  cublasLtHandle_t lt;
  cublasLtMatmulDesc_t op=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  cublasLtMatmulPreference_t pref=nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int algorithms=0;
  cublasStatus_t heuristic_status;
  Bf16Gemm(cublasLtHandle_t handle,int m,int n,int k):lt(handle) {
    LT(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t transpose=CUBLAS_OP_T,normal=CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
    LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)));
    // C^T=W X^T, column-major [n,m] has row-major [m,n] bytes.
    LT(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,k));
    LT(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,k));
    LT(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n));
    LT(cublasLtMatmulPreferenceCreate(&pref));size_t size=WORKSPACE;
    LT(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&size,sizeof(size)));
    heuristic_status=cublasLtMatmulAlgoGetHeuristic(lt,op,a,b,c,c,pref,1,&heuristic,&algorithms);
  }
  ~Bf16Gemm() {
    if(pref)cublasLtMatmulPreferenceDestroy(pref);
    if(c)cublasLtMatrixLayoutDestroy(c);if(b)cublasLtMatrixLayoutDestroy(b);if(a)cublasLtMatrixLayoutDestroy(a);
    if(op)cublasLtMatmulDescDestroy(op);
  }
  void call(const void* weight,const void* input,void* output,void* workspace) const {
    float one=1,zero=0;
    LT(cublasLtMatmul(lt,op,&one,weight,a,input,b,&zero,output,c,output,c,
                     &heuristic.algo,workspace,WORKSPACE,0));
  }
};

static void compare_activation_bytes(const HostQuantized& expected,const Buffer& packed,const Buffer& scales) {
  std::vector<uint8_t> data(expected.packed.size()),scale(expected.scales.size());
  CK(cudaMemcpy(data.data(),packed.p,data.size(),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(scale.data(),scales.p,scale.size(),cudaMemcpyDeviceToHost));
  if(data!=expected.packed||scale!=expected.scales)throw std::runtime_error("Online activation encoding differs from CUDA-intrinsic CPU reference");
}
static void sampled_output_checks(const Shape& shape,int m,const HostQuantized& a,const HostQuantized& b,
                                  const Buffer& fp4,const Buffer& bf16,double& max_fp4,double& max_bf16) {
  std::vector<std::pair<int,int>> locations={{0,0},{0,shape.n-1},{m-1,0},{m-1,shape.n-1},{m/2,shape.n/2}};
  for(int i=0;i<16;++i)locations.push_back({(i*73+19)%m,(i*179+23)%shape.n});
  max_fp4=max_bf16=0;
  for(auto location:locations) {
    int row=location.first,col=location.second;double ref_fp4=0,ref_bf16=0;
    for(int k=0;k<shape.k;++k) {
      ref_fp4+=a.decoded(row,k)*b.decoded(col,k);
      ref_bf16+=double(value(row,k,false))*value(col,k,true);
    }
    __half observed_fp4;__nv_bfloat16 observed_bf16;
    CK(cudaMemcpy(&observed_fp4,static_cast<const __half*>(fp4.p)+size_t(row)*shape.n+col,2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&observed_bf16,static_cast<const __nv_bfloat16*>(bf16.p)+size_t(row)*shape.n+col,2,cudaMemcpyDeviceToHost));
    double expected_fp4=__half2float(__float2half(float(ref_fp4)));
    double expected_bf16=__bfloat162float(__float2bfloat16(float(ref_bf16)));
    double got_fp4=__half2float(observed_fp4),got_bf16=__bfloat162float(observed_bf16);
    double error_fp4=std::fabs(got_fp4-expected_fp4),error_bf16=std::fabs(got_bf16-expected_bf16);
    if(!std::isfinite(got_fp4)||!std::isfinite(got_bf16)||
       error_fp4>1e-3+1e-3*std::fabs(expected_fp4)||error_bf16>1e-2+1e-2*std::fabs(expected_bf16))
      throw std::runtime_error("Sampled row-major decoded-input GEMM reference failed");
    max_fp4=std::max(max_fp4,error_fp4);max_bf16=std::max(max_bf16,error_bf16);
  }
}

int main(int argc,char** argv) {
 try {
  int samples=30,warmup=10,only_m=0;std::string only_shape;bool verify_only=false,cpu_only=false;
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];
    if(arg=="--verify-only")verify_only=true;
    else if(arg=="--cpu-self-test")cpu_only=true;
    else if(arg.rfind("--samples=",0)==0)samples=std::stoi(arg.substr(10));
    else if(arg.rfind("--warmup=",0)==0)warmup=std::stoi(arg.substr(9));
    else if(arg.rfind("--m=",0)==0)only_m=std::stoi(arg.substr(4));
    else if(arg.rfind("--shape=",0)==0)only_shape=arg.substr(8);
    else throw std::runtime_error("Unknown argument: "+arg);
  }
  if(samples<1||warmup<0||only_m<0)throw std::runtime_error("Invalid sample/warmup/M count");
  if(cpu_only) {
    if(median({4,1,3,2})!=2.5||median({9,1,4})!=4)throw std::runtime_error("Median self-test failed");
    HostQuantized q(17,64,true,.25f);
    for(int r=0;r<17;++r)for(int k=0;k<64;++k)if(!std::isfinite(q.decoded(r,k)))throw std::runtime_error("Host encoding failed");
    printf("CPU self-test PASS: true median, finite deterministic encoding; no CUDA API called\n");return 0;
  }
  int selected=0;
  for(auto& shape:SHAPES)for(int m:MS)if((only_shape.empty()||only_shape==shape.tag)&&(!only_m||only_m==m))++selected;
  if(!selected)throw std::runtime_error("Shape/M filter matches no declared benchmark case");
  cudaDeviceProp device;CK(cudaGetDeviceProperties(&device,0));
  int runtime=0,driver=0;CK(cudaRuntimeGetVersion(&runtime));CK(cudaDriverGetVersion(&driver));
  fprintf(stderr,"# gpu=%s sm_%d%d driver=%d runtime=%d cublasLt=%zu\n",
          device.name,device.major,device.minor,driver,runtime,cublasLtGetVersion());
  cublasLtHandle_t lt;LT(cublasLtCreate(&lt));Buffer workspace(WORKSPACE);
  fprintf(stderr,"# deterministic_inputs=v1 weight_tensor_scale=0.25 output=row_major "
                 "timing=CUDA_event_true_median samples=%d warmup=%d\n",samples,warmup);
  fprintf(stderr,"# fp4=online_activation_quantize+rowmajor_scaled_adapter; bf16=prepared_GEMM; "
                 "preparation_and_reference_checks_excluded; per_case=all_activation_bytes+21_sampled_outputs\n");
  printf("shape,M,N,K,samples,warmup,weight_tensor_scale,fp4_us_p50,bf16_us_p50,speedup,fp4_check_max_abs,bf16_check_max_abs,status\n");
  int completed=0,unavailable=0;
  for(auto& shape:SHAPES)for(int m:MS) {
    if((!only_shape.empty()&&only_shape!=shape.tag)||(only_m&&only_m!=m))continue;
    int n=shape.n,k=shape.k;size_t input_count=size_t(m)*k,weight_count=size_t(n)*k;
    HostQuantized aq(m,k,false,1.f),bq(n,k,true,.25f);
    std::vector<__half> input_f16(input_count);
    std::vector<__nv_bfloat16> input_bf16(input_count),weight_bf16(weight_count);
    for(int row=0;row<m;++row)for(int col=0;col<k;++col) {
      float v=value(row,col,false);input_f16[size_t(row)*k+col]=__float2half(v);
      input_bf16[size_t(row)*k+col]=__float2bfloat16(v);
    }
    for(int row=0;row<n;++row)for(int col=0;col<k;++col)
      weight_bf16[size_t(row)*k+col]=__float2bfloat16(value(row,col,true));
    Buffer x16(input_count*2),xb(input_count*2),wb(weight_count*2),ap(input_count/2),bp(weight_count/2),
           asw(aq.scales.size()),bsw(bq.scales.size()),c4(size_t(m)*n*2),cb(size_t(m)*n*2);
    CK(cudaMemcpy(x16.p,input_f16.data(),input_count*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xb.p,input_bf16.data(),input_count*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(wb.p,weight_bf16.data(),weight_count*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bp.p,bq.packed.data(),bq.packed.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bsw.p,bq.scales.data(),bq.scales.size(),cudaMemcpyHostToDevice));
    CK(cudaMemset(asw.p,0,aq.scales.size()));
    Bf16Gemm bf16(lt,m,n,k);
    auto fp4_call=[&]() {
      CK(apxinf_nvfp4_quantize_activation(x16.p,ap.p,asw.p,m,k,0));
      return apxinf_fp4_gemm_rowmajor_scaled_f16(m,n,k,ap.p,bp.p,asw.p,bsw.p,.25f,
                  static_cast<__half*>(c4.p),workspace.p,WORKSPACE,0);
    };
    auto fp4_status=fp4_call();CK(cudaDeviceSynchronize());
    if(fp4_status!=CUBLAS_STATUS_SUCCESS && fp4_status!=CUBLAS_STATUS_NOT_SUPPORTED)LT(fp4_status);
    if(bf16.heuristic_status!=CUBLAS_STATUS_SUCCESS && bf16.heuristic_status!=CUBLAS_STATUS_NOT_SUPPORTED)
      LT(bf16.heuristic_status);
    if(fp4_status==CUBLAS_STATUS_NOT_SUPPORTED || bf16.heuristic_status!=CUBLAS_STATUS_SUCCESS || bf16.algorithms==0) {
      fprintf(stderr,"# unavailable shape=%s M=%d fp4_status=%d bf16_heuristic=%d bf16_algorithms=%d\n",
              shape.tag,m,int(fp4_status),int(bf16.heuristic_status),bf16.algorithms);
      printf("%s,%d,%d,%d,%d,%d,0.25,,,,,,unavailable\n",shape.tag,m,n,k,samples,warmup);++unavailable;continue;
    }
    LT(fp4_status);
    auto bf16_call=[&]() {bf16.call(wb.p,xb.p,cb.p,workspace.p);};
    bf16_call();CK(cudaDeviceSynchronize());
    compare_activation_bytes(aq,ap,asw);
    double error4,errorb;sampled_output_checks(shape,m,aq,bq,c4,cb,error4,errorb);
    if(verify_only)printf("%s,%d,%d,%d,0,0,0.25,,,,%.9g,%.9g,verified_samples\n",shape.tag,m,n,k,error4,errorb);
    else {
      double t4=measure([&]() {LT(fp4_call());},warmup,samples);
      double tb=measure(bf16_call,warmup,samples);
      printf("%s,%d,%d,%d,%d,%d,0.25,%.6f,%.6f,%.6f,%.9g,%.9g,ok\n",
              shape.tag,m,n,k,samples,warmup,t4,tb,tb/t4,error4,errorb);
    }
    fflush(stdout);++completed;
  }
  LT(cublasLtDestroy(lt));
  fprintf(stderr,"# completed=%d unavailable=%d; unavailable cases have no timing or speedup\n",completed,unavailable);
  return completed?0:3;
 } catch(const std::exception& e) {fprintf(stderr,"ERROR: %s\n",e.what());return 2;}
}
