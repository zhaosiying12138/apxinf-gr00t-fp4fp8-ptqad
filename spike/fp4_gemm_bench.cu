// ============================================================================
// fp4_gemm_bench.cu — FP4-VLA spike (go/no-go gate)
//
// Question: does block-scaled NVFP4 GEMM (E2M1 data + per-16 E4M3 scales)
// work on a *GeForce* Blackwell (sm_120, RTX 5090 Laptop) via cuBLASLt,
// and how does it compare to FP8 (E4M3, per-tensor) and BF16?
//
// Layout used (cuBLASLt "TN"): A stored MxK K-major, B stored NxK K-major,
// C stored MxN col-major.  C[m,n] = sum_k A[m,k]*B[n,k].
//   Adesc = (rows=K, cols=M, ld=K, ORDER_COL, opA=T)
//   Bdesc = (rows=K, cols=N, ld=K, ORDER_COL, opB=T)
//   Cdesc = (rows=M, cols=N, ld=M, ORDER_COL)
// Outer scale mode: scaleA is ceil(K/16) x M col-major, scaleB ceil(K/16) x N.
//
// Outputs CSV to stdout.  --verify cross-checks against a CPU fp64 reference
// on a small shape by dequantizing exactly what was uploaded to the GPU.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cublasLt.h>

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)
#define CHECK_LT(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cublasLt error %d at %s:%d\n",(int)s,__FILE__,__LINE__); exit(1);} } while(0)

// ---------------- host-side format converters ----------------
static inline uint8_t f32_to_e4m3(float v) {
  return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
static inline float e4m3_to_f32(uint8_t b) {  // CUDA 13: free fn removed, use type cast
  __nv_fp8_e4m3 v; v.__x = (__nv_fp8_storage_t)b;
  return (float)v;
}
static inline uint8_t f2_to_e2m1(float lo, float hi) {
  __nv_fp4x2_e2m1 v(make_float2(lo, hi));   // CUDA 13: ctor + round mode API
  return (uint8_t)v.__x;
}
static inline float2 e2m1_to_f2(uint8_t p) {
  __nv_fp4x2_e2m1 v; v.__x = (__nv_fp4x2_storage_t)p;
  return (float2)v;
}
static inline uint8_t f32_to_ue8m0(float v) {  // MX scale: 2^(b-127), no zero/NaN handling for bench
  int e = (int)std::floor(std::log2(std::max(v, 1e-30f)) + 0.5f) + 127;
  e = std::clamp(e, 1, 254);
  return (uint8_t)e;
}
static inline float ue8m0_to_f32(uint8_t b) {
  return std::exp2f((float)(int)b - 127.0f);
}
static inline float e2m1_max() { return 6.0f; }
static inline float e4m3_max() { return 448.0f; }

// ---------------- quantized tensor containers ----------------
enum class Dt { BF16, FP8, NVFP4, MXFP4 };
static const char* dt_name(Dt d) {
  switch(d){case Dt::BF16:return "bf16";case Dt::FP8:return "fp8";case Dt::NVFP4:return "nvfp4";case Dt::MXFP4:return "mxfp4";}
  return "?";
}

struct Tensor {  // row-major rows x K host buffer of raw fp32 + quantized forms
  int rows = 0, K = 0;
  std::vector<float> f32;        // rows*K
  // bf16
  std::vector<__nv_bfloat16> bf16;
  // fp8 per-tensor
  std::vector<uint8_t> fp8; float fp8_scale = 1.f;
  // fp4 (nvfp4: block16 e4m3; mxfp4: block32 ue8m0)
  std::vector<uint8_t> fp4;          // packed pairs, rows*K/2 bytes
  std::vector<uint8_t> fp4_scale;    // ceil(K/bs) x rows col-major
  int block = 16;
};

static void quantize(Tensor& t, Dt dt, int rows, int K, const float* data) {
  t.rows = rows; t.K = K; t.f32.assign(data, data + (size_t)rows*K);
  if (dt == Dt::BF16) {
    t.bf16.resize((size_t)rows*K);
    for (size_t i=0;i<t.f32.size();++i) t.bf16[i] = __float2bfloat16(t.f32[i]);
    return;
  }
  if (dt == Dt::FP8) {
    float amax=0; for(float v:t.f32) amax=std::max(amax,std::fabs(v));
    t.fp8_scale = amax>0 ? amax/e4m3_max() : 1.f;
    t.fp8.resize(t.f32.size());
    for(size_t i=0;i<t.f32.size();++i) t.fp8[i]=f32_to_e4m3(t.f32[i]/t.fp8_scale);
    return;
  }
  int bs = (dt==Dt::NVFP4)?16:32; t.block=bs;
  int kb = (K+bs-1)/bs;
  t.fp4.assign((size_t)rows*K/2, 0);
  t.fp4_scale.assign((size_t)kb*rows, 0);
  bool flat = getenv("FP4_FLAT_SCALE");  // probe: all scales = 1.0 -> tests data packing alone
  for(int r=0;r<rows;++r){
    for(int b=0;b<kb;++b){
      float amax=0;
      for(int j=0;j<bs;++j){ int idx=r*K+b*bs+j; amax=std::max(amax,std::fabs(t.f32[idx])); }
      uint8_t sb; float sd;
      if(dt==Dt::NVFP4){ sb=f32_to_e4m3(amax/e2m1_max()); sd=e4m3_to_f32(sb); }
      else { sb=f32_to_ue8m0(amax/e2m1_max()); sd=ue8m0_to_f32(sb); }
      if(!(sd>0)) sd=1.f;
      if(flat){ sb=f32_to_e4m3(1.0f); sd=1.0f; }
      t.fp4_scale[(size_t)b + (size_t)r*kb] = sb;
      for(int j=0;j<bs;j+=2){
        int idx=r*K+b*bs+j;
        uint8_t p=f2_to_e2m1(t.f32[idx]/sd, t.f32[idx+1]/sd);
        t.fp4[(size_t)idx/2]=p;
      }
    }
  }
}
// exact dequantization of what the GPU sees (for the CPU reference)
static void dequant_row(const Tensor& t, Dt dt, int r, std::vector<float>& out) {
  out.resize(t.K);
  if(dt==Dt::BF16){ for(int k=0;k<t.K;++k) out[k]=__bfloat162float(t.bf16[(size_t)r*t.K+k]); return; }
  if(dt==Dt::FP8){ for(int k=0;k<t.K;++k) out[k]=e4m3_to_f32(t.fp8[(size_t)r*t.K+k])*t.fp8_scale; return; }
  int kb=(t.K+t.block-1)/t.block;
  for(int b=0;b<kb;++b){
    float sd = (dt==Dt::NVFP4)? e4m3_to_f32(t.fp4_scale[(size_t)b+(size_t)r*kb])
                              : ue8m0_to_f32(t.fp4_scale[(size_t)b+(size_t)r*kb]);
    for(int j=0;j<t.block;++j){
      int idx=r*t.K+b*t.block+j; if(idx>=(int)t.f32.size()) break;
      uint8_t p=t.fp4[(size_t)idx/2];
      float2 pr=e2m1_to_f2(p);
      out[b*t.block+j] = (j%2==0? pr.x: pr.y)*sd;
    }
  }
}

// ---------------- cuBLASLt harness ----------------
struct BenchResult { bool ok=false; int algos=0; float ms=0, tflops=0, gbps=0; std::string note; };

static cublasLtHandle_t g_lt;  // CUDA 13 cuBLASLt requires a handle for matmul/heuristic

// candidate layouts for the FP4 block-scale tensor upload (see --slayout):
//   row   : per-row contiguous        idx = m*KB + kb            (== col-major KB x M)
//   block : per-block across rows     idx = kb*M + m             (== col-major M x KB)
//   row4  : row, KB padded to mult of 4                            (docs hint "MN x K4")
//   block4: block, M padded to mult of 4
static int g_slayout = 0;  // 0=row 1=block 2=row4 3=block4 4=hw-swizzle
static const char* slayout_name(int i){static const char* n[]={"row","block","row4","block4","hw"};return n[i];}

static size_t scale_upload(const Tensor& t, int rows, int K, std::vector<uint8_t>& out) {
  int KB=(K+t.block-1)/t.block;
  out.clear();
  if(getenv("FP4_FLAT_SCALE")){          // probe mode: layout-independent upload
    out.assign((size_t)rows*KB, f32_to_e4m3(1.0f));
    return out.size();
  }
  if(g_slayout==0 || g_slayout==2){
    int KBp = (g_slayout==2)? ((KB+3)/4)*4 : KB;
    out.assign((size_t)rows*KBp, 0);
    for(int r=0;r<rows;++r) for(int b=0;b<KB;++b)
      out[(size_t)r*KBp+b] = t.fp4_scale[(size_t)b+(size_t)r*KB];
  } else if(g_slayout==4){
    // hardware swizzle, fully decoded by single-byte slot probing (see docs/spike-notes.md):
    //   offset(row r, block b) = PS*(r/128) + 512*(b/4) + 16*(r%32) + 4*((r/32)%4) + (b%4)
    // where PS = 512*ceil(KB/4).  AKA NVIDIA "128x4 tiled, rows interleaved in 32-row chunks".
    size_t PS = (size_t)512*((KB+3)/4);
    size_t need = PS*((rows+127)/128);
    out.assign(need, 0);
    for(int r=0;r<rows;++r) for(int b=0;b<KB;++b)
      out[PS*(r/128) + (size_t)512*(b/4) + (size_t)16*(r%32) + (size_t)4*((r/32)%4) + (b%4)]
        = t.fp4_scale[(size_t)b+(size_t)r*KB];
  } else if(g_slayout==5){
    // raw passthrough: t.fp4_scale IS the physical buffer (for slot probing)
    out.assign(t.fp4_scale.begin(), t.fp4_scale.end());
  } else {
    int Mp = (g_slayout==3)? ((rows+3)/4)*4 : rows;
    out.assign((size_t)Mp*KB, 0);
    for(int r=0;r<rows;++r) for(int b=0;b<KB;++b)
      out[(size_t)b*Mp+r] = t.fp4_scale[(size_t)b+(size_t)r*KB];
  }
  return out.size();
}

// Probe 2: pin data to 1.0 everywhere, all scales 1.0 except ONE buffer slot = 2.0.
// The location & size of the anomaly in C reveals the scale mapping the GPU actually uses.
//   buffer written in "row" layout (m*KB+kb); slot 1 is (m=0,kb=1) under row semantics.
static void apply_probe(Tensor& A, Tensor& B, int M, int N, int K, int slot) {
  std::fill(A.fp4.begin(), A.fp4.end(), 0x22);  // e2m1 pair (1.0, 1.0)
  std::fill(B.fp4.begin(), B.fp4.end(), 0x22);
  int KB=(K+A.block-1)/A.block;
  A.fp4_scale.assign((size_t)M*KB, f32_to_e4m3(1.0f));
  B.fp4_scale.assign((size_t)N*KB, f32_to_e4m3(1.0f));
  A.fp4_scale[slot] = f32_to_e4m3(2.0f);
  for(auto&v:A.f32) v=1.f; for(auto&v:B.f32) v=1.f;
}
static void dump_probe(const char* tag, int M, int N, int K, const float* C, int slot) {
  // expected baseline C[m,n]=K; find rows deviating
  fprintf(stderr, "# probe %s slot=%d K=%d\n", tag, slot, K);
  for(int m=0;m<4;++m) fprintf(stderr, "#   C[%d,0]=%.1f C[%d,1]=%.1f\n", m, C[(size_t)m], m, C[(size_t)m+(size_t)N*0+1]);
  int firstBad=-1; float maxDev=0;
  for(int m=0;m<M && firstBad<0;++m) if(std::fabs(C[(size_t)m]-K)>0.5f) firstBad=m;
  for(size_t i=0;i<(size_t)M*N;++i){ float d=std::fabs(C[i]-K); if(d>maxDev) maxDev=d; }
  int nBad=0; for(size_t i=0;i<(size_t)M*N;++i) if(std::fabs(C[i]-K)>0.5f) nBad++;
  fprintf(stderr, "#   firstBadRow=%d maxDev=%.1f nBadCells=%d (nBad==N -> whole-row anomaly; ==M -> whole-col)\n",
          firstBad, maxDev, nBad);
}

// Probe 3 (MAP): full decode of the scale mapping.  A data all 1.0, A scale(m,kb)=1+0.5*kb
// (same every row); B has N=16 rows, row n has data 1.0 ONLY in block n.  Then the ideal
// C[m,n] = 16 * s_A(m, n)  -> prints exactly which scale the GPU pairs with block n.
static void apply_map(Tensor& A, Tensor& B, int M, int N, int K) {
  int KB=(K+A.block-1)/A.block;
  std::fill(A.fp4.begin(), A.fp4.end(), 0x22);
  A.fp4_scale.assign((size_t)M*KB, 0);
  for(int r=0;r<M;++r) for(int b=0;b<KB;++b) A.fp4_scale[(size_t)r*KB+b]=f32_to_e4m3((float)(r+1)+0.5f*b);
  B.fp4.assign((size_t)N*K/2, 0x00);
  B.fp4_scale.assign((size_t)N*KB, f32_to_e4m3(1.0f));
  for(int n=0;n<N && n<KB;++n){ int base=n*K + n*A.block; for(int j=0;j<A.block;j+=2) B.fp4[(size_t)(base+j)/2]=0x22; }
  for(auto&v:A.f32) v=1.f; for(auto&v:B.f32) v=0.f;
}
static void dump_map(int M, int N, int K, const float* C) {
  fprintf(stderr, "# map: each cell = scale value the GPU paired with block n of row m\n");
  int rows[] = {0,1,2,3,4,5,15,16,17,31,32,33,63,64,100,127,128,129,M-1};
  for(auto m : rows){ if(m<0||m>=M) continue;
    fprintf(stderr, "#   m=%d: ", m);
    for(int n=0;n<N;++n) fprintf(stderr, "%.1f ", C[(size_t)m+(size_t)n*M]/16.0f);
    fprintf(stderr, "\n");
  }
}

static BenchResult run_lt(Dt dt, int M, int N, int K, const Tensor& A, const Tensor& B,
                          void* dA, void* dB, void* dC, void* dAs, void* dBs, float* dS8A, float* dS8B,
                          void* workspace, size_t wsSize, int warmup, int iters, bool verify)
{
  BenchResult R;
  cudaDataType ta, tb, tc; cublasComputeType_t ct = CUBLAS_COMPUTE_32F;
  if(dt==Dt::BF16){ ta=tb=CUDA_R_16BF; tc=CUDA_R_32F; }
  else if(dt==Dt::FP8){ ta=tb=CUDA_R_8F_E4M3; tc=CUDA_R_32F; }
  else { ta=tb=CUDA_R_4F_E2M1; tc=CUDA_R_32F; }

  cublasLtMatmulDesc_t op; CHECK_LT(cublasLtMatmulDescCreate(&op, ct, CUDA_R_32F));
  // classic "TN": A stored KxM col-major (K contiguous) with opA=T -> MxK;
  //              B stored KxN col-major (K contiguous) with opB=N -> KxN.
  //              Both operands K-major, which is what block-scaled FP4 requires.
  cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
  CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  if(dt==Dt::FP8){
    CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dS8A, sizeof(dS8A)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dS8B, sizeof(dS8B)));
  } else if(dt==Dt::NVFP4 || dt==Dt::MXFP4){
    cublasLtMatmulMatrixScale_t mode = (dt==Dt::NVFP4)
      ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3     // CUDA 13: unified vec16 mode
      : CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasStatus_t sa = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
    cublasStatus_t sb = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
    if(sa!=CUBLAS_STATUS_SUCCESS || sb!=CUBLAS_STATUS_SUCCESS){ R.note="scale-mode attr rejected"; return R; }
    CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dAs, sizeof(dAs)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dBs, sizeof(dBs)));
    (void)dS8A; (void)dS8B;
  }

  cublasLtMatrixLayout_t LA, LB, LC;  // CUDA 13: layouts are column-major only (no ORDER arg)
  CHECK_LT(cublasLtMatrixLayoutCreate(&LA, ta, K, M, K));
  CHECK_LT(cublasLtMatrixLayoutCreate(&LB, tb, K, N, K));
  CHECK_LT(cublasLtMatrixLayoutCreate(&LC, tc, M, N, M));

  cublasLtMatmulPreference_t pref; CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));
  cublasLtMatmulHeuristicResult_t heur[4]; int nres=0;
  cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_lt, op, LA, LB, LC, LC, pref, 4, heur, &nres);
  cublasLtMatmulPreferenceDestroy(pref);
  if(hs!=CUBLAS_STATUS_SUCCESS || nres==0){
    R.note = "heuristic: " + std::to_string((int)hs) + " algos=0";
    cublasLtMatrixLayoutDestroy(LA); cublasLtMatrixLayoutDestroy(LB);
    cublasLtMatrixLayoutDestroy(LC); cublasLtMatmulDescDestroy(op);
    return R;
  }
  R.algos = nres;

  // upload
  size_t aBytes = (dt==Dt::BF16)? sizeof(__nv_bfloat16)*A.f32.size()
               : (dt==Dt::FP8)  ? A.f32.size()
               : (size_t)A.rows*K/2;
  size_t bBytes = (dt==Dt::BF16)? sizeof(__nv_bfloat16)*B.f32.size()
               : (dt==Dt::FP8)  ? B.f32.size()
               : (size_t)B.rows*K/2;
  const void* hA = (dt==Dt::BF16)? (const void*)A.bf16.data() : (dt==Dt::FP8)? (const void*)A.fp8.data() : (const void*)A.fp4.data();
  const void* hB = (dt==Dt::BF16)? (const void*)B.bf16.data() : (dt==Dt::FP8)? (const void*)B.fp8.data() : (const void*)B.fp4.data();
  CHECK_CUDA(cudaMemcpy(dA, hA, aBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB, bBytes, cudaMemcpyHostToDevice));
  if(dt==Dt::NVFP4 || dt==Dt::MXFP4){
    static std::vector<uint8_t> sa, sb;
    CHECK_CUDA(cudaMemcpy(dAs, sa.data(), scale_upload(A,M,K,sa), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBs, sb.data(), scale_upload(B,N,K,sb), cudaMemcpyHostToDevice));
  }
  if(dt==Dt::FP8){
    CHECK_CUDA(cudaMemcpy(dS8A, &A.fp8_scale, 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dS8B, &B.fp8_scale, 4, cudaMemcpyHostToDevice));
  }
  float one=1.f, zero=0.f;

  auto call=[&](){
    CHECK_LT(cublasLtMatmul(g_lt, op, &one, dA, LA, dB, LB, &zero, dC, LC, dC, LC,
                            &heur[0].algo, workspace, wsSize, nullptr /*stream 0*/));
  };
  for(int i=0;i<warmup;++i) call();
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t e0,e1; CHECK_CUDA(cudaEventCreate(&e0)); CHECK_CUDA(cudaEventCreate(&e1));
  CHECK_CUDA(cudaEventRecord(e0));
  for(int i=0;i<iters;++i) call();
  CHECK_CUDA(cudaEventRecord(e1)); CHECK_CUDA(cudaEventSynchronize(e1));
  float ms; CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1)); ms/=iters;
  R.ms=ms; R.tflops = 2.0*M*N*K/(ms*1e-3)/1e12;
  double abytes = aBytes + ((dt==Dt::NVFP4||dt==Dt::MXFP4)? (double)((K+A.block-1)/A.block)*M : 0);
  double bbytes = bBytes + ((dt==Dt::NVFP4||dt==Dt::MXFP4)? (double)((K+A.block-1)/A.block)*N : 0);
  R.gbps = (abytes+bbytes+(double)M*N*4)/ (ms*1e-3) / 1e9;
  R.ok = true;

  if(verify){
    std::vector<float> C((size_t)M*N);
    CHECK_CUDA(cudaMemcpy(C.data(), dC, C.size()*4, cudaMemcpyDeviceToHost));
    std::vector<float> ar, br; double maxrel=0; int nz=0; int shown=0;
    for(int m=0;m<M;++m){ dequant_row(A,dt,m,ar);
      for(int n=0;n<N;++n){ dequant_row(B,dt,n,br);
        double acc=0; for(int k=0;k<K;++k) acc += (double)ar[k]*br[k];
        float g = C[(size_t)m+(size_t)n*M];
        double rel = std::fabs(acc - g) / (std::fabs(acc)+1e-3);
        if(rel>maxrel) maxrel=rel; nz++;
        if(rel>0.2 && shown<8){ fprintf(stderr,"#   mismatch m=%d n=%d cpu=%.4f gpu=%.4f ratio=%.6f\n", m,n,acc,g, g/(acc+1e-9)); shown++; } } }
    R.note += " verify_maxrel=" + std::to_string(maxrel);
    if(getenv("FP4_PROBE")) dump_probe(dt_name(dt), M, N, K, C.data(), 1);
    if(getenv("FP4_MAP")) dump_map(M, N, K, C.data());
  }

  cublasLtMatrixLayoutDestroy(LA); cublasLtMatrixLayoutDestroy(LB);
  cublasLtMatrixLayoutDestroy(LC); cublasLtMatmulDescDestroy(op);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return R;
}

int main(int argc, char** argv) {
  bool verify=false; int iters=50; std::string dtypes="bf16,fp8,nvfp4,mxfp4";
  for(int i=1;i<argc;++i){
    if(!strcmp(argv[i],"--verify")) verify=true;
    else if(!strncmp(argv[i],"--iters=",8)) iters=atoi(argv[i]+8);
    else if(!strncmp(argv[i],"--dtypes=",9)) dtypes=argv[i]+9;
    else if(!strncmp(argv[i],"--slayout=",10)){
      const char* s=argv[i]+10;
      g_slayout = !strcmp(s,"row")?0 : !strcmp(s,"block")?1 : !strcmp(s,"row4")?2 : 3;
    }
  }
  cudaDeviceProp prop; CHECK_CUDA(cudaGetDeviceProperties(&prop,0));
  CHECK_LT(cublasLtCreate(&g_lt));
  size_t wsSize=256ULL<<20; void *ws; CHECK_CUDA(cudaMalloc(&ws, wsSize));
  size_t maxBytes=(size_t)8192*8192*4+64;
  void *dA,*dB,*dC,*dAs,*dBs; float *dS8A, *dS8B;
  CHECK_CUDA(cudaMalloc(&dA,maxBytes)); CHECK_CUDA(cudaMalloc(&dB,maxBytes));
  CHECK_CUDA(cudaMalloc(&dC,maxBytes)); CHECK_CUDA(cudaMalloc(&dAs,maxBytes));
  CHECK_CUDA(cudaMalloc(&dBs,maxBytes)); CHECK_CUDA(cudaMalloc(&dS8A,4)); CHECK_CUDA(cudaMalloc(&dS8B,4));

  int driver=0,rt=0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&driver);
  fprintf(stderr, "# gpu=%s sm_%d%d vram=%.1fGB driver=%d runtime=%d cublasLt=%zu cuda_ver=%d.%d\n",
          prop.name, prop.major, prop.minor, prop.totalGlobalMem/1073741824.0,
          driver, rt, cublasLtGetVersion(), CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);

  if(verify){
    fprintf(stderr, "# verify mode M=N=K=256\n");
    std::mt19937 g(7); std::normal_distribution<float> d(0,1);
    int M=256,N=256,K=256;
    Tensor A,B; std::vector<float> a((size_t)M*K), b((size_t)N*K);
    for(auto&v:a) v=d(g)*0.1f; for(auto&v:b) v=d(g)*0.1f;
    for(Dt dt:{Dt::BF16,Dt::FP8,Dt::NVFP4,Dt::MXFP4}){
      if(dtypes.find(dt_name(dt))==std::string::npos) continue;
      quantize(A,dt,M,K,a.data()); quantize(B,dt,N,K,b.data());
      // try all candidate scale layouts for FP4; report the first that verifies
      if(getenv("FP4_SLOT")){
        // single-byte probe: all scales 1.0 except ONE physical byte = 2.0.
        // deviations in C reveal the logical (row, block) that byte controls.
        int slot = atoi(getenv("FP4_SLOT"));
        int MM=256, KK=256, NN=16;
        std::fill(A.fp4.begin(), A.fp4.end(), 0x22);
        std::fill(B.fp4.begin(), B.fp4.end(), 0x22);
        A.fp4_scale.assign(16384, f32_to_e4m3(1.0f));
        B.fp4_scale.assign(16384, f32_to_e4m3(1.0f));
        A.fp4_scale[slot] = f32_to_e4m3(2.0f);
        g_slayout=5;
        Tensor AA=A, BB=B; // sizes: quantize() set rows/K on A,B already (256/16 rows)
        AA.rows=MM; AA.K=KK; BB.rows=NN; BB.K=KK;
        auto R=run_lt(dt,MM,NN,KK,AA,BB,dA,dB,dC,dAs,dBs,dS8A,dS8B,ws,wsSize,2,2,false);
        if(R.ok){
          std::vector<float> C((size_t)MM*NN);
          CHECK_CUDA(cudaMemcpy(C.data(), dC, C.size()*4, cudaMemcpyDeviceToHost));
          fprintf(stderr, "# slot=%d ->", slot);
          int shown=0;
          for(int m=0;m<MM && shown<6;++m) for(int n=0;n<NN && shown<6;++n){
            float dev = C[(size_t)m+(size_t)n*MM]-KK;
            if(std::fabs(dev)>0.5f){ fprintf(stderr, " (m=%d,b=%d,dev=%.0f)", m, n, dev/16.0f); shown++; }
          }
          if(!shown) fprintf(stderr, " (none)");
          fprintf(stderr, "\n");
        } else fprintf(stderr, "# slot=%d -> run failed (%s)\n", slot, R.note.c_str());
        continue;
      }
      if(getenv("FP4_MAP")){
        // FP4_MAP="M,K" overrides shape (N=min(16,ceil(K/16)))
        int MM=M, KK=K; const char* mp=getenv("FP4_MAP");
        if(strchr(mp,',')){ sscanf(mp,"%d,%d",&MM,&KK); }
        int NN = std::min(16,(KK+15)/16);
        apply_map(A,B,MM,NN,KK);
        g_slayout=4;  // upload via hw swizzle formula (being decoded)
        auto R=run_lt(dt,MM,NN,KK,A,B,dA,dB,dC,dAs,dBs,dS8A,dS8B,ws,wsSize,2,2,true);
        fprintf(stderr, "# map-run M=%d N=%d K=%d KB=%d\n", MM, NN, KK, (KK+15)/16);
        continue;
      }
      if(getenv("FP4_PROBE")){
        apply_probe(A,B,M,N,K, atoi(getenv("FP4_PROBE")));
        g_slayout=0;
        auto R=run_lt(dt,M,N,K,A,B,dA,dB,dC,dAs,dBs,dS8A,dS8B,ws,wsSize,2,2,true);
        fprintf(stderr, "# probe-run %-6s ok=%d %s\n", dt_name(dt), (int)R.ok, R.note.c_str());
        continue;
      }
      if(dt==Dt::NVFP4 || dt==Dt::MXFP4){
        for(int sl=4; sl<5; ++sl){   // "hw" swizzle is the decoded winner; see docs/spike.md
          g_slayout=sl;
          auto R=run_lt(dt,M,N,K,A,B,dA,dB,dC,dAs,dBs,dS8A,dS8B,ws,wsSize,2,2,true);
          fprintf(stderr, "# verify %-6s slayout=%-7s ok=%d %s\n", dt_name(dt), slayout_name(sl), (int)R.ok, R.note.c_str());
        }
      } else {
        auto R=run_lt(dt,M,N,K,A,B,dA,dB,dC,dAs,dBs,dS8A,dS8B,ws,wsSize,2,2,true);
        fprintf(stderr, "# verify %-6s ok=%d %s\n", dt_name(dt), (int)R.ok, R.note.c_str());
      }
    }
    return 0;
  }

  // shape sweep: prefill-like (compute bound) + decode/action-head-like (memory bound)
  struct S{int M,N,K;const char*tag;};
  std::vector<S> shapes={
    {2048,2048,2048,"prefill-s"},{4096,2048,2048,"prefill-m"},{4096,8192,2048,"prefill-l"},
    {8192,2048,4096,"prefall-xl"},{4096,4096,4096,"square-l"},
    {1,4096,4096,"b1-s"},{1,8192,2048,"b1-l"},{4,8192,2048,"b4-l"},{16,8192,2048,"b16-l"},
  };
  printf("dtype,tag,M,N,K,ok,algos,ms,tflops,gbps,note\n");
  std::mt19937 g(7); std::normal_distribution<float> dist(0,1);
  for(auto& s:shapes){
    std::vector<float> a((size_t)s.M*s.K), b((size_t)s.N*s.K);
    for(auto&v:a) v=dist(g)*0.1f; for(auto&v:b) v=dist(g)*0.1f;
    for(Dt dt:{Dt::BF16,Dt::FP8,Dt::NVFP4,Dt::MXFP4}){
      if(dtypes.find(dt_name(dt))==std::string::npos) continue;
      Tensor A,B; quantize(A,dt,s.M,s.K,a.data()); quantize(B,dt,s.N,s.K,b.data());
      auto R=run_lt(dt,s.M,s.N,s.K,A,B,dA,dB,dC,dAs,dBs,dS8A,dS8B,ws,wsSize,10,iters,false);
      printf("%s,%s,%d,%d,%d,%d,%d,%.4f,%.2f,%.2f,%s\n", dt_name(dt), s.tag, s.M, s.N, s.K,
             (int)R.ok, R.algos, R.ms, R.tflops, R.gbps, R.note.c_str());
      fflush(stdout);
    }
  }
  return 0;
}
