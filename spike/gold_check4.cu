// gold_check4.cu — CORRECT reference: GEMM(A_deq, B_deq) vs hardware GEMM(A_fp4, B_fp4).
// This is the apples-to-apples comparison the earlier checks got wrong (they
// used the RAW activation as reference, mixing in 9% activation-quant noise).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include <cublasLt.h>

extern "C" cublasStatus_t apxinf_fp4_gemm_f16(
    int m, int n, int k, const void* a_packed, const void* b_packed,
    const void* a_scale, const void* b_scale, __half* c,
    void* ws, size_t ws_bytes, cudaStream_t stream);

static std::vector<uint8_t> rd(const char* p){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<uint8_t>v(n);fread(v.data(),1,n,f);fclose(f);return v;}

int main(){
  int m=64,n=256,k=512;
  auto ap=rd("/tmp/gold_a_packed.bin"), asw=rd("/tmp/gold_a_swz.bin");
  auto bp=rd("/tmp/gold_packed.bin"), bsw=rd("/tmp/gold_swz.bin");
  auto adeq=rd("/tmp/gold_a_deq.bin"), wdeq=rd("/tmp/gold_wdeq.bin");
  std::vector<float> a(m*k), w(n*k);
  memcpy(a.data(),adeq.data(),adeq.size());
  memcpy(w.data(),wdeq.data(),wdeq.size());
  void *dA,*dAS,*dB,*dBS,*dC,*wsp; size_t ws=256<<20;
  cudaMalloc(&dA,ap.size());cudaMalloc(&dAS,asw.size());cudaMalloc(&dB,bp.size());
  cudaMalloc(&dBS,bsw.size());cudaMalloc(&dC,(size_t)m*n*2);cudaMalloc(&wsp,ws);
  cudaMemcpy(dA,ap.data(),ap.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dAS,asw.data(),asw.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,bp.data(),bp.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dBS,bsw.data(),bsw.size(),cudaMemcpyHostToDevice);
  auto st=apxinf_fp4_gemm_f16(m,n,k,dA,dB,dAS,dBS,(__half*)dC,wsp,ws,0);
  std::vector<__half> C((size_t)m*n);
  cudaMemcpy(C.data(),dC,C.size()*2,cudaMemcpyDeviceToHost);
  double maxrel=0; long bad1=0, bad5=0;
  double corr_num=0, corr_da=0, corr_db=0;
  for(int i=0;i<m;++i)for(int j=0;j<n;++j){
    double acc=0; for(int t=0;t<k;++t)acc+=(double)a[i*k+t]*w[j*k+t];
    double g=__half2float(C[(size_t)i+(size_t)j*m]);
    double rel=std::abs(acc-g)/(std::abs(acc)+1e-3);
    if(rel>maxrel)maxrel=rel;
    if(rel>0.01)bad1++;
    if(rel>0.05)bad5++;
    corr_num+=(acc)*(g); corr_da+=acc*acc; corr_db+=g*g;
  }
  double corr=corr_num/(std::sqrt(corr_da)*std::sqrt(corr_db)+1e-30);
  printf("CORRECT-REF: status=%d maxrel=%.6f bad(>1%%)=%ld bad(>5%%)=%ld corr=%.6f\n",
         (int)st,maxrel,bad1,bad5,corr);
  printf("VERDICT: %s\n", corr>0.9999?"THE COMPOSITION WAS CORRECT ALL ALONG (reference bug in earlier checks)":"real bug remains");
  return 0;
}
