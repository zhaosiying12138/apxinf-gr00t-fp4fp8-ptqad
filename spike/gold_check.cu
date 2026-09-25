// gold_check.cu — run the adapter on the EXACT gold bytes that fail in Rust.
// If C++ passes here, the difference is the Rust invocation context (handle/
// stream); if C++ fails, the gold convention itself mismatches the adapter.
#include <cstdio>
#include <cstdint>
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
  auto x16=rd("/tmp/gold_xf16.bin");
  auto wdeq=rd("/tmp/gold_wdeq.bin");
  printf("loaded: ap=%zu asw=%zu bp=%zu bsw=%zu x=%zu wdeq=%zu\n",
         ap.size(),asw.size(),bp.size(),bsw.size(),x16.size(),wdeq.size());
  void *dA,*dAS,*dB,*dBS,*dC,*dX,*dW; size_t ws=256<<20;
  cudaMalloc(&dA,ap.size());cudaMalloc(&dAS,asw.size());
  cudaMalloc(&dB,bp.size());cudaMalloc(&dBS,bsw.size());
  cudaMalloc(&dC,(size_t)m*n*2);cudaMalloc(&dX,x16.size());cudaMalloc(&dW,wdeq.size());
  cudaMemcpy(dA,ap.data(),ap.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dAS,asw.data(),asw.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,bp.data(),bp.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dBS,bsw.data(),bsw.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dX,x16.data(),x16.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dW,wdeq.data(),wdeq.size(),cudaMemcpyHostToDevice);
  void* wsp; cudaMalloc(&wsp,ws);
  auto st=apxinf_fp4_gemm_f16(m,n,k,dA,dB,dAS,dBS,(__half*)dC,wsp,ws,0);
  printf("gemm status=%d\n",(int)st); if(st)return 1;
  std::vector<__half> C(m*n);
  cudaMemcpy(C.data(),dC,C.size()*2,cudaMemcpyDeviceToHost);
  // reference: x(f16) @ wdeq^T in double
  const __half* xh=(const __half*)dX; (void)xh;
  std::vector<float> xf(m*k);
  for(int i=0;i<m*k;++i) xf[i]=__half2float(((const __half*)x16.data())[i]);
  std::vector<float> wf(n*k);
  memcpy(wf.data(),wdeq.data(),wdeq.size());
  double maxrel=0; long bad=0;
  for(int i=0;i<m;++i)for(int j=0;j<n;++j){
    double acc=0; for(int t=0;t<k;++t) acc+=(double)xf[i*k+t]*wf[j*k+t];
    double g=__half2float(C[i*n+j]);
    double rel=std::abs(acc-g)/(std::abs(acc)+1e-3);
    if(rel>maxrel)maxrel=rel; if(rel>0.01)bad++;
  }
  printf("GOLD-IN-CPP maxrel=%.6f bad(>1%%)=%ld -> %s\n",maxrel,bad,bad?"FAIL":"PASS");
  for(int idx=0;idx<6;++idx){
    int i=idx/3, j=(idx%3)*80+40;
    double acc=0; for(int t=0;t<k;++t) acc+=(double)xf[i*k+t]*wf[j*k+t];
    double g=__half2float(C[i*n+j]);
    printf("  ref=%.5f got=%.5f ratio=%.5f
",acc,g,g/(acc+1e-12));
  }
  return bad?2:0;
}
