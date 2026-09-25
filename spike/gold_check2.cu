// gold_check2.cu — adapter on gold bytes + error-structure printout.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#include <cublasLt.h>

extern "C" cublasStatus_t apxinf_fp4_gemm_f16(
    int m, int n, int k, const void* a_packed, const void* b_packed,
    const void* a_scale, const void* b_scale, __half* c,
    void* ws, size_t ws_bytes, cudaStream_t stream);

static std::vector<uint8_t> rd(const char* p){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<uint8_t>v(n);fread(v.data(),1,n,f);fclose(f);return v;}

#include <cmath>
static std::vector<uint8_t> enc_flat(const std::vector<float>& x, int rows,int K,
                                     std::vector<uint8_t>& packed_out){
  // CONSTANT scale per tensor: sb = e4m3(global_amax/6) for every block
  float amax=0; for(float v:x) amax=std::fmax(amax,std::fabs(v));
  uint8_t sb=0; { float m=amax/6, e=0; while(m>=2){m*=.5;e+=1;} while(m<1){m*=2;e-=1;}
    int man=(int)((m-1)*8+0.5); if(man>7){man=0;e+=1;} sb=(uint8_t)(((int)(e+7)<<3)|man); }
  float sd; { int e=((sb>>3)&0xF); float mm=(sb&7)/8.0f; sd = e==0 ? (sb&7)*0.001953125f : (1+mm)*powf(2.0f,e-7); }
  packed_out.assign(rows*K/2, 0);
  int kb=K/16;
  std::vector<uint8_t> swz(512*((kb+3)/4)*((rows+127)/128), sb); // EVERY byte = sb (all layouts same)
  const float grid[8]={0,.5,1,1.5,2,3,4,6};
  for(int r=0;r<rows;++r) for(int b=0;b<kb;++b)
    for(int j=0;j<16;j+=2){
      auto q=[&](float v){ float a=std::fabs(v)/sd; int c = a<0.25f?0: a<0.75f?1: a<1.25f?2: a<1.75f?3: a<2.5f?4: a<3.5f?5: a<5.0f?6:7;
        uint8_t code = (uint8_t)c; float val = grid[c];
        return std::make_pair( v<0 ? (code|8) : code, val*(v<0?-1.0f:1.0f)*sd ); };
      auto [c0,v0]=q(x[r*K+b*16+j]); auto [c1,v1]=q(x[r*K+b*16+j+1]);
      packed_out[(size_t)(r*K+b*16+j)/2]=(uint8_t)(c0|(c1<<4));
    }
  return swz;
}
int main(){
  int m=64,n=256,k=512;
  auto ap=rd("/tmp/gold_a_packed.bin"), asw=rd("/tmp/gold_a_swz.bin");
  auto bp=rd("/tmp/gold_packed.bin"), bsw=rd("/tmp/gold_swz.bin");
  auto x16=rd("/tmp/gold_xf16.bin");
  auto wdeq=rd("/tmp/gold_wdeq.bin");
  printf("loaded: ap=%zu asw=%zu bp=%zu bsw=%zu x=%zu wdeq=%zu\n",
         ap.size(),asw.size(),bp.size(),bsw.size(),x16.size(),wdeq.size());
  void *dA,*dAS,*dB,*dBS,*dC; size_t ws=256<<20;
  cudaMalloc(&dA,ap.size());cudaMalloc(&dAS,asw.size());
  cudaMalloc(&dB,bp.size());cudaMalloc(&dBS,bsw.size());
  cudaMalloc(&dC,(size_t)m*n*2);
  cudaMemcpy(dA,ap.data(),ap.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dAS,asw.data(),asw.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,bp.data(),bp.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dBS,bsw.data(),bsw.size(),cudaMemcpyHostToDevice);
  void* wsp; cudaMalloc(&wsp,ws);
  auto st=apxinf_fp4_gemm_f16(m,n,k,dA,dB,dAS,dBS,(__half*)dC,wsp,ws,0);
  printf("gemm status=%d\n",(int)st); if(st)return 1;
  std::vector<__half> C(m*n);
  cudaMemcpy(C.data(),dC,C.size()*2,cudaMemcpyDeviceToHost);
  std::vector<float> xf(m*k), wf(n*k);
  for(int i=0;i<m*k;++i) xf[i]=__half2float(((const __half*)x16.data())[i]);
  memcpy(wf.data(),wdeq.data(),wdeq.size());
  double maxrel=0; long bad=0;
  for(int i=0;i<m;++i)for(int j=0;j<n;++j){
    double acc=0; for(int t=0;t<k;++t) acc+=(double)xf[i*k+t]*wf[j*k+t];
    double g=__half2float(C[i*n+j]);
    double rel=std::abs(acc-g)/(std::abs(acc)+1e-3);
    if(rel>maxrel)maxrel=rel; if(rel>0.01)bad++;
  }
  printf("GOLD-IN-CPP maxrel=%.6f bad(>1pct)=%ld\n",maxrel,bad);
  for(int idx=0;idx<6;++idx){
    int i=idx/3, j=(idx%3)*80+40;
    double acc=0; for(int t=0;t<k;++t) acc+=(double)xf[i*k+t]*wf[j*k+t];
    double g=__half2float(C[i*n+j]);
    printf("  ref=%.5f got=%.5f ratio=%.5f\n",acc,g,g/(acc+1e-12));
  }
  // EXPERIMENT: re-quantize A with CONSTANT scale, rerun -> isolates per-block scale handling
  {
    std::vector<float> xf_all(m*k);
    for(int i=0;i<m*k;++i) xf_all[i]=__half2float(((const __half*)x16.data())[i]);
    std::vector<uint8_t> ap_flat; auto asw_flat = enc_flat(xf_all, m, k, ap_flat);
    cudaMemcpy(dA, ap_flat.data(), ap_flat.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dAS, asw_flat.data(), asw_flat.size(), cudaMemcpyHostToDevice);
    auto st2 = apxinf_fp4_gemm_f16(m,n,k,dA,dB,dAS,dBS,(__half*)dC,wsp,ws,0);
    cudaMemcpy(C.data(),dC,C.size()*2,cudaMemcpyDeviceToHost);
    // reference with the SAME flat dequant semantics
    double mr=0; long bd=0;
    for(int i=0;i<m;++i)for(int j=0;j<n;++j){
      double acc=0; for(int t=0;t<k;++t) acc+=(double)xf[i*k+t]*wf[j*k+t];
      double g=__half2float(C[i*n+j]);
      double rel=std::abs(acc-g)/(std::abs(acc)+1e-3);
      if(rel>mr)mr=rel; if(rel>0.01)bd++;
    }
    printf("FLAT-A-SCALE: status=%d maxrel=%.6f bad=%ld -> % MSG");
  }
  return bad?2:0;
}
