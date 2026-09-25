// gold_check3.cu — decisive experiment: variable-scale A/B vs FLAT-scale A/B.
// If flat passes and variable fails on the SAME data, the selected algo does
// not honor per-block VEC16 scales (or reads a different layout).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cublasLt.h>

extern "C" cublasStatus_t apxinf_fp4_gemm_f16(
    int m, int n, int k, const void* a_packed, const void* b_packed,
    const void* a_scale, const void* b_scale, __half* c,
    void* ws, size_t ws_bytes, cudaStream_t stream);

static std::vector<uint8_t> rd(const char* p){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<uint8_t>v(n);fread(v.data(),1,n,f);fclose(f);return v;}
static uint8_t e4m3_enc(float v){
  if(v>=448.f) return 0x7E; if(v<=0.f) return 0;
  float m=v; int e=0; while(m>=2.f){m*=.5f;e+=1;} while(m<1.f){m*=2.f;e-=1;}
  int exp=e+7; if(exp<=0){int s=(int)(v*512.f+0.5f); return (uint8_t)std::min(std::max(s,0),7);}
  float g=(m-1.f)*8.f; int fl=(int)g; float fr=g-fl;
  int man = fr>0.5f?fl+1 : fr<0.5f?fl : (fl%2==0?fl:fl+1);
  if(man>7){man=0;exp+=1;} if(exp>15) return 0x7E;
  return (uint8_t)((exp<<3)|man);
}
static float e4m3_dec(uint8_t b){
  int e=(b>>3)&0xF; float m=(b&7);
  return e==0 ? m*0.001953125f : (1.f+m*0.125f)*std::pow(2.f,(float)e-7.f);
}
static float e2m1_mag(float a){
  if(a<0.25f)return 0.f; if(a<0.75f)return 0.5f; if(a<1.25f)return 1.f;
  if(a<1.75f)return 1.5f; if(a<2.5f)return 2.f; if(a<3.5f)return 3.f;
  if(a<5.0f)return 4.f; return 6.f;
}
// quantize rows x K with PER-BLOCK scales (variable) and CONSTANT scale; return both
static void quant(const std::vector<float>& x,int rows,int K,bool flat,
                  std::vector<uint8_t>& packed,std::vector<uint8_t>& swz,std::vector<float>& deq){
  int kb=K/16; packed.assign((size_t)rows*K/2,0); deq.assign(rows*K,0.f);
  float gscale=0; for(float v:x) gscale=std::fmax(gscale,std::fabs(v));
  uint8_t gsb=e4m3_enc(gscale/6.f); float gsd=e4m3_dec(gsb); if(gsd<=0)gsd=1;
  size_t PS=512ull*((kb+3)/4);
  swz.assign(PS*((rows+127)/128), flat?gsb:0);
  for(int r=0;r<rows;++r)for(int b=0;b<kb;++b){
    float amax=0; for(int j=0;j<16;++j)amax=std::fmax(amax,std::fabs(x[r*K+b*16+j]));
    uint8_t sb= flat?gsb:e4m3_enc(amax/6.f);
    float sd=e4m3_dec(sb); if(sd<=0)sd=1;
    if(!flat) swz[PS*(r/128)+512ull*(b/4)+16ull*(r%32)+4ull*((r/32)%4)+(b%4)]=sb;
    for(int j=0;j<16;j+=2){
      float q0=e2m1_mag(std::fabs(x[r*K+b*16+j])/sd);
      float q1=e2m1_mag(std::fabs(x[r*K+b*16+j+1])/sd);
      deq[r*K+b*16+j]  =(x[r*K+b*16+j]  <0?-q0:q0)*sd;
      deq[r*K+b*16+j+1]=(x[r*K+b*16+j+1]<0?-q1:q1)*sd;
      auto nib=[](float q)->int{return q==0?0:q==0.5f?1:q==1?2:q==1.5f?3:q==2?4:q==3?5:q==4?6:7;};
      int lo=nib(q0)|(x[r*K+b*16+j]  <0?8:0);
      int hi=nib(q1)|(x[r*K+b*16+j+1]<0?8:0);
      packed[(size_t)(r*K+b*16+j)/2]=(uint8_t)(lo|(hi<<4));
    }
  }
}
static double run_and_check(int m,int n,int k,void*dA,void*dB,void*dAS,void*dBS,void*dC,void*wsp,size_t ws,
                            const std::vector<float>& a_deq,const std::vector<float>& b_deq,const char* tag){
  auto st=apxinf_fp4_gemm_f16(m,n,k,dA,dB,dAS,dBS,(__half*)dC,wsp,ws,0);
  std::vector<__half> C((size_t)m*n);
  cudaMemcpy(C.data(),dC,C.size()*2,cudaMemcpyDeviceToHost);
  double mr=0; long bd=0;
  for(int i=0;i<m;++i)for(int j=0;j<n;++j){
    double acc=0; for(int t=0;t<k;++t)acc+=(double)a_deq[i*k+t]*b_deq[j*k+t];
    double g=__half2float(C[(size_t)i+(size_t)j*m]);
    double rel=std::abs(acc-g)/(std::abs(acc)+1e-3);
    if(rel>mr)mr=rel; if(rel>0.02)bd++;
  }
  printf("%-18s status=%d maxrel=%.6f bad=%ld -> %s\n",tag,(int)st,mr,bd,bd?"FAIL":"PASS");
  return mr;
}
int main(){
  int m=64,n=256,k=512;
  // SAME random data for both sides
  auto lcg=[](uint64_t&s){s=s*6364136223846793005ULL+1442695040888963407ULL;return (double)((s>>33)&0xFFFFFF)/16777216.0-0.5;};
  uint64_t sa=7,sb=9;
  std::vector<float> A(m*k),B(n*k);
  for(auto&v:A)v=lcg(sa)*0.5f;   // WIDE dynamic range within rows: multiply by block-varying gain
  for(int r=0;r<m;++r)for(int b=0;b<k/16;++b){float g=std::pow(10.f,(float)b/(k/16));for(int j=0;j<16;++j)A[r*k+b*16+j]*=g;}
  for(auto&v:B)v=lcg(sb)*0.5f;
  for(int r=0;r<n;++r)for(int b=0;b<k/16;++b){float g=std::pow(10.f,(float)b/(k/16));for(int j=0;j<16;++j)B[r*k+b*16+j]*=g;}

  std::vector<uint8_t> apV,asV,apF,asF,bpV,bsV,bpF,bsF; std::vector<float> aV,aF,bV,bF;
  quant(A,m,k,false,apV,asV,aV); quant(A,m,k,true,apF,asF,aF);
  quant(B,n,k,false,bpV,bsV,bV); quant(B,n,k,true,bpF,bsF,bF);
  void *dA,*dAS,*dB,*dBS,*dC,*wsp; size_t ws=256<<20;
  cudaMalloc(&dA,apV.size());cudaMalloc(&dAS,asV.size());cudaMalloc(&dB,bpV.size());
  cudaMalloc(&dBS,bsV.size());cudaMalloc(&dC,(size_t)m*n*2);cudaMalloc(&wsp,ws);
  // 1) both variable
  cudaMemcpy(dA,apV.data(),apV.size(),cudaMemcpyHostToDevice);cudaMemcpy(dAS,asV.data(),asV.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,bpV.data(),bpV.size(),cudaMemcpyHostToDevice);cudaMemcpy(dBS,bsV.data(),bsV.size(),cudaMemcpyHostToDevice);
  run_and_check(m,n,k,dA,dB,dAS,dBS,dC,wsp,ws,aV,bV,"VAR-A x VAR-B");
  // 2) both flat
  cudaMemcpy(dA,apF.data(),apF.size(),cudaMemcpyHostToDevice);cudaMemcpy(dAS,asF.data(),asF.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,bpF.data(),bpF.size(),cudaMemcpyHostToDevice);cudaMemcpy(dBS,bsF.data(),bsF.size(),cudaMemcpyHostToDevice);
  run_and_check(m,n,k,dA,dB,dAS,dBS,dC,wsp,ws,aF,bF,"FLAT-A x FLAT-B");
  // 3) flat A, variable B
  cudaMemcpy(dA,apF.data(),apF.size(),cudaMemcpyHostToDevice);cudaMemcpy(dAS,asF.data(),asF.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,bpV.data(),bpV.size(),cudaMemcpyHostToDevice);cudaMemcpy(dBS,bsV.data(),bsV.size(),cudaMemcpyHostToDevice);
  run_and_check(m,n,k,dA,dB,dAS,dBS,dC,wsp,ws,aF,bV,"FLAT-A x VAR-B");
  // 4) var A, flat B
  cudaMemcpy(dA,apV.data(),apV.size(),cudaMemcpyHostToDevice);cudaMemcpy(dAS,asV.data(),asV.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,bpF.data(),bpF.size(),cudaMemcpyHostToDevice);cudaMemcpy(dBS,bsF.data(),bsF.size(),cudaMemcpyHostToDevice);
  run_and_check(m,n,k,dA,dB,dAS,dBS,dC,wsp,ws,aV,bF,"VAR-A x FLAT-B");
  // BYTE DUEL: C++-quantize the gold x16 data, compare vs python gold bytes
  {
    auto x16 = rd("/tmp/gold_xf16.bin");
    std::vector<float> gx(m*k);
    for(int i=0;i<m*k;++i) gx[i]=__half2float(((const __half*)x16.data())[i]);
    std::vector<uint8_t> p,s; std::vector<float> d;
    quant(gx,m,k,false,p,s,d);
    auto gp = rd("/tmp/gold_a_packed.bin"); auto gs = rd("/tmp/gold_a_swz.bin");
    long pmm=0,smm=0;
    for(size_t i=0;i<p.size();++i) if(p[i]!=gp[i])pmm++;
    for(size_t i=0;i<s.size();++i) if(s[i]!=gs[i])smm++;
    printf("BYTE DUEL: packed diff %ld/%zu swz diff %ld/%zu",pmm,p.size(),smm,s.size());
    if(pmm||smm){ for(size_t i=0;i<p.size() && pmm>0 && i<64;++i) if(p[i]!=gp[i]){printf("  first diff @%zu cpp=%02x py=%02x",i,p[i],gp[i]);} }
  }
  return 0;
}
