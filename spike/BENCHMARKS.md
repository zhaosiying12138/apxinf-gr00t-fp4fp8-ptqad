# 算子基准：构建、正确性与计时范围

所有程序独立于机器人评测。先完成数值验收，再记录时延；不可把算子倍率解释为完整策略加速。下面的 GPU 命令应在设备空闲时执行。

## 构建与 CPU 自检

```bash
CUDA_VISIBLE_DEVICES= make -C spike -j1 fp4_gemm_bench fp4_opbench fp8_probe
CUDA_VISIBLE_DEVICES= ./spike/fp4_opbench --cpu-self-test
```

`fp4_opbench` 直接链接当前 `cublaslt_fp4_adapter.cu`，使用 `apxinf_fp4_gemm_rowmajor_scaled_f16`。CPU 自检验证真实中位数与有限的确定性编码，不创建 CUDA 上下文；它不能代替后续 GPU 检查。

## GEMM 原语

```bash
set -euo pipefail
mkdir -p results/spike results/engine
./spike/fp4_gemm_bench --verify 2>&1 | tee results/spike/verify_ptqad_20260929.log
./spike/fp4_gemm_bench --iters=50 \
  > results/spike/gemm_ptqad_20260929.csv \
  2> results/spike/env_ptqad_20260929.log
```

默认采用 `hw` 块缩放布局。普通计时拒绝其他布局或诊断环境变量；`--verify --slayout=...` 仍可作为独立诊断。验证与计时使用同一 `run_lt` 合同：输入为 `[M,K]` 与 `[N,K]`，输出为列主序 FP32 `[M,N]`；NVFP4 二级张量尺度在这个基准中为 1。验证使用非方形 `128×256×256`，按实际编码值反量化后计算 CPU FP64 点积，并逐元素检查 `abs_error <= 1e-4 + 1e-4*abs(reference)`。失败返回非零；无法取得算法的格式单独报告 `unavailable`，不能当作通过数值验收。

计时在一次 CUDA event 区间内执行指定次数的 GEMM，报告平均每次耗时；量化、上传和算法选择在计时外。CSV 保持如下字段：

```text
dtype,tag,M,N,K,ok,algos,ms,tflops,gbps,note
```

`ok=0` 且没有可用算法的行不具有有效时延或吞吐。`gbps` 为依据输入、块尺度及输出字节数计算的名义带宽，不是硬件计数器测量。

## 在线激活量化与 GEMM 管线

```bash
./spike/fp4_opbench --verify-only --shape=gemma-1152 --m=522
./spike/fp4_opbench --samples=30 --warmup=10 \
  > results/engine/fp4_opbench_ptqad_20260929.csv \
  2> results/engine/fp4_opbench_ptqad_20260929.log
```

两条路径使用相同、完全初始化且可被 F16/BF16 精确表示的确定性原始输入。NVFP4 权重采用 CUDA 浮点编解码构建，权重二级尺度为非单位值 `0.25`，激活尺度为 1；两条 GEMM 均返回行主序 `[M,N]`。输入中包含正、负和全零块。

每个形状先比较全部在线激活 payload/scale 字节与 CPU CUDA-intrinsic 编码参考，再核对 21 个输出位置的反量化输入点积，包括边角与内部位置。FP4 和 BF16 各自对照自身表示及输出舍入；这不是两种精度彼此相等的测试。采样输出检查的容差为 FP4 `1e-3 + 1e-3*abs(reference)`、BF16 `1e-2 + 1e-2*abs(reference)`。这些检查不等于完整输出矩阵逐元素验收；原生算子另有完整的小矩阵合同测试。

正式计时中，FP4 包含在线 F16 激活量化与行主序 scaled adapter 的调用；BF16 为预先准备描述符和算法的 GEMM。权重准备、CPU/GPU 数据搬运和参考检查在计时外。每次调用各有独立 CUDA event 区间，偶数个样本的 P50 取排序后中间两项的平均。FP4 先测、BF16 后测，均单独 warmup；硬件功率、温度和时钟状态应随实验记录。

```text
shape,M,N,K,samples,warmup,weight_tensor_scale,fp4_us_p50,bf16_us_p50,speedup,fp4_check_max_abs,bf16_check_max_abs,status
```

`status=ok` 才有计时与倍率；`verified_samples` 是仅数值验收，不带计时；`unavailable` 不提供时延或倍率。`speedup` 定义为所声明两个计时范围的 `bf16_us_p50 / fp4_us_p50`。

## FP8 描述符探针

```bash
./spike/fp8_probe 2>&1 | tee results/spike/fp8_probe_ptqad_20260929.log
```

探针对 `[M,N,K]=[512,2048,2048]` 初始化全部 A/B，分别检查 E4M3×E4M3 到 F32、BF16、F16、E4M3 及带 D-scale 的 E4M3。输出保留 heuristic status、算法数、matmul status、CUDA launch 和同步状态。`NO ALGO` 是该描述符组合不可用的实测结果；实际执行错误使进程返回非零。本探针不比较数值输出，也不证明整条模型 FP8 路径可用。
