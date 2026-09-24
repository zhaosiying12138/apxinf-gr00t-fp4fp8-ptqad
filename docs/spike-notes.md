# FP4 Spike 结果（RTX 5090 Laptop, sm_120, CUDA 13.3, cuBLASLt 13.6）

日期：2026-09-24 · 代码：`spike/fp4_gemm_bench.cu` · 原始数据：`results/spike/`

## 结论：GO ✅

| 发现 | 证据 |
|---|---|
| NVFP4 block-scaled GEMM 在 GeForce sm_120 上可用 | heuristic 返回 4 个算法；数值校验 maxrel < 5e-7 |
| 计算受限形状（VLM prefill 类）NVFP4 大胜 | 4096³: BF16 92 → FP8 281 → **NVFP4 506 TFLOPS**（1.8× FP8，5.5× BF16）；8192×2048×4096 达 504 TFLOPS |
| 访存受限形状（batch-1 GEMV 类）FP4 输给 FP8 | b1: FP8 有效带宽 ~1250-1490 GB/s vs NVFP4 ~430-570 GB/s（cuBLASLt 的 FP4 小 M 路径未优化） |
| MXFP4（VEC32_UE8M0）sm_120 不支持 | heuristic 状态 7，所有形状 0 算法 |

**对混合精度设计的直接含义**：VLM backbone prefill（计算受限、batch>1 rollout）用 NVFP4 有 1.6-1.8× 加速；batch-1 的小 GEMM（动作头路径）应留在 FP8/BF16——这为"backbone FP4 + action head FP8"的混合精度准则提供了第一批硬数据。

## scale 张量 swizzle 布局（逆向解码过程）

cuBLASLt 的 `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` 要求 scale 张量按
**128×4 tile、行按 32 行分块交错**存放（NVIDIA 文档称 "MN x K4 tiled layout"）。
CUDA 13 API 无 order 参数（列主序 only）、`cublasLtMatmul/heuristic` 需 handle、algo 参数为指针、
`cuda_fp4.h`/`cuda_fp8.h` 转换走类型运算符——这些 API 变化均已适配。

解码方法（三层探针，全部保留在代码里，`FP4_FLAT_SCALE` / `FP4_PROBE` / `FP4_MAP` / `FP4_SLOT` 环境变量触发）：

1. **平 scale 探针**：所有 scale=1.0 → 误差恰为 0 ⇒ 数据打包（E2M1 nibble 序）与硬件完全一致，问题只在 scale 布局；
2. **单字节异常探针**：仅在物理缓冲某字节放 2.0，观察 C 的异常单元格 ⇒ 定位行/块映射；
3. **字节槽扫描**：扫物理偏移 0..4096，每个字节槽对应的逻辑行实测出来，拟合出公式。

最终公式（已通过随机数据全量验证，`offset(row r, block b)`）：

```
PS = 512 * ceil(KB/4)          # KB = ceil(K/16)，每 128 行一panel的步长
offset = PS*(r/128) + 512*(b/4) + 16*(r%32) + 4*((r/32)%4) + (b%4)
```

实测锚点（M=K=256, KB=16）：slot 0→(行0), 4→行32, 16→行1, 31→行97, 100→行38,
127→行103, 128→行8, 200→行76, 255→行111, 256→行16, 512/1024/1536→行0(块组1/2/3),
2048→行128, 4096→未使用。

## 性能原始数据（摘要）

| 形状 | BF16 TFLOPS | FP8 | NVFP4 | NVFP4/FP8 |
|---|---|---|---|---|
| 2048³ | 90.7 | 268.6 | 417.2 | 1.55× |
| 4096×2048×2048 | 91.5 | 259.3 | 441.1 | 1.70× |
| 4096×8192×2048 | 98.8 | 267.8 | 464.1 | 1.73× |
| 8192×2048×4096 | 88.9 | 276.3 | 503.9 | 1.82× |
| 4096³ | 92.5 | 281.2 | 506.2 | 1.80× |

| batch-1 形状 | BF16 GB/s | FP8 GB/s | NVFP4 GB/s |
|---|---|---|---|
| 1×4096×4096 | 774 | 1253 | 428 |
| 1×8192×2048 | 530 | 1490 | 521 |

注意：笔记本 TGP 波动，正式论文测量需分块跑+记录功耗（见 docs/PLAN.md 测量协议）。

## FP8 输出 dtype 探针（2026-09-25，spike/fp8_probe.cu）

引擎 pi05 fp8_static 在 sm_120 报 cuBLAS status 15 的根因：

| GEMM 配置（E4M3×E4M3） | heuristic | matmul |
|---|---|---|
| → F32 / BF16 / F16 输出 | 4 algos | ✓ 成功 |
| → **E4M3 输出**（± D-scale） | **status 15, 0 algos** | ✗ |

引擎 tuning key 把 fp8 GEMM 的 output_dtype 设为 F8E4M3（激活也走 FP8 存储省带宽，
Thor sm_110 验证过）；GeForce sm_120 的 cuBLASLt 不提供该组合。
**修复**：fp8 GEMM 输出 dtype 改 BF16（probe 已验证可用），代价是激活带宽 ×2；
对应上游 PR：sm≠110 时强制 BF16 输出。
