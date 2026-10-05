# APXInf 原生 NVFP4 执行实现

本文说明仓内补丁已经实现的 π0.5 packed NVFP4 路径：权重如何加载、激活如何在线编码，以及低精度 GEMM 如何进入 CUDA Graph。编译、打包、验收和计时命令见 [native-pi05.md](native-pi05.md)，完整源码走读见[论文附录 A](../paper/sections/14-附录A-核心源码走读与APXInf框架解析.md)。

GR00T 的正式闭环实验使用 PyTorch W4A4 QDQ 与独立 BF16 LoRA 旁路。其 packed 基座加 LoRA 的原生执行器尚未接通，因此 GR00T 成功率与本页的 π0.5 原生执行分别报告。

## 1. 模型路由与数据流

`apxinf-core` 提供张量和类型，`apxinf-cuda` 管理设备、缓冲区与算子，`apxinf-model` 负责模型结构和权重。π0.5 的 `nvfp4_static` 变体使用 `Fp4Blocks` 包装 `Bf16Blocks`，实际调用链为：

```text
Fp4DeviceWeights::from_artifact_dir
  → Fp4Blocks / gemm_maybe_fp4
  → fp4_linear_bf16
  → fp4_linear
  → row-major cuBLASLt adapter
```

CPU 转换入口生成 99 个投影产物：语言与动作分支各 18 层、每层 QKV 和 gate/up 两个投影，共 72 个；视觉分支 27 个 QKV 投影。这个数检查的是产物完整性。当前 `Fp4Blocks::vision` 仍委托 BF16 视觉路径；语言和动作分支按实际名称路由，缺少低精度产物或未接入的投影使用 BF16。

**BF16 权重和 packed 权重共同驻留。** packed 文件的编码压缩比不能直接解释为完整引擎的显存节省。模型配置、实际路由和测得显存需要分别核对。

## 2. packed 权重合同

设权重形状为 `[N,K]`，要求 `K` 能被 16 整除。每份产物包含：

| 文件或字段 | 含义 |
|---|---|
| `<stem>.packed.u8` | E2M1 数据，每字节两个元素；偶数 K 索引在低四位，长度为 `N*K/2` 字节。 |
| `<stem>.scale.u8` | 每 16 个元素一个 E4M3 块缩放，按硬件物理布局存放，并补齐尾部 tile。 |
| `manifest.json` | 张量名称、二维形状、`block=16`、FP32 `tscale`、文件长度等。 |

加载器验证形状、块长、数据和 scale 字节数，以及有限正的 `tscale`，随后上传。它不在加载时重新量化。通用加载合同支持非单位 `tscale`；本项目 π0.5 原生打包入口固定 `tscale=1`。CPU 准备入口另生成 producer manifest，绑定源 checkpoint、转换器和每个产物的 SHA-256。

## 3. 块缩放的物理布局

`VEC16_UE4M3` 的逻辑 scale 矩阵为 `[N,K/16]`。物理缓冲以 128 行、4 个块为一个 512 字节 tile，行索引在 tile 内交错排列。设逻辑行号为 `r`、块号为 `b`：

```text
KB = K / 16
PS = 512 * ceil(KB / 4)
offset(r,b) = PS * floor(r / 128) + 512 * floor(b / 4)
            + 16 * (r mod 32) + 4 * (floor(r / 32) mod 4)
            + (b mod 4)
scale_bytes = 512 * ceil(KB / 4) * ceil(N / 128)
```

例如 `(r,b)=(0,0)、(0,3)、(32,0)、(1,0)` 的字节偏移依次是 `0、3、4、16`。不足一个 tile 的行和块仍占用物理空间；这些 padding 字节必须清零。NumPy 的 `swizzle_scales`、原生激活 kernel 与加载器遵循同一规则。布局探针和数值参考见 [spike-notes.md](spike-notes.md)。

## 4. 在线激活编码与矩阵输出

原生线性层先将 BF16 输入转成 F16，再在线生成 E2M1 激活码和每 16 元素一个的 E4M3 块缩放。GEMM 接收两侧的低精度码与 scale，输出 F16，最后转回 BF16。在线激活的张量缩放固定为 `tau_x=1`，权重的 `tau_w` 随权重视图传入，乘积作为 cuBLASLt 的 `alpha`：

```text
Y[M,N] = (tau_x * tau_w) * (qx * sx)[M,K] * transpose((qw * sw)[N,K])
```

cuBLASLt 适配器交换输入操作数并使用转置后的列主序解释，将目标行主序 `[M,N]` 输出写入正确位置。该合同与裸 GEMM 探针的列主序 FP32 输出不同，必须单独验证。非单位张量缩放也单独进入验收，避免只测 `alpha=1` 而遗漏缩放错误。

## 5. CUDA Graph 中的内存与计划

`GraphWorkspace` 持有输入类型转换、激活 packed 数据、激活 scale 和输出的持久缓冲；context 另外持有可串行复用的 64 MiB GEMM scratch。执行计划按 host thread、CUDA 设备、stream、`M/N/K` 和 scratch 容量缓存。准备与捕获在同一 host thread 完成，捕获期间复用已准备计划。

每次调用重新绑定本层的 scale 指针和 `alpha`。同形状的两层可以共享计划，但不能沿用上一层的缩放。激活量化前对完整 scale tile 的清零也进入 Graph，每次重放都会执行，避免 arena 中残留数据进入 padding 区。当前验收覆盖单设备、匹配的 context/stream 与串行调用。

## 6. 已通过的验收与结果范围

[正式日志目录](../results/native_graph_20260929/) 保存四道检查，均记录退出状态 0：

| 检查 | 实际覆盖与结果 |
|---|---|
| 行主序输出与张量缩放 | 2 个非方阵 × 2 个非单位 scale；对 CPU 反量化乘积的最大绝对差为 0。 |
| 改变输入的 Graph 重放 | 4 次重放、2 组不同的 scale 指针与数值，逐输出核对通过。 |
| 激活 scale padding | 对完整 512 字节逐项检查；2 次重放前污染缓冲，行和 K 两个方向均通过。 |
| 完整 π0.5 `RequireGraph` | 2 组 eager 参考、3 次输入变化重放；每次 1,600 个输出，与对应 eager 结果的最大绝对差为 0。 |

完整模型验收证明同一 NVFP4 路径的 eager 与 Graph 一致。它不衡量量化相对 BF16 的行为损失。原生策略延迟、GEMM 与单层管线的全部测量口径和数据见[论文附录 C](../paper/sections/15-附录C-执行基准与完整测量.md)；GR00T 的 QAD/OPD 成功率按[闭环复现协议](reproduce-ptqad.md)独立验证。
