# FP4 GEMM 布局验证与基准说明

`spike/fp4_gemm_bench.cu` 验证低位宽输入、块缩放布局和 cuBLASLt GEMM，并测量不同矩阵形状下的算子耗时。它提供 APXInf 原生 NVFP4 接线所需的算子证据；模型执行、激活在线编码成本和 LIBERO 成功率由其他入口测量。

## 1. 编译与运行

完成仓库 CUDA 开发环境安装后，在项目根目录执行：

```bash
bash spike/run.sh
```

脚本先 `make`，再执行数值验证，最后运行性能扫表。每次生成一组带时间戳的 `results/spike/verify_*.log`、`gemm_*.csv` 和 `env_*.log`。数值验证失败会停止脚本。GPU 基准应与模型训练、其他计时进程串行运行。

需要分别执行时可使用：

```bash
make -C spike
./spike/fp4_gemm_bench --verify
./spike/fp4_gemm_bench --iters=50 --dtypes=bf16,fp8,nvfp4,mxfp4 --slayout=hw
```

`--slayout` 接受 `hw|row|block|row4|block4`。正式性能扫表只接受默认硬件布局 `hw`，并拒绝任何已设置的 `FP4_FLAT_SCALE`、`FP4_SLOT`、`FP4_MAP` 或 `FP4_PROBE` 环境变量，防止诊断布局混入性能结果。

## 2. 数值参考比较什么

普通 `--verify` 使用 `M=128、N=256、K=256`，逐项检查全部 32,768 个输出。CPU 参考先解码**实际低精度输入**，再以 FP64 累加点积；验收条件为有限输出，且每项满足：

```text
abs(gpu - reference) <= 1e-4 + 1e-4 * abs(reference)
```

这检查低精度算子是否正确计算其输入，不用量化前的浮点结果作为精确参考。量化前后的差值属于量化误差，应另外研究。日志分别记录 `verified`、`failed` 和 `unavailable`；未找到算法的格式没有通过数值验证。

## 3. NVFP4 scale 的布局

E2M1 数据以每字节两个元素打包，偶数 K 索引位于低四位。每 16 个元素对应一个 E4M3 scale；硬件要求这些 scale 以 128×4 tile、每 32 行交错的方式排列。对逻辑 `[N,K/16]` scale 矩阵：

```text
KB = K / 16
PS = 512 * ceil(KB / 4)
offset(r,b) = PS * floor(r / 128) + 512 * floor(b / 4)
            + 16 * (r mod 32) + 4 * (floor(r / 32) mod 4)
            + (b mod 4)
scale_bytes = 512 * ceil(KB / 4) * ceil(N / 128)
```

`r` 是逻辑行号，`b` 是沿 K 轴的 16 元素块号。最后一个物理 tile 也要完整分配，逻辑范围外的字节填零。激活矩阵使用同一公式，将行数 `N` 换成 `M`。

代码中的诊断探针用于解释这个映射：先将 scale 全设为 1，隔离 E2M1 数据打包；再只改变一个物理字节，观察它影响哪些输出；最后扫描字节槽，核对逻辑行/块与物理偏移。诊断变量只用于 `--verify` 模式，普通随机输入的全量验收才是正式数值记录。

## 4. 性能计时的边界

默认扫表包含 9 个形状，覆盖较大的矩阵乘法和 `M=1、4、16` 的小批量形状。每个形状预热 10 次，再以 CUDA event 测量 50 次 GEMM 的平均耗时。量化、scale 排列、CPU 到 GPU 上传和算法查询都在计时之外。

该探针的输出为**列主序 FP32**，二级缩放为 1。CSV 保存：

```text
dtype,tag,M,N,K,ok,algos,ms,tflops,gbps,note
```

其中 `tflops=2*M*N*K/(time_seconds*10^12)`；`gbps` 按输入编码、输入 scale 和 FP32 输出字节计算，是等效带宽。无可用 heuristic 表明当前硬件、库版本、形状和布局组合没有得到可执行算法；它不证明该格式在所有硬件上均不受支持。`ok=0` 行的零耗时不参与加速比计算。

## 5. 当前保留的证据

| 记录 | 用途 |
|---|---|
| [verify_ptqad_20260929.log](../results/spike/verify_ptqad_20260929.log) | BF16、FP8、NVFP4 的数值验收及不可用格式记录。 |
| [gemm_ptqad_20260929.csv](../results/spike/gemm_ptqad_20260929.csv) | 全部 9 个形状、4 种请求格式的原始性能行。 |
| [env_ptqad_20260929.log](../results/spike/env_ptqad_20260929.log) | GPU、驱动、CUDA、cuBLASLt 与计时合同。 |
| [fp8_probe_ptqad_20260929.log](../results/spike/fp8_probe_ptqad_20260929.log) | FP8 输入配不同输出 dtype 的算法可用性检查。 |

完整数据表及环境条件见[论文附录 C](../paper/sections/15-附录C-执行基准与完整测量.md)。阅读时保留四种测量的区别：这里的裸 GEMM 不含激活编码；`fp4_opbench` 的 18 个真实形状包含在线激活编码；策略接口计时包含模型调用及其明示的前后处理；LIBERO 成功率衡量动作进入环境后的闭环结果。原生模型的行主序输出、非单位 scale 与 Graph 重放还须通过[引擎验收](native-pi05.md)。
