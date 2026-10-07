# GR00T W4A4 原生执行缺口与最小接通路线

本次为 2026-10-07 的源码只读审计。没有运行 GPU、编译新产物或实现新的执行路径。v12 仍使用 Torch QDQ；下述路线是待完成工作，不能计入已有闭环成功率或加速证据。

## 当前能够执行什么

现有 APXInf GR00T 已有正式策略、官方预处理、通用 GR00T executor、BF16/FP8/INT8 精度实现和图执行。`third_party/apxinf-robo/apxinf/crates/apxinf-model/src/gr00t/vla_runtime.rs:260` 只构造这三种精度；`python/apxinf/apxinf/policies/impls/gr00t.py:140` 明确拒绝其他精度。它没有 GR00T NVFP4 权重类型、LoRA 装载或 W4A4 executor。

已有真实 NVFP4 算子位于 APXInf 的 `crates/apxinf-cuda/adapters/cublaslt_fp4_adapter.cu`，Rust 包装为 `src/kernels/fp4.rs`。算子执行 F16 输入的在线 NVFP4 量化及 packed W4A4 GEMM，输出 F16；BF16 包装先转 F16，计算后再转 BF16。权重使用 E2M1、每 16 元素一个 E4M3 scale、物理 scale 排列及 FP32 tensor scale；激活 tensor scale 固定为 1。

`exp/prepare_native_pi05.py` 固定 π0.5 的 18/18/27 层及 99 个投影，不能处理 GR00T。`exp/run_native_graph_gates.sh` 的完整模型检查也是 π0.5。其成功不构成 GR00T 数值一致性、成功率或延迟证据。

## 推荐先完成的最小路线

保留目前完整 Torch GR00T 的官方处理器、模型控制流、动作解码、服务和 LIBERO 评测，只把量化矩阵乘主分支替换为 **APXInf 的真实 NVFP4 CUDA 算子**，LoRA 仍在 BF16 旁路执行。这个交付应准确称为“APXInf CUDA 算子驱动的 Torch GR00T”，不能称为完整 Rust GR00T 引擎。

1. **添加薄算子桥接，不重写模型。** 当前 `apxinf_py` 只导出 `ModelRunner` 和 tokenizer，模型桥是 host NumPy 输入输出，没有 Torch、DLPack 或 CUDA array 接口。安装的 SO 与 `target/wheel/release/libapxinf_py.so` 均未导出 FP4 动态符号，不能直接假设 `ctypes` 可调用。最小扩展是薄 ATen/CUDA 绑定，链接同一份 APXInf adapter：`apxinf_nvfp4_quantize_activation`、`apxinf_fp4_prepare_rowmajor_f16`、`apxinf_fp4_gemm_rowmajor_prepared_f16`。绑定必须使用 Torch 当前 CUDA stream，检查 device/dtype/contiguity/shape，保留张量生命周期，复用 scratch，传递 CUDA/cuBLAS 错误；不得用逐层 CPU 拷贝连接。
2. **导出当前基座对应的 packed 权重。** `quant/ptq/bake.py:382` 只保存反量化后转回原 dtype 的值。`bake_category.py:99` 虽获得 encoding，随后也没有保存 packed 本体。不能对这些 BF16 值或合并后的 LoRA 权重重新估计 scale。应从精确原始 BF16 checkpoint 和冻结 RTN recipe 重放编码，逐层验证 `decode(packed).to(original_dtype)` 等于现有量化基座。复用 `category_fp4.py:167` 的 `pack_exact` 思路，保存代码、scale、tensor scale、布局、逻辑/物理 shape、别名和文件 SHA。CategorySpecificLinear 的 `[C,K,N]` 要按类别转成 `[N,K]`，K=132 补零到 144，激活补同样的零。
3. **保持训练与部署的两分支语义。** `rl/w4a4_lora.py:55` 已有 `base_operator` 扩展点，但收到的是 `QDQ(x)`，`w4a4_deploy.py:35` 也未透传该参数。不能把它直接接在线量化算子导致二次量化。应新增明确的 native 主分支接口：主分支由原始输入在线量化一次，旁路计算 `B(A(x)) * alpha/rank`，bias 只加一次，A/B 不合并或重新量化。主 v12 的 QDQ 路径保持原样。
4. **覆盖真实执行的全部目标。** 普通 Linear、7 个 CategorySpecificLinear 都要按冻结配方接入并记录实际调用次数；468 个参与动作前向的 LoRA 目标不能遗漏。3 个 embedding/position 张量只有权重量化，不应称为 A4 GEMM。未执行 lm_head 与实际执行层分开统计。若继续保留反量化 BF16 权重用于查表或 fallback，其实际显存也必须计入，不能用文件编码比例代替显存节省。
5. **对齐两个尚存的数值边界。** v12 对有限值启用了 `FP4VLA_SATURATE_F16_ACTIVATIONS=1`，而 native BF16→F16 cast 没有该 clamp；桥接必须显式复现它并拒绝 NaN/Inf。Torch QDQ 参考也没有模拟 native GEMM 的 F16 输出再转 BF16及 bias 顺序，packed 精确编码不代表整个 Linear 或整个模型逐比特一致。这些差异必须测量，不能直接继承成功率。

## 最小验收顺序

先在 CPU 冻结导出清单、A/B 及基座身份并测试布局、scale、padding、旁路输入和拒绝错误配方。GPU 空闲后串行完成以下检查：

1. 对真实 GR00T 矩阵形状逐层比较 packed 算子、明确舍入边界的数值参考与现有 QDQ；包含类别 K=132、输出 N=132、零块、不同 tensor scale、重复/变化输入和非默认 stream。当前 `spike/fp4_opbench.cu` 只测 M=522/778/2048，不覆盖动作头小 M，不能推定 M=1、40 等形状的可用性与性能。
2. 固定官方处理器输出、embodiment、真实初始噪声和四步积分日程，比较同一 PTQ/QAD/OPD checkpoint 的 QDQ 与 native kernel 输出；同时检查完整有效 16×7 动作块、各动作维度、gripper 命令与逐步漂移。发生差异时先定位首个不一致层，不用最终 success 掩盖转换错误。
3. 同一个现有 LIBERO 服务增加显式 backend 选择与 fail-closed manifest，沿用相同 reset 身份、动作步数、种子、统计和协议。重新运行同模型真实 kernel 的闭环，不能把原 QDQ 成功率直接贴给 native。记录完整覆盖和 fallback；只测一部分层只能声称这一部分使用真实 kernel。
4. BF16、PTQ native、QAD native、OPD native 分别独立进程计时。固定同一批真实输入和噪声，排除初次加载/编译/plan 构建，预热后记录同步完成的模型调用及端到端策略调用 P50/P95/P99、原始数组、峰值显存和部署文件字节。Torch＋算子桥接包括 Python 调度、cast、在线量化、bias、LoRA 和小矩阵开销，速度可能不提升；应报告实测结果。

## 完整 Rust GR00T 路线还需要什么

若最终目标是 APXInf 完整 Rust GR00T 引擎，应扩展现有 `Gr00tPrecisionExecution` / `DeviceLinearWeights`，新增 NVFP4 权重与 BF16 A/B 旁路类型、load/precision 注册和 Python 策略参数；复用现有 GR00T executor，而不是新造 runtime。还需完整映射权重转置、类别选择、输出 modulation 重排、backbone 的稳定投影名称与 LoRA 对应关系。Q/K/V 的独立 A/B 不应被未经证明地折叠成一组因子；可以先关闭相关融合。算子、图捕获 arena 和残差缓冲的生命周期及复用也需独立验收。

该路线已有 `scripts/bench_gr00t.py`、`examples/gr00t_bench.rs` 与共享 LIBERO evaluator 可扩展；现有 benchmark 仍只接受 BF16/FP8/INT8，且本次检查未发现已编译的 `target/wheel/release/examples/gr00t_bench`。现有 GR00T 文档的阈值和历史平台结果不能替代本机、本权重、本 W4A4＋LoRA 的复测。
