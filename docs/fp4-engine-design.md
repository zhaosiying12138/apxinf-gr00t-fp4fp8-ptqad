# NVFP4 引擎集成设计（fp4_weights / executor / GEMM 接入蓝图）

状态：设计定稿，待实施（RAM/GPU 空闲窗口）。遵循 engine 的
`model-layer-architecture.md` 所有权规则与 fp8 家族模式。

## 1. 工件契约（已由 quant/nvfp4_convert.py 产出，weights/pi05.nvfp4/）

```
<stem>.packed.u8   rows×K/2 字节；E2M1 对（低 nibble=偶数 k），K%16==0
<stem>.scale.u8    物理 swizzle 布局（PS=512·ceil(KB/4)，offset 见 fp4_quant.swizzle_scales）
<stem> 在 manifest.json: shape、block=16、tscale(F32)、rel_frob、字节数
```

加载时 tscale 以 per-tensor FP32 传入 cuBLASLt 的 alpha（或预乘进 D 反缩放），
与 QAD 的"scale 冻结"约束兼容（工件不可变）。

## 2. 新增模块（crates/apxinf-model/src/pi05/，仿 fp8_* 三元组）

- `fp4_weights.rs`：从工件目录构建 `Fp4DeviceWeights`（packed + swizzled scales
  device 常驻 + per-tensor tscale 表）；`from_artifact(dir, &manifest)`；
  校验 checkpoint_identity 对齐 fp8_weights.rs 的 profile 惯例。
- `fp4_executor.rs`：GemmOp::Fp4F16（E2M1×E2M1→F16 输出，probe 已验证该组合在
  sm_120 可用；FP4 输出必然不可用，同 FP8 教训直接不设）。
- `fp4_runtime.rs`：model_variant 新增 `nvfp4_static`；加载路径
  `AutoPolicy.from_pretrained(model_dir, model_variant="nvfp4_static",
  fp4_artifact=<dir>)`——不复制 BF16 权重，直接从工件构建。
- 非 eligible 层（norm/embedding/lm_head/action 首末层，由敏感性分析定）留 BF16：
  fp4_weights 加载时按 manifest 豁免表回退 safetensors BF16。

## 3. GEMM 接入（crates/apxinf-cuda/src/kernels/gemm/）

- providers/cublaslt.rs 新增 `(GemmOp::Fp4F16, TacticBackend::CublasLt)` 分支：
  `fp4::set_cublaslt_fp4_gemm_heuristic(m,n,k,value)` —— 布局沿用 spike 验证的
  TN（A: KxM ld=K opT；B: KxN ld=K opN；C: MxN F16 ld=M），scale 指针 =
  VEC16_UE4M3 模式（CUDA 13 无 OUTER 变体，见 spike-notes swizzle 推导）。
- tuning key：activation_dtype=E2M1(F4)、weight_dtype=E2M1、output=F16、
  scale_mode=Block16Vec（新增枚举，区别于 PerTensor）。
- 激活侧量化：per-16 block E4M3 scale 的逐层校准扩展 calibrate 脚本（fp8 框架
  256 个 per-tensor scale → fp4 需 per-block，但硬件按 block 自带，校准只决定
  激活侧 tscale；块内 scale 在线由 kernel 算或校准时直方图化——设计取后者，
  精度优先）。
- 风险与回退：若 cuBLASLt FP4 激活路径在真实形状不稳（spike 只测了 GEMM 域），
  降级 W4A16：权重侧 FP4 解包 kernel + BF16 激活（访存收益仍在，spike b1 数据
  支持 ~520GB/s 等效带宽）。

## 4. 校准扩展（calibrate_pi05.py 家族）

- `--format nvfp4`：逐层记录激活 amax 直方图（per-block 上界），输出
  `calibration_fp4.json`：per-tensor 激活 tscale + 每 block 的 E4M3 表。
- 数据源复用 LIBERO 采集路径（32 obs 起步，敏感性网格 calib-abl 到 1024）。

## 5. 评测挂钩

- bench_engine.py `--variant nvfp4_static` 直接可用（kwarg 透传）。
- 敏感性网格：manifest 豁免表按 exp/configs/sensitivity.json 的 pt4-* 配置生成
  不同工件（转换器加 `--allow/--deny` 过滤）。

## 6. 实施顺序（每步可独立验证）

1. apxinf-cuda: Fp4F16 GEMM provider + 单测（随机形状 vs CPU 参考，bit-exact
   口径同 spike）→ `cargo test -p apxinf-cuda`（RAM ~2GB）
2. fp4_weights.rs + 加载单测（工件往返）
3. fp4_executor/runtime 接线 → maturin 重编 wheel
4. bench_engine nvfp4_static 冒烟（数值 sanity：动作分布 vs BF16 KL）
5. LIBERO 评测进敏感性网格
