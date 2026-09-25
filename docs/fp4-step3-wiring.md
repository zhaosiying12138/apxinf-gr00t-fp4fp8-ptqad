# fp4 步骤 3-5 接线施工图（2026-09-25 读码产出）

## 已核实的引擎结构（pi05 家族）

```
model/model.rs (220 行)
  enum ModelVariant { Fp8Static{model,time_embeddings}, Bf16{..}, Int8Dynamic{..} }
  ├─ name() -> "fp8_static"          ← 新增 "nvfp4_static"
  ├─ input_dtype() -> F16(FP8)/BF16  ← fp4 用 F16（激活侧输出 F16）
  ├─ captured_patch_dtype()          ← fp4: 不做 patch 预量化 → input_dtype
  └─ infer(patches, token_ids, ...)  ← match 分派

  build_fp8_static_model(backend, config, weights, scales)
    -> Fp8StaticBlocks::new(backend, config, weights, scales)
    -> Pi05Model::from_blocks(...)        ← fp4 同构：build_fp4_static_model(
                                             backend, config, fp4_weights, bf16_fallback_weights)

model/blocks/
  bf16.rs 621 行 / fp8_static.rs 1014 行 / int8_dynamic.rs 575 行
  Fp4StaticBlocks 预估 ~1000 行（以 fp8_static.rs 为模板，Linear→adapter 调用）

load.rs (141 行)
  from_safetensors + variant 解析 + calibration.json 惯例（fp8 已验证）
  fp4: model_variant="nvfp4_static" + fp4_artifact 参数（或约定 <model>/fp4/ 目录）
  → Fp4DeviceWeights::from_artifact_dir（已实现+测试）+ 豁免表 → BF16 回退权重
```

## 分四个可编译增量（每周期一个，独立 cargo check 通过）

1. **3a**：`Fp4StaticBlocks` 骨架——仅 VLM backbone 的一个 Linear 走 adapter
   （`apxinf_fp4_gemm_f16` Rust 侧 wrapper：激活在线量化 kernel 或先 BF16 激活
   → 量化 kernel（新增小 CUDA kernel 或复用引擎现有 quantize 算子）；
   权重从 Fp4DeviceWeights 取）；其余层暂时 BF16。编译+单测：单层 GEMM vs CPU。
2. **3b**：补全 backbone 全部 Linear + DiT action head 的 fp4 路径
   （豁免表生效：embedding/norm/lm_head/首末层留 BF16）。
3. **4**：`ModelVariant::Fp4Static` + builder + load 接线 + `model_variant=nvfp4_static`
   解析；maturin 重编 wheel。
4. **5**：`bench_engine.py --variant nvfp4_static` 冒烟（动作分布 sanity vs BF16）
   → LIBERO 成功率首测（敏感性网格 pt4-* 配置）。

## 关键技术点（实现时注意）

- **激活量化**：adapter 要求激活也是 E2M1+swizzle scale → 需要"BF16→NVFP4 在线
  量化"kernel（逐块 amax→E4M3→E2M1 打包+swizzle 写出）。引擎有 fp8 patch 预量化
  先例（captured_patch_dtype F8E4M3 路径）可参考；自写 CUDA kernel ~100 行
  （放 adapters/cublaslt_fp4_adapter.cu 里，模式同 spike 的 CPU 量化）。
- **豁免层的 BF16 权重**：Fp4DeviceWeights 缺失的张量回退 safetensors BF16
  （load.rs 的常规路径），Blocks 里按 tensor 名分派。
- **CUDA Graph**：fp8 路径的 graph 捕获流程照搬；adapter 每次调用重建 descriptor
  ——graph 捕获下 cublasLtMatmul 的 descriptor 建在 capture 前即可（spike 模式
  已验证 graph-friendly？未验证——step 5 冒烟时确认，必要时把 descriptor 缓存
  进 adapter 静态状态，同引擎其他 static adapter）。

## 资源需求

3a/3b 纯写码+cargo check（RAM ~3GB）；4 重编 wheel（RAM ~5GB）；5 需 GPU 窗口
（bench 10min + LIBERO 评测 2-4h 按网格规模）。
