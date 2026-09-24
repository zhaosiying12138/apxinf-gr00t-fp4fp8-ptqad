# FP4-VLA：面向 VLA 模型的 NVFP4 量化与恢复（GR00T N1.7 / π0.5 on RTX 5090 Laptop）

在消费级 Blackwell（RTX 5090 Laptop, sm_120）上，基于 [ApxInf](https://github.com/infinigence/ApxInf) /
[APXinf-robo](https://github.com/RLinf/APXinf-robo) 推理引擎，对 VLA 模型（NVIDIA Isaac GR00T N1.7、
π0.5）做 **NVFP4 block-scaled 量化**的系统性研究：

1. cuBLASLt block-scaled NVFP4 GEMM 在 GeForce sm_120 上的可用性/性能（spike，go/no-go）；
2. 逐模块量化敏感性分析（VLM backbone vs flow-matching DiT action head）→ 混合精度准则；
3. 闭环失效模式刻画 + 权重误差→动作误差→成功率传播链分析；
4. 量化恢复阶梯：PTQ → 校准式 scale 学习 → 蒸馏 + 小规模 RL（一天预算闭环）；
5. 系统测量：延迟分解 / batch 吞吐 / 显存 / 功耗，含首批 GeForce 上的 GR00T N1.7 数据。

## 目录结构

```
setup/        环境搭建、权重/数据下载（一切可恢复数据的入口）
spike/        FP4 go/no-go microbenchmark（cuBLASLt NVFP4/FP8/BF16）
quant/        NVFP4 权重转换、fake-quant 节点、激活校准
exp/          敏感性网格、LIBERO 评测、失效模式与传播链分析
rl/           蒸馏恢复 + 小规模 PPO 恢复（一天预算）
third_party/  APXinf-robo（含引擎子模块），gitignore
weights/      HF 权重，gitignore（setup/03_download_weights.sh 恢复）
data/         数据集/校准集，gitignore（setup/04_download_data.sh 恢复）
results/      实验 JSON/CSV（提交小文件；大产物由脚本重生成）
docs/         实验日志与计划
paper/        中文论文（单 HTML + 知乎导出包）
```

## 复现

```bash
# 1. 环境快照（记录 GPU/驱动/CUDA/依赖版本，写入 results/env/）
bash setup/00_env_report.sh

# 2. 构建引擎（clone APXinf-robo + maturin 编译 sm_120 + Python venv）
bash setup/02_build_engine.sh

# 3. FP4 spike（go/no-go 门）
cd spike && make && ./run.sh   # 结果写入 ../results/spike/

# 4. 权重与数据（可重复执行，断点续传）
bash setup/03_download_weights.sh

# 5. 评测/敏感性/系统测量：见 exp/ 与 docs/PLAN.md
```

硬件假设：RTX 5090 Laptop 24GB（sm_120）、CUDA ≥12.8（本仓库在 13.3 验证）、24 核 CPU、≥64GB RAM、≥150GB 空闲磁盘。

## 许可

代码 Apache-2.0（与上游引擎一致）。第三方权重与数据遵循各自许可。
