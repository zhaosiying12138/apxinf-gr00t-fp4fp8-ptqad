#!/usr/bin/env python3
"""Generate the concise, evidence-bound project README for the v12 release."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def read(path: str):
    return json.loads((PAPER / "evidence" / path).read_text(encoding="utf-8"))


def row(final, name):
    return final["public_arms"].get(name) or final["control"][name]


def main() -> None:
    final = read("final_results.json")
    protocol = json.loads((PAPER / "evidence" / "protocol" / "recovery_protocol_v12_rtn_w4a4.json").read_text())
    inventory = read("recipe_inventory.json")
    recipe = inventory["recipe"]
    recipe_budget = inventory["recipes"][recipe]
    residual_bytes = inventory["recovery_residual"]["target_bytes"]
    source_bytes = recipe_budget["source_tensor_bytes"]
    packed_bytes = recipe_budget["target_full_checkpoint_bytes"]
    net_bytes = packed_bytes + residual_bytes
    net_compression = source_bytes / net_bytes
    engine_specs = [("GR00T BF16（Torch 基线）", "results/engine/gr00t_bf16_ptqad_20260929.json"),
                    ("π0.5 BF16", "results/engine/pi05_bf16_ptqad_20260929.json"),
                    ("π0.5 NVFP4（APXInf 独立路径）", "results/engine/pi05_nvfp4_ptqad_20260929.json")]
    engine_rows = []
    for label, rel in engine_specs:
        path = ROOT / rel
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            engine_rows.append(f"| {label} | {data['model_ms_p50']:.2f} | {data['total_ms_p50']:.2f} | {data['hz_p50']:.2f} | `{rel}` |")
    engine_table = "\n".join(engine_rows) or "| （无已登记的独立基准） | — | — | — | — |"
    arm_rows = [("BF16", "bf16"), ("W4A4 PTQ", "ptq"), ("PTQ + QAD", "qad"),
                ("continued-QAD", "continued_qad"), ("PTQ + QAD + OPD", "qad_opd")]
    metrics = "\n".join(f"| {label} | {row(final, key)['successes']}/{row(final, key)['episodes']} | {row(final, key)['success_rate']*100:.2f}% |"
                         for label, key in arm_rows)
    shots = sorted(p.stem for p in (PAPER / "figs").glob("shot_*.png"))
    shot_table = "\n".join(f"| `{name}` | [查看截图](paper/figs/{name}.png) |" for name in shots)
    tasks = list(row(final, "bf16")["per_task"])
    task_table = "\n".join("| `" + task + "` | " + " | ".join(
        f"{row(final, key)['per_task'][task]['successes']}/{row(final, key)['per_task'][task]['episodes']}" for _, key in arm_rows) + " |"
        for task in tasks)
    root = f'''# apxinf-gr00t-fp4fp8-ptqad

APXInf × GR00T：**全 NVFP4 W4A4 PTQ + QAD/OPD 量化域恢复**。本项目把视觉—语言—动作模型的权重与激活压到四位，冻结量化基座，用成功演示训练低秩修正，再在学生访问状态上加入 BF16 教师速度监督，并以 LIBERO-10 闭环成功率验收。

## 项目意义

低位宽推理的难点不在于把张量写成四位，而在于量化误差会经过闭环反馈改变下一次观测。项目提供一条可审计链：NVFP4 数值规则 → W4A4 PTQ → QAD 演示恢复 → OPD 学生状态监督 → 逐回合配对评测。GR00T 的行为结果来自 Torch W4A4 QDQ 参考路径；APXInf 原生 NVFP4/FP8 kernel 与图执行是独立的延迟基准，不能混写成 GR00T 已经接入原生 kernel。

## v12 最终指标

协议：`{protocol['id']}`，SHA-256：`{final['source']['protocol']['sha256']}`。479 个 eligible 权重张量全部使用 NVFP4，469 个 ordinary Linear 和 7 个 CategorySpecificLinear 使用 W4A4 激活；QAD/OPD 在 468 个 ordinary Linear 上训练 rank=32、alpha=64 的 BF16 LoRA。

| 配置 | 成功回合 | 成功率 |
|---|---:|---:|
{metrics}

每个任务的原始分子/分母如下（五臂使用同一任务顺序与官方初态）：

| 任务 | BF16 | W4A4 PTQ | QAD | continued-QAD | QAD+OPD |
|---|---:|---:|---:|---:|---:|
{task_table}

配对差值、bootstrap 区间、McNemar 检验和 Holm 校正见 [`paper/evidence/final_results.json`](paper/evidence/final_results.json) 与 [`paper/evidence/paired_comparison.json`](paper/evidence/paired_comparison.json)。编码预算见 [`paper/evidence/recipe_inventory.json`](paper/evidence/recipe_inventory.json)：物理键口径和去共享别名口径均包含 NVFP4 数据、E4M3 块尺度、FP32 二级尺度、未量化张量和恢复旁路。

按 shape-derived 字节账本，NVFP4 主分支为 **{packed_bytes:,} B**，BF16 LoRA 恢复旁路为 **{residual_bytes:,} B**，合计 **{net_bytes:,} B**；相对同一口径 BF16 权重 **{source_bytes:,} B**，净压缩比为 **{net_compression:.3f}×**。原始 JSON 同时给出 eligible-only、full-checkpoint 和去共享别名三种口径。

## 环境需求

- Ubuntu 22.04/24.04（WSL2 可用）、NVIDIA 驱动、CUDA、可用的 24 GB 以上 GPU。
- Python 3.12；训练环境由 `setup/06_install_recovery.sh` 创建，APXInf 构建环境由 `setup/01_install_dev_tools.sh` 和 `setup/02_build_engine.sh` 管理。
- GR00T N1.7 权重、LIBERO-10 官方初态 bank、演示数据；默认路径可在命令中覆盖。
- 评测依赖独立的 GR00T 服务 Python 和 LIBERO rollout Python。`ffmpeg` 必须可执行；脚本会设置 `HF_HUB_OFFLINE=1` 和 `NO_ALBUMENTATIONS_UPDATE=1`，避免评测时触网。

## 安装

```bash
git clone git@github.com:zhaosiying12138/apxinf-gr00t-fp4fp8-ptqad.git
cd apxinf-gr00t-fp4fp8-ptqad
bash setup/00_env_report.sh
bash setup/01_install_dev_tools.sh
bash setup/04_install_libero.sh
bash setup/05_restore_all.sh
bash setup/06_install_recovery.sh
# 需要 APXInf 原生算子时再执行：
bash setup/02_build_engine.sh
bash setup/03_download_weights.sh
```

训练与评测环境由脚本锁定并安装以下运行时组件：PyTorch/CUDA、`transformers`、`accelerate`、`peft`、`safetensors`、`numpy`、`msgpack`/`msgpack-numpy`、`pyzmq`、`mujoco`、LIBERO、OpenCV、`ffmpeg` 和 GR00T；APXInf 原生构建还需要 Rust/Cargo、`uv`、`maturin`、CUDA toolkit 与对应 GPU 架构。具体版本以 `setup/locks/manifest.json`、各环境的 `pip freeze` 和运行时 manifest 为准，避免用系统 Python 混装。

安装完成后先做静态检查：

```bash
python3 -m py_compile eval/rollout_seeded.py eval/run_recovery_eval.py exp/run_w4a4_recovery.py
python3 paper/run_cpu_checks.py
```

## 从 PTQ 到闭环评测

以下命令就是本次 v12 的执行入口。路径必须使用新的运行目录；不要把旧运行目录中的数字复制到正文。

```bash
export PROJECT=$PWD
export PTQAD_MEDIA_LIB=/home/zhaosiying/miniforge3/envs/media7/lib
export PTQAD_ZMQ_TIMEOUT_MS=120000
export PYTHON=/home/zhaosiying/miniforge3/envs/triton-dev/bin/python
export GR00T=/home/zhaosiying/codebase/groot-fsdp2/Isaac-GR00T
export GR00T_PY=$GR00T/.venv/bin/python
export LIBERO_PY=$GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python
export DATASET=$GR00T/demo_data/libero_demo
export RUN=$PROJECT/results/reruns/rtn_w4a4_release_20261006_01/recovery_v12
export SELECTION=$PROJECT/results/reruns/rtn_w4a4_release_20261006_01/selection_final/selection.json
export PROTOCOL=$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json

# 开发集：BF16 46/50，RTN 全覆盖 W4A4 PTQ 41/50；随后运行恢复驱动。
$PYTHON exp/run_w4a4_recovery.py \\
  --run-dir $RUN --protocol-file $PROTOCOL --ptq-selection $SELECTION \\
  --base $PROJECT/weights/GR00T-N1.7-LIBERO/libero_10 \\
  --gr00t-repo $GR00T --python $GR00T_PY --rollout-python $LIBERO_PY \\
  --dataset $DATASET --port-base 6920 \\
  --capture-dataset $PROJECT/results/reruns/rtn_w4a4_release_20261006_01/teacher_supervision_v12_clean \\
  --allow-opd-nonimprovement \\
  --until all
```

驱动按顺序完成两个 QAD 学习率、QAD 选择、学生采集、教师缓存、continued-QAD、两个 OPD 权重、开发选择和最终五臂 held-out。任何阶段中断后，使用相同命令和 `--until qad_dev|qad_selection|recovery_dev|opd_selection|all` 续跑；只有对应 `stage.json` 和 `runtime_metrics.json` 均为 complete 才允许继续。

评测请求的 ZeroMQ 超时由 `PTQAD_ZMQ_TIMEOUT_MS=120000` 控制。上游 GR00T 默认 15 秒，首次 W4A4/DIT 请求可能超过该值；`eval/rollout_seeded.py` 在本地 rollout 进程中注入同一值，并把它写入每个 `eval_manifest.json`。这修复了旧运行中出现的 `zmq.error.Again: Resource temporarily unavailable`，不改变 episode 的重试语义。

## 证据、论文和验证

```bash
# 从完成的 final_manifest 生成发布结果（只读取 held-out，拒绝不完整运行）
rm -f paper/evidence/final_results.json paper/evidence/recipe_inventory.json paper/evidence/frontier_comparison.json
python3 paper/extract_final_evidence.py \\
  --run-dir $RUN --out /tmp/v12_final_results.json

# 生成 shape-derived 编码预算（临时路径，安装证据包时复制进去）
python3 paper/build_recipe_inventory_v11.py \\
  --checkpoint results/reruns/rtn_w4a4_pressure_20261006_01/rtn_category \\
  --recovery-manifest $RUN/artifacts/merge_opd_025/recovery_manifest.json \\
  --recipe-name rtn_all --out /tmp/v12_recipe_inventory.json

# 将完成的 final_manifest、逐回合日志、协议和训练成本材料化并安装到发布证据目录
python3 paper/materialize_final_evidence.py \\
  --final-manifest $RUN/final_manifest.json \\
  --recipe-inventory /tmp/v12_recipe_inventory.json \\
  --out paper/_build/final_bundle_v12 \\
  --orchestrator-run $RUN
python3 paper/install_final_evidence.py \\
  --bundle paper/_build/final_bundle_v12 \\
  --archive paper/evidence.v11.archive
# 完成审计后可删除替换前的归档，发布包只保留 v12 证据
rm -rf paper/evidence.v11.archive

# 安装完成后写入 frontier 的发布路径
python3 paper/build_final_frontier.py \\
  --final-results paper/evidence/final_results.json \\
  --paired-comparison paper/evidence/paired_comparison.json \\
  --inventory paper/evidence/recipe_inventory.json \\
  --out paper/evidence/frontier_comparison.json

# 用 v12 结果重写正文和 README，再构建离线发布包
python3 paper/update_v12_release.py
python3 paper/write_readme_v12.py
python3 paper/make_figs.py
python3 paper/build_html.py
python3 paper/export_zhihu.py
python3 paper/validate_publication.py
python3 paper/package_publication.py --check
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 node paper/qa_browser.cjs
git diff --check
```

`paper/paper.html` 是内嵌 KaTeX、SVG、PNG 的离线论文；`paper/zhihu/article.md` 是同一正文的知乎发布稿。`paper/evidence/` 保存协议副本、逐臂 manifest、逐任务 JSON、原始 rollout/server 日志、训练成本和哈希清单。

## Ubuntu 执行截图

所有图均来自真实 WSL Ubuntu 紫色主题终端，尺寸为 3840×2280（去除 taskbar）。截图只证明命令确实执行；数值以 JSON/日志为准。

| 环节 | 截图 |
|---|---|
{shot_table}

重新拍摄截图时使用 `wsl-ubuntu-screenshot` skill 的 managed wrapper：命令必须写入 WSL 脚本并原样执行，输出 PNG 保存在 Windows 本地盘后再裁剪 taskbar、安装到 `paper/figs/` 并用 `paper/record_capture.py` 登记哈希。训练和评测截图不能与其他 GPU 作业并行。

## 目录说明

```text
exp/                 冻结协议、恢复驱动和编排
quant/ rl/           NVFP4 PTQ、W4A4 QDQ、QAD/OPD 训练实现
eval/                服务、配对重置、闭环评测与统计
paper/               HTML、知乎 Markdown、图表、截图和证据包
setup/               环境、权重、引擎和恢复依赖安装脚本
patches/             APXInf/GR00T 集成补丁
```

## 复现边界

结果针对固定的 LIBERO-10 任务、官方初态 bank 和一个训练 seed；它不声称未见任务、多 seed 或实机泛化。APXInf 的原生 NVFP4/FP8 延迟属于独立工程基准，当前 GR00T 闭环使用 Torch 数值 QDQ。QAD/OPD 的 BF16 LoRA 旁路必须计入净压缩比，不能把“479 个张量全部 NVFP4”写成整个 checkpoint 每个字节都是四位。

## 独立执行基准

下表只报告模型前向和端到端 wrapper 延迟，不替代 LIBERO 闭环成功率，也不把 APXInf π0.5 路径写成 GR00T 已接入原生 kernel。

| 路径 | model p50 (ms) | total p50 (ms) | p50 Hz | 原始 JSON |
|---|---:|---:|---:|---|
{engine_table}

## 许可证与引用

代码和补丁按仓库许可证发布；GR00T、APXInf、LIBERO、MuJoCo 和相关模型权重遵循各自上游许可证。引用方法时请同时引用 [`paper/paper.html`](paper/paper.html)、协议 [`exp/recovery_protocol_v12_rtn_w4a4.json`](exp/recovery_protocol_v12_rtn_w4a4.json) 和最终证据 [`paper/evidence/final_results.json`](paper/evidence/final_results.json)。
'''
    (ROOT / "README.md").write_text(root, encoding="utf-8")
    paper_readme = f'''# v12 W4A4 论文与证据包

这里保存离线论文 [`paper.html`](paper.html)、知乎稿 [`zhihu/article.md`](zhihu/article.md)、17 张 Ubuntu 执行截图、协议副本和逐回合证据。正文只使用 v12 RTN 全覆盖 W4A4 实验；工程安装、训练、评测和发布命令以根目录 [`README.md`](../README.md) 为准。

| 配置 | 闭环成功率 |
|---|---:|
| BF16 | {row(final, 'bf16')['successes']}/{row(final, 'bf16')['episodes']}（{row(final, 'bf16')['success_rate']*100:.2f}%） |
| W4A4 PTQ | {row(final, 'ptq')['successes']}/{row(final, 'ptq')['episodes']}（{row(final, 'ptq')['success_rate']*100:.2f}%） |
| PTQ + QAD | {row(final, 'qad')['successes']}/{row(final, 'qad')['episodes']}（{row(final, 'qad')['success_rate']*100:.2f}%） |
| continued-QAD | {row(final, 'continued_qad')['successes']}/{row(final, 'continued_qad')['episodes']}（{row(final, 'continued_qad')['success_rate']*100:.2f}%） |
| PTQ + QAD + OPD | {row(final, 'qad_opd')['successes']}/{row(final, 'qad_opd')['episodes']}（{row(final, 'qad_opd')['success_rate']*100:.2f}%） |

`final_results.json`、`paired_comparison.json` 和每个 held-out 臂的原始 rollout/server 日志构成唯一数字来源；截图是命令执行凭证，不替代 JSON 统计。
'''
    (PAPER / "README.md").write_text(paper_readme, encoding="utf-8")


if __name__ == "__main__":
    main()
