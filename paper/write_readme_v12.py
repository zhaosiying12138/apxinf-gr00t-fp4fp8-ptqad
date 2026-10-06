#!/usr/bin/env python3
"""Write v12 project and paper README files from verified evidence.

The writer is fail-closed: it refuses to render while the published evidence
still points at an old protocol, has a partial arm, or lacks one of the 17
registered terminal captures. It never starts an experiment.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import readme_v12_metrics

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
EVIDENCE = PAPER / "evidence"
EXPECTED_PROTOCOL_ID = "w4a4-recovery-v12-rtn"
EXPECTED_VERSION = 12
ARMS = [("BF16", "bf16"), ("W4A4 PTQ", "ptq"), ("PTQ + QAD", "qad"),
        ("continued-QAD", "continued_qad"), ("PTQ + QAD + OPD", "qad_opd")]
CONTRASTS = [("PTQ − BF16", "ptq_vs_bf16"), ("QAD − PTQ", "qad_vs_ptq"),
             ("OPD − QAD", "opd_vs_qad"),
             ("OPD − continued-QAD", "opd_vs_continued_qad")]
CAPTURE_ORDER = ["shot_bake", "shot_collect", "shot_evalserver", "shot_fp8probe",
                 "shot_gemm", "shot_gr00t", "shot_nvfp4", "shot_opbench", "shot_opd",
                 "shot_opdcache", "shot_packed", "shot_pi05", "shot_probe", "shot_qad",
                 "shot_qat", "shot_rollout", "shot_verify"]


def fail(message: str) -> None:
    raise SystemExit("write_readme_v12: " + message)


def read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing evidence file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON {path}: {exc}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def arm(final: dict[str, Any], key: str) -> dict[str, Any]:
    item = final.get("public_arms", {}).get(key)
    if item is None:
        item = final.get("control", {}).get(key)
    require(isinstance(item, dict), f"final_results is missing arm {key}")
    return item


def png_size(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    require(raw[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}")
    require(raw[12:16] == b"IHDR" and len(raw) >= 24, f"missing PNG IHDR: {path}")
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


def verify() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    final = read(EVIDENCE / "final_results.json")
    # The shared release contract binds the repository protocol SHA, selected
    # RTN recipe and all five arm/task counters.  Keep this call here even
    # though the checks below also validate publication-local files: it makes
    # the writer and validate_publication.py agree on the same root protocol.
    try:
        from v12_publication_contract import validate_v12_results
        validate_v12_results(final, root=ROOT)
    except ImportError:
        fail("paper/v12_publication_contract.py is missing")
    except ValueError as exc:
        fail(str(exc))
    protocol_path = EVIDENCE / "protocol" / "recovery_protocol_v12_rtn_w4a4.json"
    protocol = read(protocol_path)
    manifest = read(EVIDENCE / "final_manifest.json")
    require(final.get("format") == "publication_final_results_v1", "final_results is not publication format")
    require(final.get("status") == "complete", "final_results is not complete")
    require(protocol.get("id") == EXPECTED_PROTOCOL_ID and protocol.get("version") == EXPECTED_VERSION,
            "published protocol is not v12 RTN")
    require(protocol.get("w4a4") is True, "published protocol is not W4A4")
    protocol_sha = sha(protocol_path)
    require(final.get("source", {}).get("protocol", {}).get("sha256") == protocol_sha,
            "final_results protocol SHA differs from evidence protocol")
    require(manifest.get("protocol_sha256") == protocol_sha,
            "final_manifest protocol SHA differs from evidence protocol")
    require(str(manifest.get("format", "")).startswith("w4a4_recovery_v12_final_manifest"),
            "final_manifest is not the v12 release manifest")
    for _, key in ARMS:
        row = arm(final, key)
        require(row.get("episodes") == 160, f"{key} must contain 160 held-out episodes")
        require(isinstance(row.get("episodes"), int), f"{key} episode count is invalid")
        require("per_task" in row and len(row["per_task"]) == 10,
                f"{key} must contain ten per-task rows")
        require(sum(x.get("episodes", 0) for x in row["per_task"].values()) == 160,
                f"{key} per-task episodes do not sum to 160")
    required = {key for _, key in ARMS}
    present = set(final.get("public_arms", {})) | set(final.get("control", {}))
    require(required <= present, "all five final arms are required")
    try:
        from readme_v12_provenance import validate_capture_provenance
        validate_capture_provenance(ROOT)
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        fail(f"screenshot provenance rejected: {exc}")
    inventory = read(EVIDENCE / "recipe_inventory.json")
    recipe_name = inventory.get("recipe")
    recipe = inventory.get("recipes", {}).get(recipe_name)
    require(isinstance(recipe, dict), "recipe inventory lacks selected recipe")
    residual = inventory.get("recovery_residual", {})
    require(isinstance(residual.get("target_bytes"), int), "recipe inventory lacks adapter bytes")
    return final, protocol, inventory, protocol_sha


def result_tables(final: dict[str, Any]) -> tuple[str, str, str]:
    rows = ["| 配置 | 成功回合 | macro_success_rate |", "|---|---:|---:|"]
    for label, key in ARMS:
        item = arm(final, key)
        macro = item.get("macro_success_rate", item["success_rate"])
        rows.append(f"| {label} | {item['successes']}/{item['episodes']} | {macro * 100:.2f}% |")
    uncertainty = final.get("uncertainty", {}).get("contrasts", {})
    paired = ["| 比较 | 点差 / pp | 95% 配对 CI / pp | 不一致回合 | 精确 McNemar p | Holm p |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, key in CONTRASTS:
        item = uncertainty.get(key)
        require(isinstance(item, dict), f"missing paired contrast {key}")
        lo, hi = item["pointwise_ci_pp"]
        paired.append(f"| {label} | {item['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                      f"{item['discordant_episodes']} | {item['exact_mcnemar_two_sided_p']:.4g} | "
                      f"{item['holm_adjusted_p']:.4g} |")
    tasks = list(arm(final, "bf16")["per_task"])
    task_rows = ["| 任务 | " + " | ".join(x[0] for x in ARMS) + " |",
                 "|---|" + "---:|" * len(ARMS)]
    for task in tasks:
        cells = [f"{arm(final, key)['per_task'][task]['successes']}/{arm(final, key)['per_task'][task]['episodes']}"
                 for _, key in ARMS]
        task_rows.append("| `" + task + "` | " + " | ".join(cells) + " |")
    return "\n".join(rows), "\n".join(paired), "\n".join(task_rows)


def training_table(protocol_sha: str) -> str:
    path = EVIDENCE / "training" / "costs.json"
    require(path.is_file(), "paper/evidence/training/costs.json is required; refusing zero/placeholder costs")
    data = read(path)
    require(data.get("status") == "complete", "training cost evidence is incomplete")
    require(data.get("protocol_sha256") == protocol_sha,
            "training cost evidence protocol differs from final results")
    require(set(data.get("training", {})) == {"qad", "continued_qad", "qad_opd"},
            "training cost evidence must contain QAD, continued-QAD and QAD+OPD")
    rows = ["| 阶段 | 更新数 | 演示窗口读取 | 探针学生反传 | 阶段耗时 / h | 显存峰值 / GiB |",
            "|---|---:|---:|---:|---:|---:|"]
    for key, label in (("qad", "QAD"), ("continued_qad", "continued-QAD"), ("qad_opd", "QAD + OPD")):
        item = data.get("training", {}).get(key)
        require(isinstance(item, dict), f"missing training cost stage {key}")
        require(item.get("status") == "completed", f"training cost stage {key} is incomplete")
        require(int(item.get("optimizer_steps", 0)) > 0 and int(item.get("demonstration_window_draws", 0)) > 0,
                f"training cost stage {key} has zero/unknown update or draw count")
        require(float(item.get("wall_seconds", 0)) > 0, f"training cost stage {key} has no measured wall time")
        require(item.get("cuda_peak_memory"), f"training cost stage {key} has no allocator peak")
        peak = max((x.get("max_memory_reserved_bytes", 0) for x in item.get("cuda_peak_memory", [])), default=0) / 2**30
        require(peak > 0, f"training cost stage {key} has zero allocator peak")
        rows.append(f"| {label} | {item.get('optimizer_steps', '—')} | {item.get('demonstration_window_draws', '—')} | "
                    f"{item.get('scheduled_teacher_backward_passes', 0)} | {item.get('wall_seconds', 0)/3600:.3f} | {peak:.2f} |")
    return "\n".join(rows)


def capture_table() -> str:
    # A direct call is a release entry point too: do not render old screenshots
    # merely because the caller bypassed verify(). Recheck hashes, not a cache.
    try:
        from readme_v12_provenance import validate_capture_provenance
        provenance = validate_capture_provenance(ROOT)
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        fail(f"screenshot provenance rejected: {exc}")
    require(set(provenance["capture_rows"]) == set(CAPTURE_ORDER),
            "verified screenshot registry differs from README order")
    labels = {
        "shot_bake": "RTN 配方与编码预算", "shot_collect": "BF16 教师轨迹采集",
        "shot_packed": "π0.5 NVFP4 打包", "shot_probe": "探针和尾批梯度",
        "shot_qad": "W4A4 QAD 训练", "shot_opdcache": "学生状态教师缓存",
        "shot_opd": "QAD→OPD 续训", "shot_qat": "激活重算与 LoRA 梯度",
        "shot_evalserver": "W4A4 服务健康 RPC", "shot_rollout": "LIBERO 闭环",
        "shot_verify": "量化格点和 checkpoint 校验", "shot_fp8probe": "FP8 描述符探针",
        "shot_gemm": "9 种 GEMM 形状", "shot_opbench": "18 个实际层形状",
        "shot_gr00t": "GR00T BF16 执行", "shot_pi05": "π0.5 BF16 执行",
        "shot_nvfp4": "π0.5 NVFP4 完整路径",
    }
    return "\n".join(f"| {i:02d} | {labels[name]} | ![{labels[name]}](paper/figs/{name}.png) |"
                     for i, name in enumerate(CAPTURE_ORDER, 1))


def engine_table() -> str:
    rows = ["| 路径 | model p50 / ms | total p50 / ms | p50 Hz | 原始记录 |",
            "|---|---:|---:|---:|---|"]
    for label, rel in (("GR00T BF16（APXInf 独立路径）", "results/engine/gr00t_bf16_ptqad_20260929.json"),
                       ("π0.5 BF16（APXInf）", "results/engine/pi05_bf16_ptqad_20260929.json"),
                       ("π0.5 NVFP4（APXInf）", "results/engine/pi05_nvfp4_ptqad_20260929.json")):
        path = ROOT / rel
        if not path.is_file():
            continue
        data = read(path)
        rows.append(f"| {label} | {data['model_ms_p50']:.3f} | {data['total_ms_p50']:.3f} | {data['hz_p50']:.3f} | [`{rel}`]({rel}) |")
    if len(rows) == 2:
        rows.append("|（未登记独立 APXInf 基准）|—|—|—|—|")
    return "\n".join(rows)


def legacy_engine_section() -> str:
    """Retain the full, already-reviewed APXInf metric section.

    The old section contains independent 2026-09-29 engine/GEMM/graph data,
    not the superseded closed-loop arm results.  Read it from the current
    checkout when available and fall back to HEAD so rerunning the writer does
    not progressively delete the detailed tables.
    """
    template = PAPER / "readme_v12_legacy_engine.md"
    source = ROOT / "README.md"
    text = template.read_text(encoding="utf-8") if template.is_file() else (source.read_text(encoding="utf-8") if source.is_file() else "")
    if template.is_file():
        return text.replace("## 独立执行基准", "### 完整独立 APXInf、GEMM 与图执行记录", 1).strip()
    start, end = text.find("## 独立执行基准"), text.find("## 运行环境")
    if not (start >= 0 and end > start):
        try:
            text = subprocess.run(["git", "show", "HEAD:README.md"], cwd=ROOT,
                                  check=True, capture_output=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return ""
        start, end = text.find("## 独立执行基准"), text.find("## 运行环境")
    if not (start >= 0 and end > start):
        return ""
    section = text[start:end].strip()
    # It is nested below the v12 summary and contains no v11 closed-loop
    # numbers. Keep all tables/details while avoiding a duplicate top-level
    # heading in the generated README.
    return section.replace("## 独立执行基准", "### 完整独立 APXInf、GEMM 与图执行记录", 1)


def build_root(final: dict[str, Any], protocol: dict[str, Any], inventory: dict[str, Any], protocol_sha: str) -> str:
    summary, paired, tasks = result_tables(final)
    details = readme_v12_metrics.render_all(ROOT, final, protocol, inventory)
    recipe = inventory["recipes"][inventory["recipe"]]
    residual = inventory["recovery_residual"]["target_bytes"]
    source = recipe["source_tensor_bytes"]
    packed = recipe["target_full_checkpoint_bytes"]
    net = packed + residual
    eligible = recipe.get("eligible_tensor_count", "—")
    fp4_fraction = recipe.get("fraction_of_eligible_params", {}).get("nvfp4", 1.0) * 100
    shots = capture_table()
    legacy_engine = legacy_engine_section()
    installation = (PAPER / "readme_v12_installation.md").read_text(encoding="utf-8").strip()
    engine_commands = (PAPER / "readme_v12_engine_commands.md").read_text(encoding="utf-8").strip()
    capture_commands = (PAPER / "readme_v12_capture_commands.md").read_text(encoding="utf-8").strip()
    return f'''# apxinf-gr00t-fp4fp8-ptqad

APXInf × GR00T：**全 NVFP4 W4A4 RTN-PTQ + QAD/OPD 量化域恢复**。
本项目把 GR00T N1.7 视觉—语言—动作模型的可量化权重和激活压到四位，冻结 W4A4 基座，用成功演示训练低秩 QAD 修正，再在学生访问状态上加入 OPD 教师速度监督，并以 LIBERO-10 的逐回合闭环成功率验收。APXInf 原生 NVFP4/FP8 kernel 与图执行另列为独立算子基准；GR00T 闭环的主结果使用 Torch W4A4 QDQ 参考路径。

## 项目意义与贡献

量化 VLA 的难点是误差会经过动作反馈改变下一次观测。本文固定数值格式、舍入规则和配对初态，完整记录 **RTN W4A4 PTQ → QAD 演示恢复 → OPD 学生状态蒸馏 → 五臂闭环**，使压缩比例、恢复效果、训练预算和复现命令可以逐项核查。

1. 全部 479 个 eligible 张量使用 NVFP4；469 个普通 Linear 与 7 个 CategorySpecificLinear 使用 W4A4 激活 QDQ。
2. QAD/OPD 在 468 个 ordinary Linear 上训练 rank=32、alpha=64 的 LoRA；参数以 FP32 训练，前向使用 BF16，部署预算计入 BF16 残差。`continued-QAD` 以同一追加更新预算作为控制臂。
3. BF16、PTQ、QAD、continued-QAD 和 QAD+OPD 使用同一 LIBERO-10 held-out 初态、任务顺序和 episode 种子；成功率、配对区间和检验只从最终 manifest 读取。
4. APXInf 的 NVFP4/FP8 编译、缩放、padding、图重放、GEMM 和完整 π0.5 路径独立验收，不把独立 kernel 延迟冒充 GR00T 闭环速度。

## v12 最终指标

协议 `{protocol['id']}`（version {protocol['version']}，W4A4=true），SHA-256 `{protocol_sha}`。每臂 10 个任务 × 16 回合，共 160 回合。

{summary}

### 配对统计

{paired}

CI 是固定任务内的 95% 配对 bootstrap 区间；p 值为精确双侧 McNemar 检验并做 Holm 校正。OPD 的独立作用由 `OPD − continued-QAD` 读取，同时报告 `OPD − QAD`，不把两者混成一个结论。

### 配对 2×2 计数与不确定性参数

{details['paired']}

表中“前者/后者”沿用比较标题顺序：例如 `OPD − continued-QAD` 的“仅前者成功”是 OPD 成功而 continued-QAD 失败。

{details['uncertainty']}

### Held-out 运行收据

{details['heldout']}

### 逐任务结果

{tasks}

### 编码预算

目标编码账本覆盖 **{eligible} 个 eligible 张量，NVFP4 占 {fp4_fraction:.2f}%**。源 BF16 权重为 **{source:,} B**，NVFP4 主分支为 **{packed:,} B**，BF16 LoRA 恢复旁路为 **{residual:,} B**，部署合计 **{net:,} B**，净压缩比 **{source / net:.3f}×**。这是 shape-derived 目标预算；checkpoint 文件本身仍可能以原始 dtype 保存。完整逐层账本见 [`paper/evidence/recipe_inventory.json`](paper/evidence/recipe_inventory.json)。

#### 物理张量与 tied alias 去重口径

{details['compression']}

#### 量化覆盖与 LoRA 旁路细账

{details['quantization']}

{details['lora']}

### 训练成本

{training_table(protocol_sha)}

训练计时包含模型准备和 checkpoint 序列化；显存为进程级 PyTorch allocator 峰值。训练成本与 held-out 成功率分开记账，完整来源见 [`paper/evidence/training/costs.json`](paper/evidence/training/costs.json)。

#### 训练参数、显存与数据准备成本

{details['training_detail']}

{details['collection_search']}

## 独立 APXInf 执行基准

下表只报告独立 APXInf 路径的策略前向延迟，不能替代 GR00T W4A4 的 LIBERO 闭环成功率。GR00T W4A4 当前闭环使用 Torch QDQ；GEMM、18 个实际层形状、数值验收和 graph replay 的完整 CSV/日志见 [`results/engine/`](results/engine/) 与 [`results/native_graph_20260929/`](results/native_graph_20260929/)。

{engine_table()}

{legacy_engine}

## 环境、依赖与安装

目标平台是 WSL2 Ubuntu 22.04/24.04、支持 `sm_120` 的 NVIDIA 驱动和 24 GB 以上 Blackwell GPU。训练服务、LIBERO rollout 和 APXInf 原生引擎使用相互独立的 Python 环境；不要把它们混装。

| 用途 | 主要依赖 |
|---|---|
| 训练与服务 | Python 3.12、PyTorch/CUDA、transformers、accelerate、peft、safetensors、numpy、msgpack、pyzmq、ffmpeg、GR00T |
| LIBERO | 独立 Python 3.12、robosuite、MuJoCo、gym、EGL/OpenGL |
| APXInf（可选） | CUDA toolkit/cuBLASLt、`nvcc`、Rust/Cargo、C++、Make |
| 论文构建 | Python 3、uv、Node.js ≥18、Playwright、CairoSVG/Pillow、Noto Sans CJK |

恢复、仿真和媒体环境的精确依赖版本，以及源码 revision、权重来源和哈希，见 [`setup/locks/manifest.json`](setup/locks/manifest.json)。原生 APXInf 构建的部分 Python 工具未锁定版本，复现时需保留安装日志；本文不将其称为完全锁定的构建环境。

{installation}

## 从编译到执行

以下命令均来自仓库实际脚本；大模型、校准缓存、训练 checkpoint 和教师张量不进入 Git。先定义新运行目录，避免把旧实验混入证据：

```bash
export PROJECT="$PWD"
source setup/recovery-env.sh
export PY="$PTQAD_PYTHON" GR00T="$GR00T_REPO" LIBERO_PY="$LIBERO_PYTHON"
export BASE="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"
export DATASET="$GR00T/demo_data/libero_demo"
export RUN="$PROJECT/results/reruns/rtn_w4a4_release_$(date -u +%Y%m%dT%H%M%SZ)"
export PROTOCOL_SRC="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"
export PROTOCOL="$PROTOCOL_SRC"
export RECOVERY="$RUN/recovery_v12"
export PTQAD_ZMQ_TIMEOUT_MS=120000
mkdir -p "$RUN"
```

### 0. CPU 契约和源码静态检查

```bash
"$PY" paper/run_cpu_checks.py --out "$PROJECT/paper/_build/cpu-checks-$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" -m py_compile quant/ptq/bake.py quant/ptq/bake_category.py \\
  eval/run_recovery_eval.py exp/run_w4a4_recovery.py exp/make_w4a4_selection.py
```

### 1. W4A4 RTN PTQ 与 CategorySpecificLinear

`quant/ptq/bake.py` 的 `rtn` 配方不读取 Hessian，直接生成全 NVFP4 普通层；`bake_category.py --method rtn_all` 补齐七个 category bank。

```bash
export PURE_RTN="$RUN/pure_rtn" CATEGORY_OUT="$RUN/rtn_category"
"$PY" quant/ptq/bake.py --base "$BASE" --out "$PURE_RTN" --recipe rtn --calibration-mode none
"$PY" quant/ptq/bake_category.py --parent "$PURE_RTN" \\
  --out "$CATEGORY_OUT" --method rtn_all --rtn-clip 1.0
```

### 2. development、教师轨迹和选择文件

冻结协议只允许 development 选择压力臂；教师轨迹和学生 collection 只用于训练，held-out 不得参与选择。

```bash
export DEV="$RUN/development" TEACHER="$RUN/teacher_supervision_v12_clean" COLLECTION="$RUN/collection" SELECTION="$RUN/selection_final"
mkdir -p "$DEV" "$COLLECTION" "$SELECTION"
# A new bake lives under RUN, so standalone reproduction uses a local protocol
# copy whose pressure checkpoint path and SHA are bound to this run. The
# checked-in PROTOCOL_SRC remains immutable and is the only input accepted by
# the formal publication writer.
cp "$PROTOCOL_SRC" "$RUN/recovery_protocol_v12_rtn_w4a4.local.json"
export PROTOCOL="$RUN/recovery_protocol_v12_rtn_w4a4.local.json"
"$PY" - "$PROTOCOL" "$CATEGORY_OUT" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text())
d["selection"]["pressure_candidate_checkpoints"]["rtn_w4a4_category"] = str(pathlib.Path(sys.argv[2]).resolve())
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\\n")
print("standalone protocol sha256:", hashlib.sha256(p.read_bytes()).hexdigest())
PY
FP4VLA_QUANT=0 FP4VLA_W4A4=0 "$PY" eval/run_recovery_eval.py --checkpoint "$BASE" --out "$DEV/bf16" --purpose development --seed 940000 --episodes 5 --gr00t "$GR00T" --server-python "$PY" --rollout-python "$LIBERO_PY" --port 6920 --protocol-file "$PROTOCOL"
FP4VLA_QUANT=0 FP4VLA_W4A4=1 "$PY" eval/run_recovery_eval.py --checkpoint "$CATEGORY_OUT" --out "$DEV/rtn_w4a4_category" --purpose development --seed 940000 --episodes 5 --gr00t "$GR00T" --server-python "$PY" --rollout-python "$LIBERO_PY" --port 6921 --protocol-file "$PROTOCOL"
python3 exp/make_w4a4_selection.py --protocol-file "$PROTOCOL" --bf16 "$DEV/bf16" --candidate rtn_w4a4_category --candidate-output "$DEV/rtn_w4a4_category" --out "$SELECTION"
FP4VLA_QUANT=0 FP4VLA_W4A4=0 "$PY" eval/run_recovery_eval.py --checkpoint "$BASE" --out "$TEACHER" --purpose teacher_supervision --seed 950000 --episodes 4 --gr00t "$GR00T" --server-python "$PY" --rollout-python "$LIBERO_PY" --port 6922 --protocol-file "$PROTOCOL"
"$PY" exp/verify_teacher_replay.py --root "$TEACHER" --protocol-file "$PROTOCOL" --teacher "$BASE"
```

### 3. QAD、continued-QAD、OPD 与五臂 held-out

恢复驱动会依次执行两个 QAD 学习率、QAD 选择、学生状态 collection、教师缓存、continued-QAD、两个 OPD 权重和最终五臂评测。完整且身份匹配的阶段可复用；半途失败的评测必须保留故障记录并改用新输出目录。当前训练 checkpoint 不含优化器与调度器状态，不能把重启训练称为无损续跑；详见 [故障恢复规则](docs/reproduce-ptqad.md)。

```bash
export PTQAD_ZMQ_TIMEOUT_MS=120000
"$PY" exp/run_w4a4_recovery.py --run-dir "$RECOVERY" --protocol-file "$PROTOCOL" \\
  --ptq-selection "$SELECTION/selection.json" --base "$BASE" --gr00t-repo "$GR00T" \\
  --python "$PY" --rollout-python "$LIBERO_PY" --dataset "$DATASET" \\
  --capture-dataset "$TEACHER" --port-base 6920 --validate-only
"$PY" exp/run_w4a4_recovery.py --run-dir "$RECOVERY" --protocol-file "$PROTOCOL" \\
  --ptq-selection "$SELECTION/selection.json" --base "$BASE" --gr00t-repo "$GR00T" \\
  --python "$PY" --rollout-python "$LIBERO_PY" --dataset "$DATASET" \\
  --capture-dataset "$TEACHER" --port-base 6920 --until all --allow-opd-nonimprovement
```

### 4. 证据、图表、HTML 和知乎稿

只有冻结 v12 发布运行的 `final_manifest.json` complete 且五臂各 160 回合时才材料化；命令不会把超时或半成品写进正文。上一步独立复现使用了 `$RUN/recovery_protocol_v12_rtn_w4a4.local.json`，其 SHA 与仓库冻结协议不同；它可以用于验证方法，但不能直接喂给正式发布工具。正式发布必须切换到本次冻结 v12 运行目录，并令 `PROTOCOL="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"`。

```bash
export RECOVERY="$PROJECT/results/reruns/rtn_w4a4_release_20261006_01/recovery_v12"
export PROTOCOL="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"
export CATEGORY_OUT="$("$PY" - "$RECOVERY/final_manifest.json" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(pathlib.Path(m["selected_pressure_checkpoint"]).resolve())
PY
)"
python3 paper/extract_final_evidence.py --run-dir "$RECOVERY" --out /tmp/v12_final_results.json
export OPD_MERGE="$("$PY" - "$RECOVERY/final_manifest.json" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
p = m.get("selected_opd_model_identity", {{}}).get("path")
if not p:
    raise SystemExit("selected_opd_model_identity.path is missing")
print(pathlib.Path(p).resolve())
PY
)"
python3 paper/build_recipe_inventory_v12.py --checkpoint "$CATEGORY_OUT" --recovery-manifest "$OPD_MERGE/recovery_manifest.json" --recipe-name rtn_all --out /tmp/v12_recipe_inventory.json
python3 paper/materialize_final_evidence.py --final-manifest "$RECOVERY/final_manifest.json" --recipe-inventory /tmp/v12_recipe_inventory.json --out paper/_build/final_bundle_v12 --orchestrator-run "$RECOVERY"
python3 paper/install_final_evidence.py --bundle paper/_build/final_bundle_v12 --archive paper/_build/evidence_archive_v12_previous
python3 paper/install_final_evidence.py --verify paper/evidence
python3 paper/collect_search_costs.py --run-dir "$RECOVERY" --out paper/evidence/search_costs
python3 paper/collect_search_costs.py --verify paper/evidence/search_costs
python3 paper/build_final_frontier.py --final-results paper/evidence/final_results.json --paired-comparison paper/evidence/paired_comparison.json --inventory paper/evidence/recipe_inventory.json --out paper/evidence/frontier_comparison.json
```

安装器只迁移本轮科学证据和截图登记；上面的命令重新归档完整搜索成本。随后从最终五臂的评测清单读取 checkpoint，记录环境、软件包和源码哈希：

```bash
"$PY" - "$RECOVERY" "$GR00T" "$PY" "$LIBERO_PY" <<'PY'
import json, pathlib, subprocess, sys
run, groot, server_python, rollout_python = sys.argv[1:]
final = json.loads((pathlib.Path(run) / "final_manifest.json").read_text())
round_dir = pathlib.Path(final["heldout_round"])
command = [sys.executable, "paper/capture_runtime.py", "--run-dir", run,
           "--out", "paper/evidence/runtime", "--gr00t", groot,
           "--server-python", server_python, "--rollout-python", rollout_python]
for arm in ("bf16", "ptq", "qad", "continued_qad", "qad_opd"):
    record = json.loads((round_dir / f"heldout_{{arm}}" / "eval_manifest.json").read_text())
    command += ["--checkpoint", f"{{arm}}={{record['checkpoint']}}"]
subprocess.run(command, check=True)
PY
```

### 5. 截图登记

最终五臂完成后先生成本轮受控脚本，再用仓库内的 Windows/WSL 采集工具逐张运行；训练和评测进程必须退出，GPU 必须空闲。

```bash
CAPTURE_ROOT="$(dirname "$RECOVERY")/captures/w4a4_final"
python3 paper/prepare_w4a4_captures.py --final-manifest "$RECOVERY/final_manifest.json" --out "$CAPTURE_ROOT"
```

{capture_commands}

截图登记完毕后，再从同一套证据生成文稿与发布包。先前的输出目录或临时文件若已存在，工具会拒绝覆盖；复跑时使用新的 staging 名称并保留来源记录。

```bash
python3 paper/update_v12_release.py
python3 paper/write_readme_v12.py
sudo apt-get install -y libcairo2 fontconfig fonts-noto-cjk
node --version  # Node.js >=18; install from https://nodejs.org/ if absent.
npm install --prefix paper/_build/renderer --save-exact playwright@1.58.2
node paper/_build/renderer/node_modules/playwright/cli.js install --with-deps chromium
python3 paper/make_figs.py && bash paper/figs/render_pngs.sh
uv run --with-requirements paper/requirements-build.txt python paper/build_html.py
python3 paper/export_zhihu.py
node paper/qa_browser.cjs
uv run --with-requirements paper/requirements-build.txt python paper/validate_publication.py
uv run --with-requirements paper/requirements-build.txt python paper/package_publication.py --check
uv run --with-requirements paper/requirements-build.txt python paper/package_publication.py
```

{engine_commands}

## Ubuntu 执行截图

发布包保留 17 张真实 WSL Ubuntu 紫色终端截图，去除 taskbar 后均为 3840×2280。截图是命令执行凭证，最终数字仍以 JSON 和日志为准。

以下为论文 17 图之外的编译补充图；命令、日志和哈希见 [`docs/evidence/ubuntu-compile/manifest.json`](docs/evidence/ubuntu-compile/manifest.json)。

![Ubuntu：三个 CUDA 程序的真实编译输出](docs/images/ubuntu-compile.png)

| 编号 | 环节 | 图片 |
|---:|---|---|
{shots}

受 W4A4 实现影响的截图必须在 GPU 空闲时按 `setup/windows_capture/README.md` 的 managed wrapper 重拍；每张图同时登记 `.sh`、`.log`、sidecar、裁剪记录和 SHA-256。

## 目录与复现边界

```text
quant/       NVFP4 格式、校准、RTN/PTQ 配方与写盘
rl/          QAD、学生状态、教师缓存、OPD 与 LoRA 导出
eval/        GR00T 服务、LIBERO 配对闭环和逐回合日志
exp/         v12 冻结协议、选择和恢复驱动
setup/       环境、权重、APXInf 构建和锁文件
paper/       HTML、知乎 Markdown、图表、截图和证据包
docs/        复现与原生引擎说明
```

结果只针对固定 LIBERO-10 任务、官方初态 bank 和本协议训练 seed；不声称未见任务、多 seed 或实机泛化。QAD/OPD 的 BF16 LoRA 旁路已计入净压缩比。APXInf 原生延迟是独立工程基准，不能直接换算为 GR00T W4A4 闭环速度。

## 许可证与引用

本仓库当前没有项目自有 `LICENSE` 文件；代码、补丁、GR00T、APXInf、LIBERO、MuJoCo、模型权重和数据均遵循各自上游许可。引用时请同时保留 [`paper/paper.html`](paper/paper.html)、协议 [`exp/recovery_protocol_v12_rtn_w4a4.json`](exp/recovery_protocol_v12_rtn_w4a4.json) 和最终证据 [`paper/evidence/final_results.json`](paper/evidence/final_results.json)。
'''


def main() -> None:
    final, protocol, inventory, protocol_sha = verify()
    root = build_root(final, protocol, inventory, protocol_sha)
    paper = "# v12 W4A4 论文与证据包\n\n"
    paper += "这里保存离线论文 [`paper.html`](paper.html)、知乎稿 [`zhihu/article.md`](zhihu/article.md)、17 张 Ubuntu 执行截图、v12 RTN W4A4 协议和逐回合证据。工程安装、编译、PTQ、QAD、OPD、闭环评测和发布命令以仓库根目录 [`README.md`](../README.md) 为准。\n\n"
    paper += "| 配置 | 闭环成功率 |\n|---|---:|\n"
    for label, key in ARMS:
        item = arm(final, key)
        paper += f"| {label} | {item['successes']}/{item['episodes']}（{item['success_rate'] * 100:.2f}%） |\n"
    paper += "\n数字唯一来源是 `final_results.json`、`paired_comparison.json`、`final_manifest.json` 和每个 held-out 臂的原始 rollout/server 日志。`validate_publication.py` 会拒绝缺失五臂、旧协议 SHA、非 3840×2280 截图或未完成状态。\n\n"
    paper += "截图登记见 [`evidence/captures.json`](evidence/captures.json) 与 [`evidence/retained_captures.json`](evidence/retained_captures.json)；图表和 HTML/知乎稿必须由根 README 的命令重建，不手工改生成文件。\n"
    (ROOT / "README.md").write_text(root, encoding="utf-8")
    (PAPER / "README.md").write_text(paper, encoding="utf-8")


if __name__ == "__main__":
    main()
