#!/usr/bin/env python3
"""Render the detailed, read-only metric blocks used by the v12 README.

This module deliberately does not know how to run an experiment.  It only
reads the publication bundle after the v12 evidence installer has completed.
The writer imports these functions when it builds the README; keeping the
accounting here prevents a hand-edited summary from silently dropping a
denominator, a pairing cell, or a cost scope.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ARMS = [("BF16", "bf16"), ("W4A4 PTQ", "ptq"), ("PTQ + QAD", "qad"),
        ("continued-QAD", "continued_qad"), ("PTQ + QAD + OPD", "qad_opd")]
CONTRASTS = [("PTQ − BF16", "ptq_vs_bf16"), ("QAD − PTQ", "qad_vs_ptq"),
             ("OPD − QAD", "opd_vs_qad"),
             ("OPD − continued-QAD", "opd_vs_continued_qad")]


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError("README metrics: " + message)


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"missing/nonfinite {field}")
    _require(value > 0 if positive else value >= 0, f"invalid {field}")
    return float(value)


def _arm(final: dict[str, Any], key: str) -> dict[str, Any]:
    row = final.get("public_arms", {}).get(key)
    if row is None:
        row = final.get("control", {}).get(key)
    if not isinstance(row, dict):
        raise ValueError(f"final evidence is missing arm {key}")
    return row


def _fmt_bytes(value: int | float | None) -> str:
    if value is None:
        return "—"
    return f"{int(value):,} B"


def _fmt_seconds(value: int | float | None) -> str:
    return "—" if value is None else f"{float(value):,.3f} s"


def heldout_table(root: Path, final: dict[str, Any]) -> str:
    """Show the per-arm receipt fields alongside the published score."""
    rows = ["| 臂 | 成功率（macro/micro） | wall / s | seed | purpose | 完成任务 |"]
    rows.append("|---|---:|---:|---:|---|---:|")
    for label, key in ARMS:
        folder = root / "paper" / "evidence" / f"heldout_{key}"
        manifest = _read(folder / "eval_manifest.json")
        summary = _read(folder / "summary.json")
        item = _arm(final, key)
        tasks = summary.get("tasks_complete", manifest.get("task_count", "—"))
        purpose = manifest.get("purpose", summary.get("purpose", "—"))
        seed = manifest.get("seed", summary.get("seed", "—"))
        wall = summary.get("wall_seconds_including_server_loads", summary.get("wall_seconds"))
        _number(wall, f"heldout_{key}.wall_seconds", positive=True)
        _require(purpose == "heldout" and tasks == 10, f"incomplete heldout_{key}")
        _require(summary.get("total_episodes") == item["episodes"] and
                 summary.get("total_successes") == item["successes"], f"heldout_{key} differs from final results")
        macro = item.get("macro_success_rate", item.get("success_rate"))
        micro = summary.get("micro_success_rate", item.get("success_rate"))
        rows.append(f"| {label} | {float(macro)*100:.2f}% / {float(micro)*100:.2f}% | "
                    f"{_fmt_seconds(wall)} | {seed} | `{purpose}` | {tasks}/10 |")
    return "\n".join(rows)


def paired_table(root: Path) -> str:
    """Render all four 2×2 paired episode tables from paired_comparison.json."""
    paired = _read(root / "paper" / "evidence" / "paired_comparison.json")
    rows = ["| 比较 | 两者成功 | 仅前者成功 | 仅后者成功 | 两者失败 | 不一致回合 |",
            "|---|---:|---:|---:|---:|---:|"]
    for label, key in CONTRASTS:
        item = paired.get(key, {})
        matrix = item.get("table_rows_baseline_success_fail_columns_treatment_success_fail")
        if isinstance(matrix, list) and len(matrix) == 2 and len(matrix[0]) == 2 and len(matrix[1]) == 2:
            both, only_baseline = matrix[0]
            only_treatment, both_fail = matrix[1]
        else:
            both = item.get("both_success", "—")
            only_treatment = item.get("only_treatment_success", "—")
            only_baseline = item.get("only_baseline_success", "—")
            both_fail = item.get("both_fail", "—")
        discordant = item.get("discordant_count", item.get("discordant_episodes", "—"))
        # Labels are treatment minus baseline, so the first policy is treatment.
        _require(all(type(x) is int and x >= 0 for x in (both, only_treatment, only_baseline, both_fail)),
                 f"invalid paired cells for {key}")
        _require(both + only_treatment + only_baseline + both_fail == item.get("paired_count") and
                 only_treatment + only_baseline == discordant, f"paired cell totals differ for {key}")
        rows.append(f"| {label} | {both} | {only_treatment} | {only_baseline} | {both_fail} | {discordant} |")
    return "\n".join(rows)


def uncertainty_block(root: Path, final: dict[str, Any]) -> str:
    """Report the frozen bootstrap and multiplicity settings before the CIs."""
    unc = final.get("uncertainty", {})
    ci = unc.get("confidence_interval", {})
    tests = unc.get("tests", {})
    _require(all(k in ci for k in ("method", "level", "replicates", "seed", "quantile")),
             "final results lack bootstrap parameters")
    _require(all(k in tests for k in ("method", "alpha", "multiplicity")), "final results lack test parameters")
    return (f"- 区间方法：`{ci.get('method', '—')}`，置信度 {float(ci.get('level', 0))*100:.1f}%，"
            f"重复次数 {ci.get('replicates', '—')}，seed `{ci.get('seed', '—')}`，"
            f"分位数规则 `{ci.get('quantile', '—')}`。\n"
            f"- 检验方法：`{tests.get('method', '—')}`；显著性水平 `{tests.get('alpha', '—')}`；"
            f"多重比较 `{tests.get('multiplicity', '—')}`。")


def compression_table(inventory: dict[str, Any]) -> str:
    """Show physical and known-alias-deduplicated encoding budgets together."""
    recipe = inventory["recipes"][inventory["recipe"]]
    alias = recipe.get("known_tied_alias_deduplicated", {})
    residual = int(inventory.get("recovery_residual", {}).get("target_bytes", 0))
    rows = ["| 口径 | BF16 源权重 | W4A4 基座 | 基座压缩比 | LoRA 旁路 | 部署合计 | 净压缩比 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    entries = [("物理 checkpoint 张量", recipe.get("source_tensor_bytes"),
                recipe.get("target_full_checkpoint_bytes")),
               ("已知 tied alias 去重", alias.get("source_tensor_bytes"),
                alias.get("target_full_bytes"))]
    for name, source, target in entries:
        _number(source, name + ".source bytes", positive=True)
        _number(target, name + ".target bytes", positive=True)
        total = (int(target) + residual) if target is not None else None
        ratio = (int(source) / total) if source and total else None
        rows.append(f"| {name} | {_fmt_bytes(source)} | {_fmt_bytes(target)} | {source/target:.3f}× | {_fmt_bytes(residual)} | "
                    f"{_fmt_bytes(total)} | {'—' if ratio is None else f'{ratio:.3f}×'} |")
    return "\n".join(rows)


def quant_lora_tables(root: Path, protocol: dict[str, Any], inventory: dict[str, Any]) -> tuple[str, str]:
    recipe = inventory["recipes"][inventory["recipe"]]
    scope = protocol.get("quantization_scope", {})
    qrows = ["| 项目 | 数值 |", "|---|---:|"]
    for label, value in (
        ("eligible 权重张量", recipe.get("eligible_tensor_count", scope.get("eligible_tensor_count", "—"))),
        ("ordinary 配方权重张量（含 embedding/position）", scope.get("ordinary_linear_eligible", "—")),
        ("ordinary Linear（激活 QDQ）", scope.get("ordinary_linear_operators", "—")),
        ("CategorySpecificLinear", scope.get("category_linear_layers", "—")),
        ("激活量化 Linear 总数", scope.get("activation_linear_count", "—")),
        ("NVFP4 权重元素", recipe.get("nvfp4_params", "—")),
        ("FP8 权重元素", recipe.get("fp8_params", 0)),
        ("BF16 权重元素", recipe.get("bf16_params", 0)),
        ("未量化张量", recipe.get("excluded_tensor_count", "—")),
    ):
        qrows.append(f"| {label} | {value:,} |" if isinstance(value, int) else f"| {label} | {value} |")
    alias = recipe.get("known_tied_alias_deduplicated", {})
    for label, value in (
        ("物理口径 NVFP4 / eligible", recipe.get("fraction_of_eligible_params", {}).get("nvfp4")),
        ("物理口径 NVFP4 / 全部元素", recipe.get("fraction_of_all_checkpoint_params", {}).get("nvfp4")),
        ("alias 去重 NVFP4 / eligible", alias.get("fraction_of_eligible", {}).get("nvfp4")),
        ("alias 去重 NVFP4 / 全部元素", alias.get("fraction_of_all", {}).get("nvfp4")),
    ):
        _number(value, label)
        qrows.append(f"| {label} | {value*100:.4f}% |")
    qrows.extend([
        f"| 物理口径量化编码（含缩放） | {_fmt_bytes(recipe.get('target_eligible_bytes'))} |",
        f"| 未量化源权重 | {_fmt_bytes(recipe.get('excluded_source_bytes'))} |",
        f"| alias 去重去掉的重复元素 | {alias.get('omitted_physical_alias_elements', '—'):,} |",
    ])
    category = recipe.get("category", {})
    for field, label in (("weights", "category 全部权重元素"), ("banks_per_layer", "每层 category banks"),
                         ("active_libero_bank", "LIBERO 激活 bank"),
                         ("active_libero_weight_elements", "LIBERO 激活 bank 权重元素"),
                         ("padding_elements", "category K-padding 元素"),
                         ("tensor_scale_count", "category 张量缩放数"), ("target_bytes", "category 编码字节")):
        value = category.get(field)
        _number(value, "category." + field)
        qrows.append(f"| {label} | {value:,} |")
    residual = inventory.get("recovery_residual", {})
    lrows = ["| LoRA 项目 | 数值 |", "|---|---:|"]
    for label, value in (("scope", residual.get("scope", "—")), ("Linear 模块", residual.get("linear_modules", "—")),
                         ("rank", residual.get("rank", "—")), ("alpha", residual.get("alpha", "—")),
                         ("可训练参数", residual.get("tensor_elements", "—")),
                         ("dtype", residual.get("dtype", "—")), ("编码字节", _fmt_bytes(residual.get("target_bytes")))):
        lrows.append(f"| {label} | {value:,} |" if isinstance(value, int) else f"| {label} | {value} |")
    return "\n".join(qrows), "\n".join(lrows)


def training_detail(root: Path, protocol: dict[str, Any], final: dict[str, Any]) -> str:
    costs = _read(root / "paper" / "evidence" / "training" / "costs.json")
    selected = costs.get("selected_recovery_settings", {})
    rows = ["| 阶段 | 学习率 | OPD 权重 | seed | 更新数 | micro/accum/effective | dtype | 参数存储 | rank/alpha | grad clip | F16 饱和 | 可训练参数 | allocated / reserved GiB |", "|---|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|"]
    for key, label in (("qad", "QAD"), ("continued_qad", "continued-QAD"), ("qad_opd", "QAD + OPD")):
        item = costs.get("training", {}).get(key, {})
        stage = root / "paper" / "evidence" / "training" / "stages" / key / "recovery_manifest.json"
        manifest = _read(stage) if stage.is_file() else {}
        alloc = max((x.get("max_memory_allocated_bytes", 0) for x in item.get("cuda_peak_memory", [])), default=0) / 2**30
        reserve = max((x.get("max_memory_reserved_bytes", 0) for x in item.get("cuda_peak_memory", [])), default=0) / 2**30
        lr = manifest.get("learning_rate", selected.get("learning_rate", "—"))
        weight = manifest.get("probe_weight", selected.get("opd_weight", 0))
        seed = manifest.get("train_seed", selected.get("train_seed", "—"))
        triple = "/".join(str(item.get(x, manifest.get(x, "—"))) for x in ("micro_batch", "gradient_accumulation_steps", "effective_global_batch"))
        rank_alpha = f"{manifest.get('rank', '—')}/{manifest.get('alpha', '—')}"
        clip = manifest.get("max_grad_norm", "未记录（阶段收据）")
        saturation = manifest.get("f16_activation_saturation", "未记录（阶段收据）")
        trainable = item.get("trainable_parameters", manifest.get("trainable_parameters", "—"))
        compute_dtype = item.get("compute_dtype", manifest.get("compute_dtype", "未记录（阶段收据）"))
        storage = item.get("parameter_storage_by_dtype", {})
        storage_text = ", ".join(f"{dtype}: {info.get('parameters', '—'):,} 参数 / {_fmt_bytes(info.get('bytes'))}"
                                 for dtype, info in storage.items()) or "未记录（阶段收据）"
        rows.append(f"| {label} | {lr} | {weight} | {seed} | {item.get('optimizer_steps', '—')} | {triple} | {compute_dtype} | {storage_text} | {rank_alpha} | {clip} | {saturation} | {trainable:,} | {alloc:.2f} / {reserve:.2f} |" if isinstance(trainable, int) else
                    f"| {label} | {lr} | {weight} | {seed} | {item.get('optimizer_steps', '—')} | {triple} | {compute_dtype} | {storage_text} | {rank_alpha} | {clip} | {saturation} | {trainable} | {alloc:.2f} / {reserve:.2f} |")
    return "\n".join(rows)


def collection_search_costs(root: Path) -> str:
    costs = _read(root / "paper" / "evidence" / "training" / "costs.json")
    search = root / "paper" / "evidence" / "search_costs" / "summary.json"
    _require(search.is_file(), "search cost summary is missing")
    search_data = _read(search)
    collection = costs.get("collection", {})
    teacher = costs.get("teacher_labeling", {})
    supervision = search_data.get("teacher_supervision", {})
    student = search_data.get("student_collection", {})
    rows = ["| 成本项 | 次数/样本 | wall / s | allocated / reserved GiB |", "|---|---:|---:|---:|"]
    rows.append(f"| 教师成功轨迹采集 | {supervision.get('successful_trajectories', '未记录')} 条 / {supervision.get('training_windows', '未记录')} 窗口 | {_fmt_seconds(supervision.get('wall_seconds'))} | 未记录（采集 wrapper） |")
    rows.append(f"| 学生 collection | {student.get('episodes', collection.get('episodes', '未记录'))} 回合 / {student.get('captured_observations', collection.get('captured_observations', '未记录'))} 观测 | {_fmt_seconds(student.get('wall_seconds', collection.get('wall_seconds')))} | 未记录（采集 wrapper） |")
    peak = teacher.get("cuda_peak_memory", [])
    alloc = max((x.get("max_memory_allocated_bytes", 0) for x in peak), default=0) / 2**30
    reserve = max((x.get("max_memory_reserved_bytes", 0) for x in peak), default=0) / 2**30
    forward = search_data.get('teacher_labeling', {}).get('forward_calls', teacher.get('actual_probes', '未记录'))
    backward = search_data.get('teacher_labeling', {}).get('backward_calls', '未记录')
    rows.append(f"| 教师标签 | {teacher.get('actual_probes', supervision.get('forward_calls', '未记录'))} probes（forward={forward}，backward={backward}） | {_fmt_seconds(teacher.get('elapsed_seconds', search_data.get('teacher_labeling', {}).get('elapsed_seconds')))} | {alloc:.2f} / {reserve:.2f} |")
    totals = search_data.get("completed_search_totals_by_timing_scope", {})
    rows.append(f"| 搜索阶段合计（已完成） | {totals.get('optimizer_steps', '未记录')} updates / {totals.get('demonstration_window_draws', '未记录')} draws | {_fmt_seconds(totals.get('wall_seconds'))} | 未记录（搜索汇总） |")
    candidates = ["\n\n**搜索候选明细**\n", "| 阶段 | 角色 | 学习率 | OPD 权重 | updates | wall / h | development / h | 是否入选 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    candidate_data = search_data.get("candidates", {})
    _require(isinstance(candidate_data, dict) and candidate_data, "search cost candidates are missing")
    for name, item in candidate_data.items():
        candidates.append(f"| `{name}` | {item.get('role', '未记录')} | {item.get('learning_rate', '未记录')} | {item.get('opd_weight', '未记录')} | "
                         f"{item.get('optimizer_steps', '未记录')} | {float(item.get('wall_seconds', 0))/3600:.3f} | "
                         f"{float(item.get('development_wall_seconds', 0))/3600:.3f} | {'是' if item.get('selected') else '否'} |")
    return "\n".join(rows + candidates)


def render_all(root: Path, final: dict[str, Any], protocol: dict[str, Any], inventory: dict[str, Any]) -> dict[str, str]:
    protocol_sha = final.get("source", {}).get("protocol", {}).get("sha256")
    _require(isinstance(protocol_sha, str) and len(protocol_sha) == 64, "missing final protocol identity")
    for rel in ("training/costs.json", "search_costs/summary.json"):
        _require(_read(root / "paper" / "evidence" / rel).get("protocol_sha256") == protocol_sha,
                 f"{rel} does not belong to the final protocol")
    for _, key in ARMS:
        _require(_read(root / "paper" / "evidence" / f"heldout_{key}" / "eval_manifest.json").get("protocol_sha256") == protocol_sha,
                 f"heldout_{key} does not belong to the final protocol")
    quant, lora = quant_lora_tables(root, protocol, inventory)
    return {
        "heldout": heldout_table(root, final),
        "paired": paired_table(root),
        "uncertainty": uncertainty_block(root, final),
        "compression": compression_table(inventory),
        "quantization": quant,
        "lora": lora,
        "training_detail": training_detail(root, protocol, final),
        "collection_search": collection_search_costs(root),
    }
