#!/usr/bin/env python3
"""Fill the polished v12 manuscript templates from verified final evidence.

Main counts, paired statistics and supplementary archives must verify before
the entry point writes public prose. Rendering itself never runs experiments.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from v12_publication_contract import validate_v12_results

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SECTIONS = PAPER / "sections"
ARMS = ("bf16", "ptq", "qad", "continued_qad", "qad_opd")
LABELS = {
    "bf16": "BF16 基线",
    "ptq": "全 NVFP4 W4A4 PTQ",
    "qad": "PTQ + QAD",
    "continued_qad": "PTQ + continued-QAD",
    "qad_opd": "PTQ + QAD + OPD",
}


def load(name: str):
    return json.loads((PAPER / "evidence" / name).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def arm_row(final: dict, name: str) -> dict:
    if name in final["public_arms"]:
        return final["public_arms"][name]
    return final["control"][name]


def task_table(final: dict) -> str:
    tasks = list(arm_row(final, "bf16")["per_task"])
    rows = ["| 任务 | BF16 | W4A4 PTQ | QAD | continued-QAD | QAD+OPD |",
            "|---|---:|---:|---:|---:|---:|"]
    for task in tasks:
        values = []
        for name in ARMS:
            row = arm_row(final, name)["per_task"][task]
            values.append(f"{row['successes']}/{row['episodes']}")
        rows.append("| `" + task + "` | " + " | ".join(values) + " |")
    return "\n".join(rows)


def headline_table(final: dict) -> str:
    rows = ["| 配置 | 闭环成功率 | 相对 BF16 | 可量化权重的 NVFP4 覆盖率 |",
            "|---|---:|---:|---:|"]
    bf = arm_row(final, "bf16")["success_rate"]
    for name in ARMS:
        row = arm_row(final, name)
        coverage = "0/479（0%）" if name == "bf16" else "479/479（100%）"
        rows.append(f"| {LABELS[name]} | {row['successes']}/{row['episodes']}（{pct(row['success_rate'])}） | {(row['success_rate'] - bf) * 100:+.2f} pp | {coverage} |")
    return "\n".join(rows)


def paired_table(final: dict) -> str:
    c = final["uncertainty"]["contrasts"]
    rows = ["| 对比 | 差值 / pp | 95% 配对区间 / pp | McNemar p | Holm p |",
            "|---|---:|---:|---:|---:|"]
    for key, label in CONTRASTS:
        item = c[key]
        lo, hi = item["pointwise_ci_pp"]
        rows.append(f"| {label} | {item['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {item['exact_mcnemar_two_sided_p']:.6g} | {item['holm_adjusted_p']:.6g} |")
    return "\n".join(rows)


CONTRASTS = (
    ("ptq_vs_bf16", "PTQ − BF16"),
    ("qad_vs_ptq", "QAD − PTQ"),
    ("opd_vs_qad", "OPD − QAD"),
    ("opd_vs_continued_qad", "OPD − continued-QAD"),
)
TEMPLATES = PAPER / "templates" / "v12"
TEMPLATE_NAMES = ("01-摘要与引言.md", "03-方法.md", "04-实验.md", "05-讨论与结论.md")
TOKEN = re.compile(r"@@([a-z_]+)@@")


def contrast_assessment(item: dict) -> str:
    """Interpret existing conditional statistics; never recompute or select them."""
    difference = item["difference_pp"]
    lo, hi = item["pointwise_ci_pp"]
    p = item["holm_adjusted_p"]
    if lo <= 0 <= hi:
        return "95% 配对区间包含零，尚不能确认差异方向，也不证明两者等价"
    if p >= .05:
        return "Holm 校正后未达到 0.05 水平，尚不能确认成功率改善或下降"
    if difference > 0 and lo > 0:
        return "配对区间与校正检验支持本设置下的成功率提高"
    if difference < 0 and hi < 0:
        return "配对区间与校正检验支持本设置下的成功率下降"
    raise ValueError("Contrast estimate and interval directions disagree")


def paired_interpretation(final: dict) -> str:
    contrasts = final["uncertainty"]["contrasts"]
    groups: dict[str, list[str]] = {}
    for key, label in CONTRASTS:
        groups.setdefault(contrast_assessment(contrasts[key]), []).append(label)
    lines = ["、".join(labels) + "：" + assessment + "。" for assessment, labels in groups.items()]
    return ("".join(lines) + "\n\n这些推断以任务内回合可独立重采样及 McNemar 的配对可交换性为前提，"
            "仅适用于这十个固定任务与一个训练种子。区间逐项计算；Holm 校正只覆盖上述四项比较。")


def five_arm_table(final: dict) -> str:
    purposes = ("未量化参考", "全覆盖量化的影响", "成功演示恢复", "追加演示更新的作用", "追加教师监督的作用")
    rows = ["| 配置 | 检验内容 | 正式成功率 |", "|---|---|---:|"]
    for name, purpose in zip(ARMS, purposes):
        row = arm_row(final, name)
        rows.append(f"| {LABELS[name]} | {purpose} | {row['successes']}/{row['episodes']}（{pct(row['success_rate'])}） |")
    return "\n".join(rows)


def result_conclusion(final: dict, compression: str) -> str:
    contrasts = final["uncertainty"]["contrasts"]
    descriptions = (
        ("ptq_vs_bf16", "量化影响"),
        ("qad_vs_ptq", "QAD 恢复"),
        ("opd_vs_qad", "OPD 完整追加阶段相对 QAD"),
        ("opd_vs_continued_qad", "OPD 相对同演示预算续训"),
    )
    lines = [f"{label}的配对差值为 {contrasts[key]['difference_pp']:+.2f} 个百分点，"
             + contrast_assessment(contrasts[key]) + "。" for key, label in descriptions]
    return ("".join(lines) + f"含 BF16 低秩旁路的净编码压缩比为 **{compression}**。"
            "校准 GPTQ 参考和完整动作诊断分别见 §4.6 与 §4.5；它们限定比较范围，"
            "不构成优于所有 PTQ 方法或实现原生 GR00T 加速的证据。")


def validate_inventory_name(final: dict, inventory: dict) -> None:
    # The protocol names the experiment candidate; the inventory builder names
    # its encoding recipe. These are intentionally different namespaces.
    if (final["selected_recipe"] != "rtn_w4a4_category" or inventory["recipe"] != "rtn_all"
            or inventory.get("schema_version") != "w4a4-selected-all-nvfp4-category-v12"):
        raise ValueError("Encoding inventory is not the selected final recipe")


def load_verified_inventory(final: dict) -> dict:
    from build_final_frontier import load_inventory
    inventory, _, _, _ = load_inventory(PAPER / "evidence/recipe_inventory.json")
    validate_inventory_name(final, inventory)
    return inventory


def render_main(final: dict, diagnostic_text: str, gptq_text: str, inventory: dict) -> dict[str, str]:
    """Fill the polished manuscript in memory, after the caller verifies evidence."""
    validate_inventory_name(final, inventory)
    budget = inventory["recipes"][inventory["recipe"]]["known_tied_alias_deduplicated"]
    source_bytes = budget["source_tensor_bytes"]
    packed_bytes = budget["target_full_bytes"]
    residual_bytes = inventory["recovery_residual"]["target_bytes"]
    if any(type(value) is not int or value <= 0 for value in (source_bytes, packed_bytes, residual_bytes)):
        raise ValueError("Encoding budget requires positive source and target tensor byte counts")
    compression = f"{source_bytes / (packed_bytes + residual_bytes):.3f}×"
    budget_table = ("| 项目 | 完整目标编码 |\n|---|---:|\n"
                    f"| BF16 源参数 | {source_bytes:,} B |\n"
                    f"| NVFP4 PTQ 主分支与未量化参数 | {packed_bytes:,} B |\n"
                    f"| BF16 低秩旁路 | {residual_bytes:,} B |\n"
                    f"| 恢复模型净压缩比 | {compression} |")
    values = {"headline_table": headline_table(final), "five_arm_table": five_arm_table(final),
              "task_table": task_table(final), "paired_table": paired_table(final),
              "paired_interpretation": paired_interpretation(final), "budget_table": budget_table,
              "net_compression": compression,
              "opd_delta_pp": f"{final['uncertainty']['contrasts']['opd_vs_continued_qad']['difference_pp']:+.2f}",
              "action_diagnostics": diagnostic_text, "gptq_reference": gptq_text,
              "conclusion": result_conclusion(final, compression)}
    rendered = {}
    for name in TEMPLATE_NAMES:
        template = (TEMPLATES / name).read_text(encoding="utf-8")
        document = TOKEN.sub(lambda match: values[match[1]], template)
        if re.search(r"@@|xxx|审阅说明|待回填|尚未全部完成|正式评测尚未完成", document):
            raise ValueError(f"Unresolved release text in {name}")
        rendered[name] = document
    return rendered


def rewrite_main(documents: dict[str, str]) -> None:
    for name, document in documents.items():
        (SECTIONS / name).write_text(document, encoding="utf-8")


def rewrite_other_sources() -> None:
    """Synchronize the source-reading appendices with the v12 public contract.

    These files predate the RTN W4A4 rerun and contain useful implementation
    notes, but they also used to describe the discarded GPTQ parent and v11
    run directory.  Blind string replacement is not sufficient here: a
    paragraph can be syntactically current while still presenting an old
    recipe as a result.  The replacements below therefore remove the old
    GPTQ result subsection and replace it with the actual v12 RTN path before
    applying the common run/protocol renames.
    """

    appendix_paths = (
        SECTIONS / "14-附录A-核心源码走读与APXInf框架解析.md",
        SECTIONS / "15-附录B-复现与证据索引.md",
        SECTIONS / "15-附录C-执行基准与完整测量.md",
    )

    def replace_between(text: str, start: str, end: str, replacement: str) -> str:
        """Replace one complete Markdown section, retaining its end heading."""
        begin = text.find(start)
        if begin < 0:
            return text
        finish = text.find(end, begin + len(start))
        if finish < 0:
            return text
        return text[:begin] + replacement.rstrip() + "\n\n" + text[finish:]

    def replace_paragraph_starting(text: str, prefix: str, replacement: str) -> str:
        """Replace one prose paragraph without depending on TeX backslashes."""
        pattern = re.compile(rf"(?m)^{re.escape(prefix)}[^\n]*(?:\n\n|\Z)")
        return pattern.sub(lambda _: replacement.rstrip() + "\n\n", text, count=1)

    for path in appendix_paths:
        text = path.read_text(encoding="utf-8")

        # A.3.2 in the old source walk described GPTQ as the selected parent.
        # v12 deliberately uses calibration-free RTN, so keep the numerical
        # definitions useful to readers while removing the stale result claim.
        if path.name.startswith("14-"):
            text = text.replace("首先定义量化数值，再用真实输入校准，随后在量化基座上训练低秩残差", "首先定义量化数值，按固定 RTN 规则生成基座，随后训练低秩残差")
            text = text.replace("源 checkpoint → 校准统计 → 全 NVFP4 PTQ 基座", "源 checkpoint → 全 NVFP4 RTN PTQ 基座")
            text = text.replace("## A.3 用真实输入选择量化参数并生成 PTQ 基座", "## A.3 从固定格式生成 RTN PTQ 基座")
            text = replace_paragraph_starting(text, "单看权重误差会忽略输入通道的使用频率。", "这一阶段按协议指定的 RTN 舍入规则把源权重写成可加载的完整 checkpoint。配方在评测前冻结，不依赖校准数据；输入统计和尺度搜索接口是单独的研究工具。")
            # The collector/Hessian chapter belongs to optional diagnostics;
            # make that boundary explicit so the public appendix cannot be
            # read as if v12 selected a calibrated parent.
            text = replace_between(
                text,
                "### A.3.1",
                "### A.3.2",
                r'''### A.3.1 采集器接口与 v12 主路径

`quant/ptq/collector.py` 仍提供完整模型的输入统计接口，便于源码测试和后续实验定位真实执行的 Linear 层。它把输入展平为形状为 `[R,K]` 的矩阵，检查模块调用次数、输入行数和 checkpoint 身份，并把统计摘要写入 manifest。这个接口解释了工程如何确认模型结构，但 v12 的正式 `rtn_w4a4_category` 配方使用 `calibration-mode=none`，不会把 Hessian、裁剪搜索或校准样本用于 PTQ 决策。

因此 v12 的可复现依赖只有源 checkpoint、固定格式合同和 bake manifest：读者不需要重新采集统计，也不能用 held-out 成功率反向选择量化参数。若运行者调用 collector 进行诊断，产物必须与 RTN bake 分开登记，不能写入五臂主结果。共享词嵌入仍按 tied alias 规则只保留一个规范来源；非 Linear 目标和未执行 bank 也会在清单中显式标注。

''',
            )
            rtn_section = r'''### A.3.2 v12 的无校准 RTN 舍入

v12 的正式压力基座使用 `calibration-mode=none` 的 RTN（round-to-nearest）路径。它不读取 Hessian、激活统计或 held-out 成功率，因而量化结果可以从源 checkpoint、固定的 NVFP4 格式和协议直接重建。对每个二维权重张量，先按全张量最大幅值确定一次 FP32 二级尺度，再按连续 16 个输入元素计算 E4M3 块尺度，最后将块内值舍入到 E2M1 格点。激活沿同一输入轴执行 NVFP4 QDQ，形成真正的 W4A4 数值压力。

```python
tau = tensor_scale(original_weight)       # one frozen tensor scale
for block in blocks_of_16(original_weight):
    scale = round_e4m3(max_abs(block) / 6 / tau)
    divisor = where(scale > 0, scale * tau, 1)
    code = round_e2m1(block / divisor)
    dequant_block = code * scale * tau
```

零块保留零块尺度；除法只使用安全分母来选择编码，不能把零块改成非零值。`quant/ptq/quantizers.py::nvfp4_dequant` 按整个权重矩阵固定二级尺度；`quant/native_activation.py::native_activation_qdq_torch` 将激活先转为 F16，激活二级尺度固定为 1。二者沿输入维度使用相同的 16 元素块边界，但不能把两种二级尺度混为一谈。服务日志记录请求格式、实际格式、padding 和未量化张量。
'''
            text = replace_between(
                text,
                "### A.3.2",
                "### A.3.3 AWQ 搜索的有效误差与可折叠条件",
                rtn_section,
            )
            text = replace_between(text, "### A.3.2", "### A.3.3 可选尺度搜索接口", rtn_section)
            text = replace_between(
                text,
                "### A.3.3 AWQ 搜索的有效误差与可折叠条件",
                "### A.3.4 精度配方、共享别名与完整索引",
                r'''### A.3.3 可选尺度搜索接口

源码还包含 AWQ 风格的输入尺度搜索与折叠规则，用于独立的算子研究。它要求逐个核对归一化、门控非线性和 GQA 共享通道，不能把局部代理误差直接当作闭环结论。v12 的正式压力基座不启用该搜索；所有层均按 A.3.2 的固定 RTN 规则写盘，避免读者把备用研究入口误认为第二套发布配方。

''',
            )
            text = replace_between(
                text,
                "`bake.py::alloc` 是显式模块规则",
                "`category_fp4.py` 再沿真实输入轴处理 7 个类别权重。",
                """`bake.py::alloc` 是显式模块规则，不根据 held-out 成功率事后修改层范围。v12 的唯一正式配方是 `rtn_w4a4_category`：472 个 ordinary recipe 张量与 7 个 CategorySpecificLinear 张量全部写入 NVFP4；运行时安装报告覆盖 469 个 ordinary Linear 与 7 个 category bank 的 W4A4 activation QDQ。请求方法、实际编码、padding、尺度和 tied alias 都写入 `ptq_recipe.json` 与 `category_ptq_recipe.json`，这些 manifest 是最终配方的唯一来源。\n\n""",
            )
            text = replace_between(
                text,
                "实际量化方法由模块规则、校准模式和张量类型共同决定。",
                "共享权重必须先于逐键写盘处理。",
                """v12 的实际编码由模块类型和固定 RTN 规则共同决定：ordinary 与 category 权重均使用 NVFP4，普通与类别 Linear 均安装 W4A4 activation QDQ；3 个 embedding/position 张量只做权重量化。每个二维层和每个类别 bank 的请求格式、实际编码、裁剪值、尺度、padding 与回退原因写入 `ptq_recipe.json` 或 `category_ptq_recipe.json`。这些记录用于验证实现是否兑现协议，不用于按结果改写层范围。\n\n""",
            )
            text = text.replace("v11 的 category manifest", "v12 的 category manifest")
            text = text.replace(
                "v11 固定 `all_ordinary_linear` 范围：468 个普通 Linear 注入 LoRA；",
                "v12 固定 `all_ordinary_linear` 范围：468 个普通 Linear 注入 LoRA；",
            )
            text = replace_paragraph_starting(
                text,
                "本轮教师监督缓存包含 148 个演示窗口",
                "演示窗口数与批量配置由 `recovery_manifest.json` 记录，文件顺序与尾批规则由数据集和加载器源码决定；上游 DataLoader 顺序读取，不打乱、不丢弃尾批。配置 $B_\\mu=1,G=16$ 时，Trainer 按本次更新实际含有的微批数 $n$ 归一化损失，尾批不能按配置上限补齐。窗口读取预算由 `paper/collect_training_costs.py` 结合样本数、`trainer_state.json` 完成步数/轮数与尾批规则推导，不能仅用“更新数×名义 batch”计算。",
            )
            text = text.replace(
                "顺序读取也意味着同一组 4 个窗口每轮都处于尾批，在该次平均损失中的单样本系数为 $1/4$，其余更新为 $1/16$。这是实际采样与归一化方式；continued-QAD 和 OPD 必须沿用同一数据顺序与尾批规则，才能比较相同演示预算下的附加教师监督。",
                "顺序读取会使末尾微批具有不同的归一化分母；continued-QAD 与 OPD 沿用同一数据顺序、尾批规则和优化器更新数，才能比较相同演示预算下的附加教师监督。",
            )
            text = text.replace(
                "按本轮 148 窗口的读取方式，完整完成 2,000 次更新时共有 500 次探针更新，其中 400 次含 16 个微批、100 次含 4 个微批，合计 6,800 次探针反向。缓存索引仅在实际执行探针时递增。",
                "探针更新与缓存读取预算由 `paper/collect_training_costs.py` 结合 OPD 配置、完成步数和尾批规则推导，并核对训练日志；缓存索引仅在实际执行探针时递增。",
            )
            text = replace_paragraph_starting(
                text,
                "按本轮 148 窗口的读取方式",
                "探针更新与缓存读取预算由 `paper/collect_training_costs.py` 结合 OPD 配置、完成步数和尾批规则推导，并核对训练日志；缓存索引仅在实际执行探针时递增。",
            )
            text = text.replace(
                "GPTQ 固定尺度、可表示性及矩阵目标",
                "RTN 固定尺度与可表示性；可选校准器的矩阵目标",
            )
            text = text.replace("RTN 固定尺度、可表示性及矩阵目标", "RTN 固定尺度与可表示性；可选校准器的矩阵目标")
            text = text.replace("并用真实 Trainer 验证 148 个样本在第 20 次更新遇到探针尾批时可继续执行。", "并用真实 Trainer 验证探针尾批可继续执行。")
            text = text.replace("| 校准 | `collector.py::accumulate`、完整模型入口 |", "| 可选校准诊断（不进入 RTN 主配方） | `collector.py::accumulate`、完整模型入口 |")
            text = text.replace("校准和演示训练可能重复采样窗口", "演示训练可能重复采样窗口")
            text = replace_paragraph_starting(
                text, "以每 4 次优化器更新添加一次为例",
                "v12 的 `opd_every=1`，每次优化器更新都加入教师项。探针更新与缓存读取预算由 `paper/collect_training_costs.py` 结合 OPD 配置、完成步数和尾批规则推导，并核对训练日志；缓存索引仅在实际执行探针时递增。实现保留原 `compute_loss` 与 `return_outputs` 协议，采用单设备 HF Trainer 的梯度累积规则。",
            )
            text = text.replace("尾批的 $n=4$，不能使用配置上限 $G=16$ 代替。", "尾批使用实际微批数 $n$，不能使用配置上限 $G=16$ 代替。")
            text = text.replace("初始 QAD 训练与正式评测的开关差异见 §3.2。", "F16 转换与饱和开关由训练和评测 manifest 分别记录，具体合同见 §3.1。")

        # Protocol/run names and stale result labels are rewritten together so
        # generated appendices never show two competing implementation claims.
        text = text.replace("recovery_protocol_v11_w4a4_category.local.json", "recovery_protocol_v12_rtn_w4a4.json")
        text = text.replace("recovery_protocol_v11_w4a4_category.json", "recovery_protocol_v12_rtn_w4a4.json")
        text = text.replace("w4a4-recovery-v11-category", "w4a4-recovery-v12-rtn")
        text = text.replace("ptqad_v11", "reruns/rtn_w4a4_release_20261006_01/recovery_v12")
        text = text.replace("ptqad_v12", "reruns/rtn_w4a4_release_20261006_01/recovery_v12")
        text = text.replace("v11", "v12").replace("V11", "V12")
        text = text.replace("20261003", "20261006")
        text = text.replace("all_nvfp4_gptq_category", "rtn_w4a4_category")
        text = text.replace("普通 Linear 使用校准后的 GPTQ 块补偿", "普通 Linear 使用无校准 RTN 舍入")
        text = text.replace("校准候选是 `rtn_w4a4_category`", "压力候选是 `rtn_w4a4_category`")

        # B.1 is intentionally concrete: readers can copy it for the released
        # run.  Do not leave a v11 environment variable after the generic pass.
        if path.name.startswith("15-附录B"):
            text = replace_between(text, 'export V12_ROOT=', 'export PROTOCOL_FILE=', r'''export V12_ROOT="$PROJECT/results/reruns/rtn_w4a4_release_20261006_01"
export PRESSURE_ROOT="$PROJECT/results/reruns/rtn_w4a4_pressure_20261006_01"
export DEV_BF16="$V12_ROOT/dev_bf16_v12"
export DEV_PTQ="$V12_ROOT/dev_rtn_v12"
export SELECTION_DIR="$V12_ROOT/selection_final"
export RECOVERY_DIR="$V12_ROOT/recovery_v12"
export PTQAD_RUN_DIR="$RECOVERY_DIR"
export PTQAD_PROTOCOL_FILE="$PROJECT/exp/recovery_protocol_v12_rtn_w4a4.json"
export PTQAD_SELECTION="$SELECTION_DIR/selection.json"
export PTQAD_CAPTURE="$V12_ROOT/teacher_supervision_v12_clean"
export PTQAD_ZMQ_TIMEOUT_MS=120000
''')
            text = replace_between(
                text,
                "## B.3",
                "## B.4 在冻结 W4A4 基座上训练 QAD",
                r'''## B.3 生成全覆盖 RTN W4A4 基座并完成 development 压力筛选

v12 不把校准统计用于 PTQ 决策。运行 `rtn_w4a4_category` bake 入口时，472 个 ordinary recipe 张量与 7 个 CategorySpecificLinear 张量全部编码为 NVFP4；469 个普通 Linear 与 7 个 category bank 同时安装激活 QDQ。bake manifest、category manifest 和激活安装报告共同证明 W4A4 覆盖，不能用单独的文件大小或一次 smoke 成功率替代。

development 使用协议固定的初态 4–8、seed 940000、每任务 5 回合，只用于确认压力基座和恢复超参数。它不参与最终 held-out 成功率。每步实际产物如下；路径以本轮 `$V12_ROOT` 为根：

| 顺序 | 操作 | 必需产物 |
|---|---|---|
| 1 | 按源 checkpoint 清单生成无校准 RTN NVFP4 ordinary 基座 | `$PRESSURE_ROOT/pure_rtn/`，含 `ptq_recipe.json` 与 bake manifest |
| 2 | 为 7 个 CategorySpecificLinear bank 生成 NVFP4 配方并验收 padding/scale | `$PRESSURE_ROOT/rtn_category/`，含 category manifest |
| 3 | 在同一协议上评测 BF16 与 W4A4 PTQ，检查激活安装和配对身份 | `$DEV_BF16/`、`$DEV_PTQ/`、`$PTQAD_SELECTION` |
| 4 | 采集十个任务的 BF16 成功演示，并核验 processor、statistics 与 reset 身份 | `$PTQAD_CAPTURE/`，供 QAD 训练读取 |

只有协议、bake 清单和 development 收据三者身份一致，才进入 QAD；held-out 结果不会反向改变 RTN 配方。

''',
            )
            text = text.replace(
                "本轮148窗口数据的尾批为4，实际读取预算见A.4.2。",
                "`recovery_manifest.json` 提供演示窗口数与批量配置；读取预算由 `paper/collect_training_costs.py` 结合 `trainer_state.json` 的完成步数、轮数和加载器尾批规则推导，归一化规则见附录 A。",
            )
            text = text.replace("OPD 每 4 步加入一次教师项", "OPD 每次优化器更新都加入教师项（`opd_every=1`）")
            text = text.replace("分析规则见 §4.4。", "分析规则见 §4.3。")
            text = text.replace(
                "`final_manifest.json` 必须绑定 v12 protocol SHA",
                "`final_manifest.json` 必须绑定 v12 RTN W4A4 protocol SHA",
            )
            text = text.replace(
                "冻结协议的 `primary_effect_minimum` 数值字段为 5 个百分点，同一对象的文字注记却写为 10 个百分点，两者不一致。原协议保持不变，这两个数值均不用于事后改变模型选择或最终统计；结果解释依据独立固定的 `paper/analysis_plan_w4a4.json`，完整报告实际差值、配对区间与四项预定检验。",
                "评测阈值、配对统计和恢复选择均由 `exp/recovery_protocol_v12_rtn_w4a4.json` 与 `paper/analysis_plan_w4a4.json` 在运行前冻结；最终结果只从五臂完整 held-out 日志计算，不以训练损失或 smoke 分数替代闭环成功率。",
            )
            text = text.replace("`<V12_ROOT>/ptq_parent/run_parameters.json`", "`<PRESSURE_ROOT>/pure_rtn/bake_manifest.json`")
            text = text.replace("| 基座和校准是否一致 | `<RUN>/calibration/calib_meta.json` | 16/32/4 结构、权重/配置/统计散列、实际窗口和每层行数 |", "| 基座和数值格式是否一致 | `paper/evidence/selected_recipe/` 的普通与类别配方及 manifest | 源权重与配置身份、RTN none、NVFP4 覆盖、block/tensor scale、padding |")
            text = text.replace("requested/actual 方法、裁剪、H 覆盖、RTN 回退、tied alias 和未量化张量", "requested/actual 方法、固定 RTN 裁剪、尺度、tied alias 和未量化张量")
            text = text.replace("BF16 成功轨迹形成十任务 148 个窗口；每 2,000 更新读取 29,600 次演示样本。OPD 另有 6,800 次探针前向/反传，教师采集和标注另计。", "十任务 BF16 成功轨迹形成演示窗口；实际样本数、每阶段读取数和 OPD 额外前向/反传由本轮 `paper/evidence/training/costs.json` 完整列出。")
            text = text.replace("15 张为本轮已核验实拍，`shot_gr00t` 与 `shot_pi05` 为经批准保留的未受影响 BF16 图。", "7 张与 v12 量化及恢复相关的图绑定本轮产物，其余图展示独立数值、原生执行或环境环节。")
            text = replace_paragraph_starting(text, "以下保留并展示全部 17 个截图环节", "以下保留并展示全部 17 个截图环节，按复现顺序排列。量化、教师演示采集、QAD、教师缓存、OPD、策略服务与学生采集七个环节使用本轮受控脚本重新实拍；其余环节沿用已核验来源。截图的命令、日志、裁剪链和图像 SHA-256 见 `paper/evidence/captures/`，定量结论来自完整结果日志。")
            text = text.replace("### 校准、量化与数值核对", "### 数据采集、量化与数值核对")
            text = text.replace("先收集层输入统计，再生成量化基座并核对编码格点。packed 图展示原生引擎所需的四位数据与缩放载荷。", "教师采集图展示用于恢复的成功演示，量化图核对无校准 RTN 基座。packed 图展示原生引擎所需的四位数据与缩放载荷，数值图核对编码格点。")

        if path.name.startswith("15-附录C"):
            text = text.replace(
                "不属于 v12 的 GR00T 五臂闭环",
                "不属于本文 GR00T 五臂闭环",
            )

        stale = re.search(r"(?i)v11|all_nvfp4_gptq_category|20261003|43\.4%|99\.0%|86/100|89/100|145/160|90\.62%|待回填|待实拍|审阅稿|xxx%", text)
        if stale:
            raise ValueError(f"Refusing stale appendix content in {path.name}: {stale.group(0)}")
        path.write_text(text, encoding="utf-8")
    figures_path = PAPER / "figures.json"
    figures = json.loads(figures_path.read_text(encoding="utf-8"))
    for info in figures.values():
        for key in ("title", "note"):
            if isinstance(info.get(key), str):
                info[key] = info[key].replace("v11", "v12").replace("唯一冻结配方", "唯一冻结配方")
    if isinstance(figures.get("gptq_block"), dict):
        figures["gptq_block"]["title"] = "图 1　NVFP4 RTN 的权重二级尺度与激活块尺度"
        figures["gptq_block"]["note"] = (
            "展示 v12 无校准 RTN 的权重路径：每个矩阵共享动态 FP32 二级尺度，"
            "每 16 个权重元素共享 E4M3 块尺度；激活路径先转 F16，二级尺度固定为 1。"
            "配方无需校准输入，不绘制未经测量的收益曲线。"
        )
    if isinstance(figures.get("shot_collect"), dict):
        figures["shot_collect"]["title"] = "运行截图 01　BF16 成功演示与教师回放核验"
        figures["shot_collect"]["note"] = (
            "只读核验冻结的 BF16 教师演示归档、任务覆盖和样本哈希；"
            "v12 RTN PTQ 不使用 Hessian 校准，截图不作为量化结果。"
        )
    figures_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    final = load("final_results.json")
    validate_v12_results(final, root=ROOT)
    from action_diagnostics_publication import load_verified, render
    diagnostic_text = render(load_verified(PAPER))
    from gptq_reference_publication import load_verified as load_gptq, render as render_gptq
    gptq_text = render_gptq(load_gptq(PAPER))
    documents = render_main(final, diagnostic_text, gptq_text, load_verified_inventory(final))
    meta_path = PAPER / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = "v12 RTN W4A4 完整五臂评测与发布证据已核验"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rewrite_main(documents)
    rewrite_other_sources()


if __name__ == "__main__":
    main()
