#!/usr/bin/env python3
"""Rewrite the public prose from the verified v12 held-out evidence.

The manuscript keeps equations, figures and source-reading appendices in the
repository, but all headline numbers are generated from ``final_results.json``
and ``paired_comparison.json``.  This prevents a completed rerun from leaving
the article with stale v11 values.
"""
from __future__ import annotations

import json
from pathlib import Path

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
    rows = ["| 配置 | 闭环成功率 | 相对 BF16 | 量化覆盖 |",
            "|---|---:|---:|---:|"]
    bf = arm_row(final, "bf16")["success_rate"]
    for name in ARMS:
        row = arm_row(final, name)
        coverage = "0%" if name == "bf16" else "100% NVFP4（479 个 eligible 张量）"
        rows.append(f"| {LABELS[name]} | {row['successes']}/{row['episodes']}（{pct(row['success_rate'])}） | {(row['success_rate'] - bf) * 100:+.2f} pp | {coverage} |")
    return "\n".join(rows)


def paired_table(final: dict) -> str:
    c = final["uncertainty"]["contrasts"]
    order = (("ptq_vs_bf16", "PTQ − BF16"), ("qad_vs_ptq", "QAD − PTQ"),
             ("opd_vs_qad", "OPD − QAD"), ("opd_vs_continued_qad", "OPD − continued-QAD"))
    rows = ["| 对比 | 差值 / pp | 95% 配对区间 / pp | McNemar p | Holm p |",
            "|---|---:|---:|---:|---:|"]
    for key, label in order:
        item = c[key]
        lo, hi = item["pointwise_ci_pp"]
        rows.append(f"| {label} | {item['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {item['exact_mcnemar_two_sided_p']:.6g} | {item['holm_adjusted_p']:.6g} |")
    return "\n".join(rows)


def rewrite_main(final: dict) -> None:
    bf = arm_row(final, "bf16"); ptq = arm_row(final, "ptq"); qad = arm_row(final, "qad")
    cont = arm_row(final, "continued_qad"); opd = arm_row(final, "qad_opd")
    inventory = load("recipe_inventory.json")
    budget = inventory["recipes"][inventory["recipe"]]
    residual_bytes = inventory["recovery_residual"]["target_bytes"]
    packed_bytes = budget["target_full_checkpoint_bytes"]
    source_bytes = budget["source_tensor_bytes"]
    net_bytes = packed_bytes + residual_bytes
    compression = f"{source_bytes / net_bytes:.3f}×"
    intro = f'''# 摘要与引言

视觉—语言—动作模型（Vision–Language–Action model，VLA）把相机观测、语言指令和机器人状态映射为动作。对这类模型，低位宽量化的目标不仅是减少权重占用，还要保持动作反馈到环境之后的任务成功率。本文围绕 **APXInf 推理生态、GR00T N1.7、NVFP4、W4A4 训练后量化（PTQ），以及量化基座上的 QAD/OPD 恢复**，给出一条从数值表示、校准、恢复训练到 LIBERO 闭环评测的可复现链条。

本文把 479 个可量化权重张量全部压到 NVFP4，并让 469 个普通 Linear 与 7 个 CategorySpecificLinear 同时采用 NVFP4 激活 QDQ，形成严格的 W4A4 压力基座。PTQ 不更新模型；QAD 在冻结基座上用成功演示训练低秩修正；OPD 再用 QAD 学生真正访问到的状态，请 BF16 教师提供动作生成速度监督。三个阶段回答三个不同问题：低位宽压力是否真实、演示是否能恢复行为、同等追加演示预算之外教师监督是否有独立价值。

最终五臂评测使用十个 LIBERO-10 任务、每臂 160 个配对回合。结果由冻结的 v12 协议和逐回合日志直接计算：

{headline_table(final)}

PTQ 相对 BF16 的变化为 **{(ptq['success_rate'] - bf['success_rate']) * 100:+.2f} 个百分点**；QAD 相对 PTQ 为 **{(qad['success_rate'] - ptq['success_rate']) * 100:+.2f} 个百分点**；OPD 相对同预算 continued-QAD 为 **{(opd['success_rate'] - cont['success_rate']) * 100:+.2f} 个百分点**。APXInf 的原生 NVFP4/FP8 算子延迟另作工程基准，不能替代 GR00T Torch W4A4 闭环结果。

**关键词：** VLA；GR00T；APXInf；NVFP4；W4A4；PTQ；QAD；OPD；LIBERO。

## 1.1 为什么必须在闭环中评估量化

策略每次生成一段动作，机器人只执行其中一部分，再把新的图像和状态送回模型；这种反复决策称为**闭环**。一次从环境重置到成功或超时的完整尝试称为 **episode（回合）**。量化造成的微小动作误差会改变下一次观测，误差可能在后续反馈中累积。因此静态层误差只能用于筛选，任务是否完成必须由环境判定。

## 1.2 从 PTQ 到 QAD，再到 OPD

PTQ 使用已有权重和校准输入确定格式、尺度和舍入结果，不进行恢复训练。本轮为了制造可恢复的 W4A4 压力，冻结的主结果使用全覆盖 NVFP4 RTN 配方；代码同时保留面向块缩放的 GPTQ 接口，但最终结果只采用协议声明的 RTN 基座。

**QAD**（项目中的量化后演示适配）冻结 PTQ 权重，只更新低秩 A/B 参数，使量化模型重新拟合成功演示的动作速度场。**OPD**（学生访问状态上的教师监督）先让 QAD 学生执行闭环，再在这些观测上缓存 BF16 教师速度标签。教师标签固定、学生输入和噪声固定，追加训练时同时使用演示损失和探针损失。

为隔离“多训练一段时间”与教师监督，设置 **continued-QAD**：从同一个 QAD 检查点出发，追加相同次数的演示更新，但不使用学生状态教师标签。OPD−continued-QAD 是教师监督的主要对照；OPD−QAD 还包含追加训练本身。

## 1.3 本文贡献

1. **完整 W4A4 压力。** 479 个 eligible 权重张量全部使用 NVFP4，476 个线性算子同时量化激活，明确记录 ordinary/category 层的覆盖和 padding。
2. **量化域恢复。** QAD 保持 W4A4 基座不变，只训练 468 个普通动作前向 Linear 的 BF16 LoRA 残差；OPD 在学生真实访问分布上提供教师速度监督。
3. **配对闭环证据。** BF16、PTQ、QAD、continued-QAD 和 QAD+OPD 使用同一任务顺序、官方初态和随机种子，逐回合保存 reset 哈希、服务日志和成功标记。
4. **工程边界清楚。** GR00T 行为结果来自 Torch W4A4 QDQ；APXInf 原生 kernel 与图执行作为独立延迟基准，二者不混写。

## 1.4 读者需要先知道的三个口径

W4A4 只描述线性主分支的权重和激活；3 个 embedding/position 张量只做权重量化。QAD/OPD 残差读取原始 BF16 输入，因此净压缩比必须把 LoRA 参数单独计入。最后，本文的成功率是闭环环境指标，不是动作 MSE，也不是原生 kernel 延迟。
'''
    (SECTIONS / "01-摘要与引言.md").write_text(intro, encoding="utf-8")

    experiment = f'''# 4. 实验

## 4.1 设置与数据划分

实验使用 GR00T N1.7 LIBERO-10，在 WSL2 Ubuntu、RTX 5090 Laptop 24 GB 上运行。开发集使用官方初态索引 4–8（50 回合）选择恢复超参数；教师监督和学生采集使用训练索引 20–23；最终 held-out 使用索引 9–19 与 24–28，共 10 个任务×16 个回合。所有任务先恢复官方初态，再执行 10 个零动作稳定步；策略每次产生 16 步动作，环境执行前 8 步，单回合最多 720 步。

QAD 教师监督来自十个任务的 BF16 成功轨迹。每个窗口保留图像、语言、状态、动作端点和有效 mask；OPD 的 160 个学生访问窗口由选定 QAD 学生采集，再由冻结教师在相同噪声和时间步上标注。开发集只用于选择学习率与 OPD 权重，held-out 结果不回流调参。

{{fig:recovery_protocol}}

## 4.2 五臂对照

QAD 与两个追加分支均使用 rank=32、alpha=64 的 LoRA。QAD 训练 2,000 次更新；continued-QAD 与 OPD 从同一个 QAD 检查点出发，各追加 2,000 次演示更新。OPD 还加入学生访问状态上的教师速度损失。正式量化路径统一使用 W4A4 激活 QDQ，并在 manifest 中记录 120,000 ms 的 ZeroMQ 单次请求超时，避免首次量化请求被默认 15 s 客户端超时截断。

| 配置 | 作用 | LIBERO-10 |
|---|---|---:|
| BF16 | 未量化参考 | {bf['successes']}/{bf['episodes']}（{pct(bf['success_rate'])}） |
| W4A4 PTQ | 全覆盖 NVFP4 压力基座 | {ptq['successes']}/{ptq['episodes']}（{pct(ptq['success_rate'])}） |
| PTQ + QAD | 成功演示恢复 | {qad['successes']}/{qad['episodes']}（{pct(qad['success_rate'])}） |
| PTQ + continued-QAD | 同预算演示续训对照 | {cont['successes']}/{cont['episodes']}（{pct(cont['success_rate'])}） |
| PTQ + QAD + OPD | 演示损失 + 学生状态教师速度监督 | {opd['successes']}/{opd['episodes']}（{pct(opd['success_rate'])}） |

{task_table(final)}

## 4.3 配对结果与解释

同一初态上的两个策略可能同时成功或同时失败，统计因此保留 episode 配对关系。按预先固定的分层 bootstrap 和精确 McNemar 检验，结果如下：

{paired_table(final)}

只有 OPD−continued-QAD 直接回答教师监督是否超过同预算演示续训；OPD−QAD 还包含额外 2,000 次更新，不能单独解释为教师项的纯增益。区间和检验只描述这十个任务、一个训练种子的条件证据，不外推到未见任务。

{{fig:ladder}}

## 4.4 编码预算与执行边界

479 个 eligible 张量全部为 NVFP4，普通与类别线性算子为 W4A4；压缩比按去别名 BF16 权重分母、NVFP4 数据、E4M3 块尺度、FP32 二级尺度、未量化张量和 BF16 LoRA 旁路共同计算。本轮 full-checkpoint 主分支为 **{packed_bytes:,} B**，LoRA 旁路为 **{residual_bytes:,} B**，净压缩比为 **{compression}**。完整字节账目由 `paper/evidence/recipe_inventory.json` 生成，不能用稠密 checkpoint 文件大小替代。APXInf 的 NVFP4/FP8 kernel、图重放和 π0.5 计时单独报告；GR00T W4A4 闭环当前是 Torch QDQ 数值参考路径。

{{fig:budget_ladder}}

{{fig:ptq_frontier}}

## 4.5 训练成本与复现

运行目录中的每个阶段都有 `recovery_manifest.json`、训练日志、checkpoint identity 和协议 SHA。继续 QAD 与 OPD 使用相同演示读取数和优化器更新数；OPD 另有学生探针前向/反传，因此耗时不等同。完整命令、环境安装、五臂评测、截图和逐文件哈希见根目录 `README.md` 及附录 B。
'''
    (SECTIONS / "04-实验.md").write_text(experiment, encoding="utf-8")

    method = '''# 3. 方法

本文把低位宽主分支与高精度恢复旁路分开定义。这样读者可以先复现 W4A4 PTQ，再复现 QAD，最后复现 OPD，而不会把 LoRA 残差误看成四位主分支的一部分。

## 3.1 NVFP4 W4A4 PTQ

NVFP4 的数据为 E2M1 四位格点；连续 16 个输入元素共享一个 E4M3 块尺度，张量再共享 FP32 二级尺度。对块 $w_b$，先计算

$$s_b=R_{\mathrm{E4M3}}(\max_j|w_{b,j}|/6),\qquad \hat w_{b,j}=R_{\mathrm{E2M1}}(w_{b,j}/s_b)s_b.$$ 

二级尺度固定为 1；零块只在除法中使用安全除数，反量化仍严格为零。激活沿最后一维按同样的 16 元素规则量化。线性层因此同时满足 W4 与 A4，服务日志会报告 469/469 ordinary 和 7/7 category 的 W4A4 安装结果。

v12 的主结果使用 `rtn_w4a4_category`：不使用校准数据，把 472 个 ordinary recipe 张量和 7 个 category 张量全部写成 NVFP4。3 个 embedding/position 张量只量化权重。每个张量的形状、padding、尺度和实际格式写入 `ptq_recipe.json`、`category_ptq_recipe.json` 与 bake manifest；这些文件是最终配方的唯一来源。

{{fig:gptq_block}}

## 3.2 QAD：冻结基座上的低秩恢复

对一个 $K\rightarrow N$ 线性层，冻结量化权重 $W_q$，增加 $A\in\mathbb R^{r\times K}$ 与 $B\in\mathbb R^{N\times r}$：

$$y=\operatorname{Linear}(Q_A(x),W_q)+\frac{\alpha}{r}\operatorname{Linear}(\operatorname{Linear}(x,A),B).$$

主分支接收 NVFP4 激活；残差分支读取量化前的 BF16 输入。v12 在 468 个 ordinary Linear 上训练 rank=32、alpha=64 的 A/B，量化权重和类别层保持冻结。训练样本来自 BF16 成功演示，动作损失只在有效动作 mask 上计算；QDQ 反向使用 STE。

## 3.3 OPD：学生访问状态上的教师速度监督

先固定 QAD 学生并运行训练分区，保存其观测、动作端点、噪声、时间步和有效 mask。BF16 教师在完全相同的输入上计算速度标签 $v^T$，缓存后不再刷新。OPD 的探针损失为

$$\ell_{\mathrm{probe}}=\frac{\sum m\odot(v_\theta-v^T)^2}{\sum m+10^{-6}}.$$

训练目标是演示损失与按协议权重加入的探针损失之和。continued-QAD 使用同一 QAD 起点、同一演示读取数和同一更新数，但不读取教师缓存；因此 OPD−continued-QAD 是教师监督的独立对照。

## 3.4 部署与闭环评测

部署保留冻结 W4A4 基座和独立 A/B adapter。把 $BA$ 预先合入权重会让残差也接收量化输入，改变训练时的函数，因此正式服务不使用这种合并。每个评测臂使用相同的任务、初态、seed、稳定步和动作步预算；`eval_manifest.json` 记录数值开关、ZeroMQ 超时、checkpoint identity 与 reset 哈希，`compare_recovery.py` 再计算五臂配对统计。
'''
    (SECTIONS / "03-方法.md").write_text(method, encoding="utf-8")

    conclusion = f'''# 5. 讨论与结论

## 5.1 结果应怎样读

PTQ−BF16 衡量全覆盖 W4A4 压力，QAD−PTQ 衡量成功演示能否恢复行为，OPD−continued-QAD 衡量在相同追加演示预算之外教师速度监督的独立作用。本文不把 APXInf 的原生 kernel 延迟解释成 GR00T 闭环速度，也不把 OPD−QAD 单独解释为纯教师增益，因为它包含额外训练阶段。

## 5.2 复现边界

实验固定 LIBERO-10 十个任务和一个训练种子；未见任务、多种子和实机验证需要另行开展。GR00T 闭环是 Torch W4A4 QDQ 参考路径，APXInf 原生执行是独立基准。QAD/OPD 只覆盖 ordinary Linear，7 个 category 层保持 W4A4 冻结；这保证了 W4A4 压力没有被恢复旁路偷偷改成 BF16。

## 5.3 结论

在冻结的 v12 协议下，BF16 为 **{bf['successes']}/{bf['episodes']}（{pct(bf['success_rate'])}）**，全 NVFP4 W4A4 PTQ 为 **{ptq['successes']}/{ptq['episodes']}（{pct(ptq['success_rate'])}）**，QAD 为 **{qad['successes']}/{qad['episodes']}（{pct(qad['success_rate'])}）**，continued-QAD 为 **{cont['successes']}/{cont['episodes']}（{pct(cont['success_rate'])}）**，QAD+OPD 为 **{opd['successes']}/{opd['episodes']}（{pct(opd['success_rate'])}）**。最终结论只由这些完整 held-out 日志支持；所有命令、截图、协议、清单和哈希都随发布包提供。
'''
    (SECTIONS / "05-讨论与结论.md").write_text(conclusion, encoding="utf-8")


def rewrite_other_sources() -> None:
    for path in (SECTIONS / "14-附录A-核心源码走读与APXInf框架解析.md",
                 SECTIONS / "15-附录B-复现与证据索引.md",
                 SECTIONS / "15-附录C-执行基准与完整测量.md"):
        text = path.read_text(encoding="utf-8")
        text = text.replace("recovery_protocol_v11_w4a4_category.json", "recovery_protocol_v12_rtn_w4a4.json")
        text = text.replace("w4a4-recovery-v11-category", "w4a4-recovery-v12-rtn")
        text = text.replace("v11", "v12").replace("V11", "V12")
        text = text.replace("ptqad_v12", "rtn_w4a4_release_20261006_01")
        text = text.replace("20261003", "20261006")
        text = text.replace("all_nvfp4_gptq_category", "rtn_w4a4_category")
        text = text.replace("普通 Linear 使用校准后的 GPTQ 块补偿", "普通 Linear 使用无校准 RTN 舍入")
        text = text.replace("校准候选是 `rtn_w4a4_category`", "压力候选是 `rtn_w4a4_category`")
        path.write_text(text, encoding="utf-8")
    figures_path = PAPER / "figures.json"
    figures = json.loads(figures_path.read_text(encoding="utf-8"))
    for info in figures.values():
        for key in ("title", "note"):
            if isinstance(info.get(key), str):
                info[key] = info[key].replace("v11", "v12").replace("唯一冻结配方", "唯一冻结配方")
    figures_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    final = load("final_results.json")
    if final.get("status") != "complete":
        raise SystemExit("final_results.json is incomplete")
    rewrite_main(final)
    rewrite_other_sources()


if __name__ == "__main__":
    main()
