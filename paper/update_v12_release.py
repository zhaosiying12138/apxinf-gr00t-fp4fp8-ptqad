#!/usr/bin/env python3
"""Rewrite the public prose from the verified v12 held-out evidence.

The manuscript keeps equations, figures and source-reading appendices in the
repository, but all headline numbers are generated from ``final_results.json``
and ``paired_comparison.json``.  This prevents a completed rerun from leaving
the article with stale v11 values.
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
    rows = ["| 配置 | 闭环成功率 | 相对 BF16 | eligible 权重的 NVFP4 覆盖率 |",
            "|---|---:|---:|---:|"]
    bf = arm_row(final, "bf16")["success_rate"]
    for name in ARMS:
        row = arm_row(final, name)
        coverage = "0/479（0%）" if name == "bf16" else "479/479（100%）"
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
    deduplicated = budget["known_tied_alias_deduplicated"]
    packed_bytes = deduplicated["target_full_bytes"]
    source_bytes = deduplicated["source_tensor_bytes"]
    net_bytes = packed_bytes + residual_bytes
    compression = f"{source_bytes / net_bytes:.3f}×"
    intro = f'''# 摘要与引言

视觉—语言—动作模型（Vision–Language–Action model，VLA）把相机观测、语言指令和机器人状态映射为动作。对这类模型，低位宽量化的目标不仅是减少权重占用，还要保持动作反馈到环境之后的任务成功率。本文围绕 **APXInf 推理生态、GR00T N1.7、NVFP4、W4A4 训练后量化（PTQ），以及量化基座上的 QAD/OPD 恢复**，给出一条从数值表示、量化、恢复训练到 LIBERO 闭环评测的可复现链条。

本文把 479 个可量化权重张量全部压到 NVFP4，并为 469 个普通线性层（Linear）与 7 个按机器人类别选择权重的线性层（CategorySpecificLinear）安装激活量化。W4A4 表示主分支权重与激活均为四位；本文用量化后再反量化（QDQ）模拟其数值。PTQ 离线量化权重，在每次前向中量化激活，不进行恢复训练；QAD 在冻结基座上用成功演示训练低秩修正；OPD 再用 QAD 学生真正访问到的状态，请 BF16 教师提供动作流的速度场标签。三个阶段回答三个不同问题：低位宽压力是否真实、演示是否能恢复行为、同等追加演示预算之外教师监督是否有独立价值。

最终五臂评测使用十个 LIBERO-10 任务、每臂 160 个配对回合。结果由冻结的 v12 协议和逐回合日志直接计算：

{headline_table(final)}

PTQ 相对 BF16 的变化为 **{(ptq['success_rate'] - bf['success_rate']) * 100:+.2f} 个百分点**；QAD 相对 PTQ 为 **{(qad['success_rate'] - ptq['success_rate']) * 100:+.2f} 个百分点**；OPD 相对同预算 continued-QAD 为 **{(opd['success_rate'] - cont['success_rate']) * 100:+.2f} 个百分点**。APXInf 的原生 NVFP4/FP8 算子延迟另作工程基准，不能替代 GR00T Torch W4A4 闭环结果。

**关键词：** VLA；GR00T；APXInf；NVFP4；W4A4；PTQ；QAD；OPD；LIBERO。

## 1.1 为什么必须在闭环中评估量化

策略每次生成一段动作，机器人只执行其中一部分，再把新的图像和状态送回模型；这种反复决策称为**闭环**。一次从环境重置到成功或超时的完整尝试称为 **episode（回合）**。量化造成的微小动作误差会改变下一次观测，误差可能在后续反馈中累积。因此静态层误差只能用于筛选，任务是否完成必须由环境判定。

## 1.2 从 PTQ 到 QAD，再到 OPD

PTQ（训练后量化）在已有权重上确定格式、尺度和舍入结果，不进行恢复训练；其中 RTN 直接舍入，GPTQ 还使用校准输入估计误差。本文采用全覆盖 NVFP4 RTN 配方作为 W4A4 PTQ 基线；代码同时保留面向块缩放的 GPTQ 接口，但最终结果只采用协议声明的 RTN 基座。

**QAD**（项目中的量化后演示适配）冻结 PTQ 权重，只更新低秩适配（LoRA）的 A/B 参数，使量化模型重新拟合成功演示的动作速度场。**OPD**（学生访问状态上的教师监督）先让 QAD 学生执行闭环，再在这些观测上缓存 BF16 教师速度标签。缓存固定输入、教师标签和随机种子，追加训练时重放相同噪声与时间，同时使用演示损失和探针损失。

为隔离“多训练一段时间”与教师监督，设置 **continued-QAD**：从同一个 QAD 检查点出发，追加相同次数的演示更新，但不使用学生状态教师标签。OPD−continued-QAD 是教师监督的主要对照；OPD−QAD 还包含追加训练本身。

## 1.3 本文贡献

1. **完整 W4A4 压力。** 479 个 eligible 权重张量全部使用 NVFP4，476 个线性算子同时量化激活，明确记录 ordinary/category 层的覆盖和 padding。
2. **量化域恢复。** QAD 保持 W4A4 基座不变，只训练 468 个参与动作前向的普通 Linear 的低秩残差；OPD 在学生真实访问分布上提供教师速度监督。
3. **配对闭环证据。** BF16、PTQ、QAD、continued-QAD 和 QAD+OPD 使用同一任务顺序、官方初态和随机种子，逐回合保存 reset 哈希、服务日志和成功标记。
4. **工程边界清楚。** GR00T 行为结果来自 Torch W4A4 QDQ；APXInf 原生 kernel 与图执行作为独立延迟基准，二者不混写。

## 1.4 读者需要先知道的三个口径

“100%”的分母是配方清单中的 479 个可量化（eligible）权重张量，不包括偏置、归一化参数或新增 LoRA。W4A4 只描述线性主分支；3 个 embedding/position 张量只量化权重。QAD/OPD 残差读取原始 BF16 输入，部署参数按 BF16 计入净压缩比。本文的成功率是闭环环境指标，不是动作均方误差（MSE）或原生 kernel 延迟。
'''
    (SECTIONS / "01-摘要与引言.md").write_text(intro, encoding="utf-8")

    experiment = f'''# 4. 实验

## 4.1 设置与数据划分

实验使用 GR00T N1.7 LIBERO-10，在 WSL2 Ubuntu、RTX 5090 Laptop 24 GB 上运行。开发集使用官方初态索引 4–8（50 回合）选择恢复超参数；教师监督和学生采集使用训练索引 20–23；最终 held-out 使用索引 9–19 与 24–28，共 10 个任务×16 个回合。所有任务先恢复官方初态，再执行 10 个零动作稳定步；策略每次产生 16 步动作，环境执行前 8 步，单回合最多 720 步。

QAD 的演示来自十个任务的 BF16 成功轨迹。每个窗口保留图像、语言、状态、动作端点和有效 mask；OPD 的 160 个学生访问窗口由选定 QAD 学生采集，再由冻结教师按固定随机种子标注速度场。开发集只用于选择学习率与 OPD 权重，held-out 结果不回流调参。

{{{{fig:recovery_protocol}}}}

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

{{{{fig:ladder}}}}

## 4.4 编码预算与执行边界

479/479 表示 eligible 张量覆盖率，不是全部模型参数的元素占比。编码预算先对已知共享别名去重；源 BF16 参数为 **{source_bytes:,} B**，主分支包含 NVFP4 数据、E4M3 块尺度、FP32 二级尺度和未量化参数，共 **{packed_bytes:,} B**；BF16 LoRA 另占 **{residual_bytes:,} B**。净压缩比按“去别名 BF16 源参数字节数 ÷（去别名主分支字节数＋LoRA 字节数）”计算，为 **{compression}**。物理张量、去别名张量及元素占比的完整账目由 `paper/evidence/recipe_inventory.json` 生成；目标编码预算不同于当前稠密 checkpoint 的文件大小。

APXInf 的 NVFP4/FP8 kernel、图重放和 π0.5 计时单独报告；GR00T W4A4 闭环当前是 Torch QDQ 数值参考路径。

{{{{fig:budget_ladder}}}}

{{{{fig:ptq_frontier}}}}

## 4.5 训练成本与复现

运行目录中的每个阶段都有 `recovery_manifest.json`、训练日志、checkpoint identity 和协议 SHA。继续 QAD 与 OPD 使用相同演示读取数和优化器更新数；OPD 另有学生探针前向/反传，因此耗时不等同。完整命令、环境安装、五臂评测、截图和逐文件哈希见根目录 `README.md` 及附录 B。
'''
    (SECTIONS / "04-实验.md").write_text(experiment, encoding="utf-8")

    method = r'''# 3. 方法

本文把低位宽主分支与高精度恢复旁路分开定义。这样读者可以先复现 W4A4 PTQ，再复现 QAD，最后复现 OPD，而不会把 LoRA 残差误看成四位主分支的一部分。

## 3.1 NVFP4 W4A4 PTQ

NVFP4 的数据为 E2M1 四位格点；连续 16 个输入元素共享一个 E4M3 块尺度，张量再共享 FP32 二级尺度。权重量化先从完整矩阵 $W$ 确定唯一二级尺度 $\tau_W$，再处理每个块 $w_b$：

$$
\begin{aligned}
\tau_W&=\operatorname{FP32}\!\left(\max(\max|W|/448,2^{-149})\right),\\
s_b&=R_{\mathrm{E4M3}}\!\left(\frac{\max_j|w_{b,j}|}{6\tau_W}\right),
\end{aligned}
$$

$$\hat w_{b,j}=R_{\mathrm{E2M1}}\!\left(\frac{w_{b,j}}{s_b\tau_W}\right)s_b\tau_W.$$

这里 RTN 的裁剪系数为 1；全零权重约定 $\tau_W=1$。块尺度为零时只在除法中使用安全分母，实际反量化仍严格为零。权重分片编码时复用同一个 $\tau_W$，不能对每个分片重新定标。

激活使用另一项明确的尺度合同：先把输入转为 F16，再沿最后一维分成 16 元素块，以 FP32 的 `amax * float32(1/6)` 求块尺度并舍入为 E4M3；激活二级尺度固定为 $\tau_A=1$。若显式启用 `FP4VLA_SATURATE_F16_ACTIVATIONS=1`，有限输入在 F16 转换前截到 $[-65504,65504]$；关闭时拒绝转换溢出，NaN/Inf 在两种设置下都拒绝。训练与评测各自的开关值保存在 manifest，不能把权重的动态二级尺度与激活固定尺度混写。服务日志报告 469/469 ordinary 和 7/7 category 的 W4A4 安装结果。

v12 的主结果使用 `rtn_w4a4_category`：不使用校准数据，把 472 个 ordinary recipe 张量和 7 个 category 张量全部写成 NVFP4。3 个 embedding/position 张量只量化权重。每个张量的形状、padding、尺度和实际格式写入 `ptq_recipe.json`、`category_ptq_recipe.json` 与 bake manifest；这些文件是最终配方的唯一来源。

{{fig:gptq_block}}

## 3.2 QAD：冻结基座上的低秩恢复

沿用 §2.2 的行向量约定，对一个 $K\rightarrow N$ 线性层，冻结量化权重 $W_q$ 与原偏置 $b$，增加不带偏置的低秩矩阵 $A\in\mathbb R^{r\times K}$ 与 $B\in\mathbb R^{N\times r}$：

$$y=Q_A(x)W_q^\top+b+\frac{\alpha}{r}(xA^\top)B^\top.$$

主分支接收 NVFP4 激活，残差分支读取量化前的 BF16 输入。469 个 ordinary Linear 中，语言输出投影 `lm_head` 不参与动作前向，因此只在其余 468 个模块上训练 rank=32、alpha=64 的 A/B；类别层保持冻结。训练参数以 FP32 保存，前向使用 BF16 autocast；部署旁路以 BF16 执行，按 BF16 参数载荷计账。QDQ 使用直通估计器（STE）：前向执行量化，反向把量化映射对输入的导数近似为 1。

对来自 BF16 成功轨迹的演示窗口，沿用 §2.1 的观测 $o$、动作端点 $a$、噪声 $z$ 和插值 $a_t=(1-t)z+ta$。令二值掩码 $m_{h,d}$ 标记有效时间步 $h$ 与控制维度 $d$，演示损失为

$$\ell_{\mathrm{demo}}=\frac{\sum_{h,d}m_{h,d}\left[v_\theta(o,a_t,t)_{h,d}-(a-z)_{h,d}\right]^2}{\sum_{h,d}m_{h,d}+10^{-6}}.$$

这里可训练参数 $\theta$ 仅包含 A/B。QAD 对实际微批的演示损失求平均后更新；填充区域不贡献损失。

## 3.3 OPD：学生访问状态上的教师速度监督

先固定 QAD 学生并运行训练分区，保存其观测 $o_i$、预测动作端点 $a_i^S$ 与有效掩码 $m_i$。教师标注阶段为每个窗口固定随机种子，重建噪声 $z_i$ 与时间 $t_i$，形成探针输入 $P_i=(o_i,(1-t_i)z_i+t_i a_i^S,t_i)$。缓存保存原输入、种子和 BF16 教师速度 $v_T(P_i)$；学生重放同一随机上下文，因此比较发生在同一个插值点。

$$\ell_{\mathrm{probe},i}=\frac{\sum_{h,d}m_{i,h,d}\left[v_\theta(P_i)_{h,d}-v_T(P_i)_{h,d}\right]^2}{\sum_{h,d}m_{i,h,d}}.$$

探针实现先拒绝空掩码，因此分母无需稳定项；教师标签固定且不求梯度。每次更新实际累积 $n$ 个微批时，联合目标为

$$\mathcal L_{\mathrm{OPD}}=\frac1n\sum_{j=1}^{n}\ell_{\mathrm{demo},j}+\lambda\frac1n\sum_{j=1}^{n}\ell_{\mathrm{probe},j}.$$

本轮每次优化器更新都加入教师项（`opd_every=1`），$\lambda$ 由开发集在协议候选中选择。缓存只采集一轮，追加训练期间不刷新。continued-QAD 使用同一 QAD 起点、演示顺序、实际读取数和更新数，只保留上式的演示项；OPD−continued-QAD 因而衡量额外教师监督的作用。

## 3.4 部署与闭环评测

部署保留冻结 W4A4 基座和独立 A/B adapter。把 $BA$ 预先合入权重会让残差也接收量化输入，改变训练时的函数，因此正式服务不使用这种合并。每个评测臂使用相同的任务、初态、seed、稳定步和动作步预算；`eval_manifest.json` 记录数值开关、ZeroMQ 超时、checkpoint identity 与 reset 哈希，`compare_recovery.py` 再计算五臂配对统计。
'''
    (SECTIONS / "03-方法.md").write_text(method, encoding="utf-8")

    conclusion = f'''# 5. 讨论与结论

## 5.1 结果应怎样读

PTQ−BF16 衡量全覆盖 W4A4 压力，QAD−PTQ 衡量成功演示能否恢复行为，OPD−continued-QAD 衡量在相同追加演示预算之外教师速度监督的独立作用。本文不把 APXInf 的原生 kernel 延迟解释成 GR00T 闭环速度，也不把 OPD−QAD 单独解释为纯教师增益，因为它包含额外训练阶段。

## 5.2 复现边界

实验固定 LIBERO-10 十个任务和一个训练种子；未见任务、多种子和实机验证需要另行开展。GR00T 闭环是 Torch W4A4 QDQ 参考路径，APXInf 原生执行是独立基准。QAD/OPD 只覆盖 ordinary Linear，7 个 category 层保持 W4A4 冻结；量化主分支保持 W4A4，BF16 低秩旁路的参数占用计入净压缩比。

## 5.3 结论

在冻结的 v12 协议下，BF16 为 **{bf['successes']}/{bf['episodes']}（{pct(bf['success_rate'])}）**，全 NVFP4 W4A4 PTQ 为 **{ptq['successes']}/{ptq['episodes']}（{pct(ptq['success_rate'])}）**，QAD 为 **{qad['successes']}/{qad['episodes']}（{pct(qad['success_rate'])}）**，continued-QAD 为 **{cont['successes']}/{cont['episodes']}（{pct(cont['success_rate'])}）**，QAD+OPD 为 **{opd['successes']}/{opd['episodes']}（{pct(opd['success_rate'])}）**。最终结论只由这些完整 held-out 日志支持；所有命令、截图、协议、清单和哈希都随发布包提供。
'''
    (SECTIONS / "05-讨论与结论.md").write_text(conclusion, encoding="utf-8")


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
                "本轮演示窗口的实际数量、文件顺序和尾批大小由 `recovery_manifest.json` 固定并随发布包提供；上游 DataLoader 顺序读取，不打乱、不丢弃尾批。配置 $B_\\mu=1,G=16$ 时，Trainer 按本次更新实际含有的微批数 $n$ 归一化损失，尾批不能按配置上限补齐。因而训练预算应报告 manifest 中的实际窗口读取次数，不能仅用“更新数×名义 batch”推断。",
            )
            text = text.replace(
                "顺序读取也意味着同一组 4 个窗口每轮都处于尾批，在该次平均损失中的单样本系数为 $1/4$，其余更新为 $1/16$。这是实际采样与归一化方式；continued-QAD 和 OPD 必须沿用同一数据顺序与尾批规则，才能比较相同演示预算下的附加教师监督。",
                "顺序读取会使末尾微批具有不同的归一化分母；continued-QAD 与 OPD 沿用同一数据顺序、尾批规则和优化器更新数，才能比较相同演示预算下的附加教师监督。",
            )
            text = text.replace(
                "按本轮 148 窗口的读取方式，完整完成 2,000 次更新时共有 500 次探针更新，其中 400 次含 16 个微批、100 次含 4 个微批，合计 6,800 次探针反向。缓存索引仅在实际执行探针时递增。",
                "探针更新次数、实际缓存读取数和尾批归一化分母由 OPD 的 `recovery_manifest.json` 记录；缓存索引仅在实际执行探针时递增。",
            )
            text = replace_paragraph_starting(
                text,
                "按本轮 148 窗口的读取方式",
                "探针更新次数、实际缓存读取数和尾批归一化分母由 OPD 的 `recovery_manifest.json` 记录；缓存索引仅在实际执行探针时递增。",
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
                "v12 的 `opd_every=1`，每次优化器更新都加入教师项。探针更新次数、实际缓存读取数和尾批归一化分母由 OPD 的 `recovery_manifest.json` 记录；缓存索引仅在实际执行探针时递增。实现保留原 `compute_loss` 与 `return_outputs` 协议，采用单设备 HF Trainer 的梯度累积规则。",
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
                "实际演示窗口数、尾批和读取次数以本轮 `recovery_manifest.json` 为准，附录 A 只解释归一化规则。",
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
    meta_path = PAPER / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = "v12 RTN W4A4 完整五臂评测与发布证据已核验"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rewrite_main(final)
    rewrite_other_sources()


if __name__ == "__main__":
    main()
