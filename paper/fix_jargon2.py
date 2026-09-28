# -*- coding: utf-8 -*-
"""Sweep remaining lab jargon: experiment 'arm' phrases, artifact-as-工件,
twin-forward, z-range, spike gloss, 混沌解耦, 更狠, 偏易, 离线 proxy.
机械臂 (robot arm) untouched; Rust 'match 臂' -> 'match 分支'."""
import io, pathlib

SECS = pathlib.Path(__file__).resolve().parent / "sections"

SPECIFIC = {
"01-摘要与引言.md": [
    ("量化更狠、恢复更难", "量化更激进、恢复更难"),
    ("QAD（演示蒸馏）与 OPD（teacher-KL）双臂同达", "QAD（演示蒸馏）与 OPD（teacher-KL）两路同达"),
    ("所有臂的取舍均以 10×10 全量闭环裁定", "所有配置的取舍均以 10×10 全量闭环裁定"),
    ("我们实证离线指标与闭环损伤混沌解耦", "我们实证离线指标与闭环损伤脱钩"),
    ("五臂 PTQ 消融给出模块级 NVFP4 敏感度排序", "五种配方的 PTQ 消融给出模块级 NVFP4 敏感度排序"),
    ("与闭环损伤混沌解耦——闭环成功率是 VLA 量化的唯一裁判", "与闭环损伤脱钩——闭环成功率是 VLA 量化的唯一裁判"),
    ("三臂恢复", "三路恢复"),
],
"02-背景与相关工作.md": [
    ("两个恢复臂恰好在两项上分工", "两条恢复路线恰好在两项上分工"),
    ("§4.4 的三臂对照逐项兑现", "§4.4 的三路对照逐项兑现"),
    ("（本文的 RWR 臂）", "（本文的 RWR 对照组）"),
],
"03-方法.md": [
    ("完整过程见 §A.2.1 的方法自述与仓库 spike/", "完整过程见 §A.2.1 的方法自述与仓库探针实验目录 spike/"),
    ("§4.3 的五臂分配消融", "§4.3 的五配方分配消融"),
    ("§4.4 的三臂对照将逐条兑现", "§4.4 的三路对照将逐条兑现"),
    ("再把 §4.4 的模块差异翻译进", "再把 §4.3 的模块差异翻译进"),
    ("与 §4.4 五臂消融收敛出的敏感度地图", "与 §4.3 五配方消融收敛出的敏感度地图"),
    ("**三个恢复臂**", "**三条恢复路线**"),
    ("避免 twin-forward 在 WSL 上的确定性崩溃", "避免教师与学生同批双前向（twin-forward）在 WSL 上的确定性崩溃"),
    ("经处理器归一化，z-range 校验）", "经处理器归一化并校验取值范围）"),
    ("离线指标与闭环损伤混沌解耦，故一切臂间取舍", "离线指标与闭环损伤脱钩，故一切配置间取舍"),
    ("快速筛臂，已知偏易", "快速筛配置，已知偏乐观"),
],
"04-实验.md": [
    ("再排除\"实现工件\"", "再排除\"实现自身缺陷\""),
    ("**排除实现工件·跨模型跨路径交叉验证**", "**排除实现缺陷·跨模型跨路径交叉验证**"),
    ("仅用于臂间快筛", "仅用于配置间快筛"),
    ("所有量化臂与恢复臂共用", "所有量化配置与恢复配置共用"),
    ("每个臂的 checkpoint", "每个配置的 checkpoint"),
    ("calib 臂把逐层输出 MSE 压到", "calib 配置把逐层输出 MSE 压到"),
    ("**逐臂排除·五臂分配消融**。固定校准不变、只动精度分配，逐臂收敛到可用配方：aggr 臂（", "**逐个排除·五配方分配消融**。固定校准不变、只动精度分配，逐个收敛到可用配方：aggr 配置（"),
    ("fp8 臂（backbone 全 per-channel FP8", "fp8 配置（backbone 全 per-channel FP8"),
    ("mixed 臂（复刻 NVIDIA 分配", "mixed 配置（复刻 NVIDIA 分配"),
    ("后两臂进入全量协议", "后两种进入全量协议"),
    ("三臂共用加性 LoRA", "三路共用加性 LoRA"),
    ("三臂都\"看起来在学\"", "三路都\"看起来在学\""),
    ("微波炉 6%→100% 双臂同达），两臂各自仅剩一次失败", "微波炉 6%→100% 两路同达），两路各自仅剩一次失败"),
    ("**离线 proxy 为何不能用**", "**离线替代指标为何不能用**"),
    ("作臂间快筛，结果连闭环 95% 以上的臂都给出", "作配置间快筛，结果连闭环 95% 以上的配置都给出"),
    ("使任何臂间比较失去分辨率", "使任何配置间比较失去分辨率"),
    ("早期两臂离线池化相关悬殊", "早期两种配置的离线池化相关悬殊"),
    ("perplexity 是有效 proxy", "perplexity 是有效替代指标"),
    ("**收尾·更狠一档的 43.4% 基座**", "**收尾·更激进一档的 43.4% 基座**"),
],
"05-讨论与结论.md": [
    ("五臂差分给出模块级敏感度地图", "五配方差分给出模块级敏感度地图"),
    ("QAD 与 OPD 双臂", "QAD 与 OPD 两路"),
    ("与闭环损伤混沌解耦，闭环成功率是 VLA 量化的唯一裁判", "与闭环损伤脱钩，闭环成功率是 VLA 量化的唯一裁判"),
    ("全部实验臂零代码改动换入即测", "全部实验配置零代码改动换入即测"),
    ("（含全套 spike 工具）", "（含全套探针实验工具，spike/ 目录）"),
    ("五臂精度分配", "五配方精度分配"),
    ("RWR 对照臂是", "RWR 对照组是"),
],
"14-附录A-核心源码走读与APXInf框架解析.md": [
    ("全部 match 臂（漏一处", "全部 match 分支（漏一处"),
    ("这是对早期 twin-forward 方案在 WSL 上确定性崩溃的修复", "这是对早期教师学生同批双前向（twin-forward）方案在 WSL 上确定性崩溃的修复"),
    ("## A.4.3 RWR 在线对照臂（lora_rwr.py）", "## A.4.3 RWR 在线对照组（lora_rwr.py）"),
    ("z-range 校验=1.00）注入训练批", "取值范围校验全数通过）注入训练批"),
    ("全部实验臂——五臂 PTQ、LoRA 合并产物、RWR 产物——", "全部实验产物——五种 PTQ 配置、LoRA 合并产物、RWR 产物——"),
    ("| 五臂量化写盘 |", "| 五配方量化写盘 |"),
    ("`GR00T_BASE_CKPT=<fp8臂>", "`GR00T_BASE_CKPT=<fp8 基座>"),
    ("| RWR 臂 |", "| RWR 组 |"),
    ("在 fp4vla 仓库 `spike/` 与 `quant/` 下成对出现", "在 fp4vla 仓库的探针实验目录 `spike/` 与 `quant/` 下成对出现"),
],
}

for fn, pairs in SPECIFIC.items():
    p = SECS / fn
    t = io.open(p, encoding="utf-8").read()
    for old, new in pairs:
        if old not in t:
            print(f"MISS {fn[:2]}: {old[:34]}")
        else:
            t = t.replace(old, new)
    io.open(p, "w", encoding="utf-8", newline="\n").write(t)
    print("done", fn[:2])

# blanket: 工件 -> 产物 (after specific pairs)
for p in sorted(SECS.glob("*.md")):
    t = io.open(p, encoding="utf-8").read()
    n = t.count("工件")
    if n:
        t = t.replace("工件", "产物")
        io.open(p, "w", encoding="utf-8", newline="\n").write(t)
    print(f"blanket {p.name[:2]}: {n} replaced")

# final audit: any 臂 outside 机械臂 / any leftover jargon
print("-- audit --")
BAD = ["五臂", "三臂", "恢复臂", "对照臂", "臂间", "工件", "twin-forward 在", "z-range 校验", "混沌解耦", "更狠", "偏易"]
for p in sorted(SECS.glob("*.md")):
    incode = False
    for i, ln in enumerate(io.open(p, encoding="utf-8").read().splitlines(), 1):
        if ln.startswith("```"):
            incode = not incode
            continue
        if incode:
            continue
        stripped = ln.replace("机械臂", "")
        hits = [k for k in BAD if k in stripped]
        if hits:
            print(f"{p.name[:2]}:{i} {hits}: {ln[:60]}")
print("audit done")
