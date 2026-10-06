# 中文论文与知乎发布包

主稿：**PTQAD：APXInf 生态下 GR00T 的全 NVFP4 W4A4 量化与 QAD/OPD 闭环恢复**。

本项目研究在 APXInf 推理生态中把 GR00T VLA 的权重和激活压到 NVFP4 W4A4 后，如何用成功演示的低秩适配（QAD）和学生访问状态上的教师监督（OPD）保持 LIBERO 闭环行为。发布包把论文正文、可离线阅读的 HTML、知乎 Markdown、逐回合日志、协议与哈希清单、图表和真实 Ubuntu 终端截图放在同一证据链中；结果只从冻结协议的五臂 held-out 评测读取。

完整复现需要 WSL2 Ubuntu、支持 `sm_120` 的 NVIDIA GPU、CUDA/PyTorch、独立的 LIBERO 仿真环境，以及 GR00T、Cosmos 和可选 APXInf 的固定版本。依赖版本、权重来源和安装脚本见仓库根目录 [README.md](../README.md) 的“依赖”“安装”和“复现流程”章节；从源码编译到五臂闭环的逐阶段命令见其“从编译到执行的命令矩阵”，更详细的故障恢复规则见 [docs/reproduce-ptqad.md](../docs/reproduce-ptqad.md)。

- `paper.html`：单文件离线阅读版，包含公式、字体、图表和全部 17 个截图环节；截图可点击放大。
- `zhihu/article.md`：与 HTML 共用正文的 Markdown 发表稿，代码块和 LaTeX 公式完整保留。
- `zhihu/images/`：技术图、数据图与运行截图。`figures.json` 记录图位说明；`evidence/captures.json` 与 `evidence/retained_captures.json` 记录来源和哈希，正式校验合并 15 张当前截图与 2 张经批准保留的 BF16 截图，覆盖全部 17 个图位。
- `sections/`：唯一正文源；修改正文后重建两种格式，避免手动修改生成文件。
- `evidence/`：本文实验的结果数组、配置、运行日志、来源散列与截图清单。
- `validation/`：发布完整性与浏览器排版校验报告。

## 重建

需要 Python 3、uv、Node.js 和中文字体（如 Noto Sans CJK SC）。本轮直接构建依赖固定在 `requirements-build.txt`；它们与 GPU 实验环境分开。图像转换还需要系统 Cairo 动态库。KaTeX 0.16.22 已随仓库保存并附 MIT 许可；读取 HTML 不需要 Node、Python 或联网。

```bash
# 在仓库根目录准备独立浏览器工具；不改 GPU Python 环境。
# setup/01 安装 uv 后，新 shell 尚未加载配置时也能找到它。
export PATH="$HOME/.local/bin:$PATH"
npm install --prefix paper/_build/renderer --save-exact playwright@1.58.2
node paper/_build/renderer/node_modules/playwright/cli.js install chromium

python3 paper/make_figs.py
bash paper/figs/render_pngs.sh
uv run --with-requirements paper/requirements-build.txt python paper/build_html.py
python3 paper/export_zhihu.py
node paper/qa_browser.cjs
uv run --with-requirements paper/requirements-build.txt python paper/validate_publication.py
uv run --with-requirements paper/requirements-build.txt python paper/package_publication.py
```

当前发布稿由 `build_html.py` 和 `export_zhihu.py` 生成，展示全部 17 个已登记图位。QAD、教师标注、OPD、策略服务和学生采集五个受 W4A4 实现影响的环节已按本轮命令重新实拍；其余环节沿用已核验来源。`validate_publication.py` 会核对最终结果、图位、截图哈希、代码块和离线资源，拒绝缺失证据的发布包。

`make_figs.py` 从结果 JSON 读取图表数值，并验证缩放地址映射。截图文件不会由构建脚本改写。`validate_publication.py` 核对已审验截图的 SHA-256、全部图位、代码块、图像路径和离线资源完整性。

`qa_browser.cjs` 将排版报告写入 `validation/browser-validation.json`，预览截图保存在不入库的 `_build/`。参考环境为 Node.js 22.22.1；本机 Playwright 使用 `PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64`。数据图依赖完整配对结果和原生执行记录。每次改稿都按“生成 HTML/Markdown → 浏览器检查 → 验证 → 打包”的顺序重建。

## 结果与证据索引

发布包只接受 `evidence/final_manifest.json` 指定的最终协议和五个 held-out 评测臂。结果的唯一数据源是 `evidence/final_results.json`，配对区间与检验来自 `evidence/paired_comparison.json`；不要从 HTML、Markdown 或旧运行目录手工复制数字。下面的命令会直接打印当前证据中的五臂分子、分母、成功率、配对差值和 Holm 校正后的检验结果，因此重评完成后无需改写命令：

当前 `final_manifest.json` 绑定 `results/reruns/v11_release_ptq_rerun_20261006_01/`。其中 PTQ 臂来自 2026-10-06 的完整 W4A4 重评：端口 6891、单任务 timeout 3600 秒、FFmpeg 已预检，10 个任务共 160 回合，结果为 145/160；旧的部分超时目录不属于发布证据。原始重评目录、组合配对目录和协议 SHA 可由 `final_results.json` 的 `source` 字段复核。

```bash
# 从仓库根目录执行。
python3 - <<'PY'
import json
from pathlib import Path

root = Path("paper/evidence")
results = json.loads((root / "final_results.json").read_text())
arms = [("bf16", "BF16"), ("ptq", "W4A4 PTQ"),
        ("qad", "PTQ + QAD"), ("continued_qad", "continued-QAD"),
        ("qad_opd", "PTQ + QAD + OPD")]
print("scope:", results.get("scope"), "status:", results.get("status"))
for key, label in arms:
    item = results.get("public_arms", {}).get(key)
    if item is None:
        item = results.get("control", {}).get(key)
    if item is None:
        print(f"{label}: missing")
        continue
    print(f"{label}: {item['successes']}/{item['episodes']} "
          f"({100 * item['success_rate']:.2f}%)")
print("paired contrasts:")
for key, item in results.get("uncertainty", {}).get("contrasts", {}).items():
    print(f"  {key}: {item['difference_pp']:+.2f} pp; "
          f"95% CI={item['pointwise_ci_pp']}; "
          f"Holm p={item['holm_adjusted_p']:.6g}")
PY
```

编码预算、量化覆盖、训练成本、逐任务结果和墙钟时间见仓库根目录 [README.md](../README.md)；原始逐回合日志由 `final_manifest.json` 的五个 `heldout_*` 身份索引。`validate_publication.py` 会拒绝缺少任一 held-out 原始日志、协议哈希不一致、结果状态未完成、截图哈希不匹配或遗留红色占位符的发布包。

## Ubuntu 截图索引

发布包保留 17 个紫色 Ubuntu 终端图位。每张图均有对应的脚本、原始日志、窗口绑定 sidecar、裁剪记录和 SHA-256；完整来源见 [`evidence/captures.json`](evidence/captures.json) 与 [`figures.json`](figures.json)。

| 图位 | 执行环节 | 图片 | 原始脚本与日志 |
|---:|---|---|---|
| 01 | 完整模型校准 | [shot_collect.png](zhihu/images/shot_collect.png) | [`shot_collect.sh`](evidence/captures/shot_collect.sh) · [`log`](evidence/captures/shot_collect.log) |
| 02 | 配方与编码预算核验 | [shot_bake.png](zhihu/images/shot_bake.png) | [`shot_bake.sh`](evidence/captures/shot_bake.sh) · [`log`](evidence/captures/shot_bake.log) |
| 03 | π0.5 单层 NVFP4 打包 | [shot_packed.png](zhihu/images/shot_packed.png) | [`shot_packed.sh`](evidence/captures/shot_packed.sh) · [`log`](evidence/captures/shot_packed.log) |
| 04 | 探针、掩码与尾批梯度 | [shot_probe.png](zhihu/images/shot_probe.png) | [`shot_probe.sh`](evidence/captures/shot_probe.sh) · [`log`](evidence/captures/shot_probe.log) |
| 05 | W4A4 QAD 训练 | [shot_qad.png](zhihu/images/shot_qad.png) | [`shot_qad.sh`](evidence/captures/shot_qad.sh) · [`log`](evidence/captures/shot_qad.log) |
| 06 | 学生状态教师标注 | [shot_opdcache.png](zhihu/images/shot_opdcache.png) | [`shot_opdcache.sh`](evidence/captures/shot_opdcache.sh) · [`log`](evidence/captures/shot_opdcache.log) |
| 07 | QAD→OPD 续训 | [shot_opd.png](zhihu/images/shot_opd.png) | [`shot_opd.sh`](evidence/captures/shot_opd.sh) · [`log`](evidence/captures/shot_opd.log) |
| 08 | 激活重算与 LoRA 梯度 | [shot_qat.png](zhihu/images/shot_qat.png) | [`shot_qat.sh`](evidence/captures/shot_qat.sh) · [`log`](evidence/captures/shot_qat.log) |
| 09 | W4A4 服务与健康 RPC | [shot_evalserver.png](zhihu/images/shot_evalserver.png) | [`shot_evalserver.sh`](evidence/captures/shot_evalserver.sh) · [`log`](evidence/captures/shot_evalserver.log) |
| 10 | LIBERO 闭环与观测采集 | [shot_rollout.png](zhihu/images/shot_rollout.png) | [`shot_rollout.sh`](evidence/captures/shot_rollout.sh) · [`log`](evidence/captures/shot_rollout.log) |
| 11 | 量化格点与 checkpoint 校验 | [shot_verify.png](zhihu/images/shot_verify.png) | [`shot_verify.sh`](evidence/captures/shot_verify.sh) · [`log`](evidence/captures/shot_verify.log) |
| 12 | FP8 描述符探针 | [shot_fp8probe.png](zhihu/images/shot_fp8probe.png) | [`shot_fp8probe.sh`](evidence/captures/shot_fp8probe.sh) · [`log`](evidence/captures/shot_fp8probe.log) |
| 13 | 9 种 GEMM 形状 | [shot_gemm.png](zhihu/images/shot_gemm.png) | [`shot_gemm.sh`](evidence/captures/shot_gemm.sh) · [`log`](evidence/captures/shot_gemm.log) |
| 14 | 18 个实际层形状 | [shot_opbench.png](zhihu/images/shot_opbench.png) | [`shot_opbench.sh`](evidence/captures/shot_opbench.sh) · [`log`](evidence/captures/shot_opbench.log) |
| 15 | GR00T BF16 执行 | [shot_gr00t.png](zhihu/images/shot_gr00t.png) | [保留图 provenance](evidence/retained_captures.json) |
| 16 | π0.5 BF16 执行 | [shot_pi05.png](zhihu/images/shot_pi05.png) | [保留图 provenance](evidence/retained_captures.json) |
| 17 | π0.5 NVFP4 完整路径 | [shot_nvfp4.png](zhihu/images/shot_nvfp4.png) | [`shot_nvfp4.sh`](evidence/captures/shot_nvfp4.sh) · [`log`](evidence/captures/shot_nvfp4.log) |

截图必须为裁剪后的 `3840×2280`，画面中的命令和数字必须能在同名 `.sh`、`.log` 与 sidecar 中复核。根目录 [README.md](../README.md) 的“从编译到执行的命令矩阵”和 [复现教程](../docs/reproduce-ptqad.md) 给出从环境安装、APXInf 编译、W4A4 校准/训练、五臂闭环到截图登记的完整命令；本 README 只维护发布包的入口和证据索引。

## 发表

先阅读 `zhihu/README.md`。发布包提供 Markdown 与本地原图；在知乎编辑器中上传图片并检查公式后，再手动发表。
