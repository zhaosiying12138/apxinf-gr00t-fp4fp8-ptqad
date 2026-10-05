# 中文论文与知乎发布包

主稿：**PTQAD：APXInf 生态下 GR00T 的全 NVFP4 W4A4 量化与 QAD/OPD 闭环恢复**。

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

## 发表

先阅读 `zhihu/README.md`。发布包提供 Markdown 与本地原图；在知乎编辑器中上传图片并检查公式后，再手动发表。
