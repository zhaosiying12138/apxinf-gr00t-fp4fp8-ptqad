# 中文论文与知乎发布包

主稿：**PTQAD：APXInf 生态下 GR00T 的全 NVFP4 W4A4 量化与 QAD/OPD 闭环恢复**。

- `paper.html`：单文件离线阅读版，包含公式、字体、图表和 17 张原始运行截图；桌面截图可点击放大。
- `zhihu/article.md`：与 HTML 共用正文的 Markdown 发表稿，代码块和 LaTeX 公式完整保留。
- `zhihu/images/`：正文全部技术图、数据图与 17 张原始运行截图；清单见 `figures.json`。
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

当前审阅稿使用 `uv run --with-requirements paper/requirements-build.txt python paper/build_review.py` 生成。它保留全部 17 张截图，并用红色 `xxx` 标出尚未完成的结果；这是给作者审阅结构的离线预览，不是正式发布。`validate_publication.py` 会按设计拒绝含有这些占位符的审阅稿；完整结果回填后再执行上面的正式构建、验证和打包流程。

`make_figs.py` 从结果 JSON 读取图表数值，并验证缩放地址映射。截图文件不会由构建脚本改写。`validate_publication.py` 核对已审验截图的 SHA-256、全部图位、代码块、图像路径和离线资源完整性。

浏览器排版验证使用 `qa_browser.cjs`；结果写入 `validation/browser-validation.json`，预览截图保存在不入库的 `_build/`。本机使用 Node.js 22.22.1；较新的 Ubuntu 上运行本轮 Playwright 时设置了 `PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64`。数据图需要完整的本轮配对结果与原生执行记录；缺少输入时构建会报错。必须先生成 HTML/Markdown，再运行浏览器检查，最后验证和打包，不能用过期的浏览器检查替代新稿验收。

## 发表

先阅读 `zhihu/README.md`。知乎端的图片上传和公式处理取决于编辑器当前行为；本包提供 Markdown 与本地原图，不声称本地相对路径会自动成为知乎托管图片。本次交付不自动发表文章。
