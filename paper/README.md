# 中文论文审阅包

当前版本为 **NVFP4 W4A4 RTN-PTQ + QAD/OPD 审阅稿**。方法与复现步骤对应冻结的 v12 协议；正式五臂和完整动作诊断尚未全部完成，红色 `xxx` 是待测值。校准 GPTQ 对照未执行，不保留其待填成功率，也不主张超过先进 PTQ。

- [离线 HTML](paper.html)：中文正文、公式、图表与运行图集。
- [知乎 Markdown](zhihu/article.md)：与 HTML 共用正文，图片位于 `zhihu/images/`。
- [审阅 ZIP](apxinf-gr00t-fp4fp8-ptqad-review.zip)：包含上述两种格式、正文源、当前协议及独立引擎图的数据。
- [完整复现命令](../README.md)：依赖、安装、量化、训练、评测、补充实验、截图和最终打包。

保留全部 17 个 Ubuntu 截图环节。十张未受本轮恢复实验影响的实拍继续展示；量化、教师演示采集、QAD、学生采集、教师缓存、OPD、策略服务七个环节保留文字图位，最终证据齐备后，实拍 CPU 只读核验原始日志和产物的命令，不重跑训练、量化或闭环评测。对应原图保存在工程内部，不进入当前 HTML、知乎稿或审阅 ZIP。

## 重建审阅版

在仓库根目录执行。该步骤只构建文稿，不运行模型：

```bash
uv run --with-requirements paper/requirements-build.txt python paper/build_review.py
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 node paper/qa_browser.cjs
```

构建器要求正文明确标记审阅状态、当前 RTN 协议身份和七个待补图位；结果图只显示占位，不读取其他实验的量化预算与成功率。`validation/review-build.json` 记录当前文件哈希，浏览器检查验证桌面与手机排版、公式、图像和离线资源。

## 正式发布

本目录当前不提供正式发布 ZIP。完整实验完成后，依次安装最终证据，归档训练成本与动作诊断，登记校准 GPTQ 未执行收据，实拍七个已完成证据核验环节，再执行 `update_v12_release.py`、`write_readme_v12.py`、图表构建、浏览器检查、`validate_publication.py` 和 `package_publication.py`。根 README 包含完整命令。

正式生成器要求当前协议下的真实完整证据；红色占位、未完成回合或缺少截图均不能通过正式发布验证。现存 `evidence/` 文件不因生成审阅稿而变成当前实验结果，最终安装器核验并替换其内容。
