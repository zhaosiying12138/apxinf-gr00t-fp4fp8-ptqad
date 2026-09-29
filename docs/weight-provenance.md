# 模型权重的下载身份与验收

本轮的字节身份见 `setup/locks/model-sources.json`。首先对已有本地文件计算 SHA-256；Cosmos 与 π0.5 另核对 Hugging Face 下载记录中的 revision 与 ETag。GR00T 没有保留原始下载记录，因此进一步查询官方仓库的不可变 revision：两个大权重分片的 LFS SHA-256 与大小匹配，五个辅助文件的下载字节也逐项匹配。这次补充核验未下载大权重、安装环境或调用 GPU。

| 资源 | 可确认的来源 | 验收与边界 |
|---|---|---|
| `nvidia/Cosmos-Reason2-2B` | HF commit `9ce19a195e423419c349abfc86fd07178b230561` | 10 个运行所需文件全部与已有下载 metadata 的 revision/ETag 对应；实际命名路径指向此 snapshot |
| `lerobot/pi05_libero_base` | HF commit `a217bfd3b14673cf2ce597e69997ab21866438dd` | 权重、config、两个 processor 文件全部匹配下载 metadata；其他本地附加文件不据此归属于 HF commit |
| `nvidia/GR00T-N1.7-LIBERO` | 等价下载 commit `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21` | 共 7 个文件与本轮实际字节身份一致；原始下载日期及当时 revision 仍未恢复，不将等价下载版本冒称为原始下载记录 |
| π0.5 `norm_stats.json` | 既有下载脚本记录的 OpenPI GCS URL | 没有原始 HTTP 响应、GCS generation 或 ETag；只确认现有文件 SHA-256，不能把当前远端版本反推为下载时版本 |
| `paligemma_tokenizer.model` | 既有脚本记录的 Big Vision GCS URL | 同上；模型与 tokenizer 的下载来源分别记录 |

`setup/03_download_weights.sh` 默认固定上述三个 commit；下载后仍必须通过已知字节校验。GR00T 的补充来源收据见 `setup/locks/gr00t-equivalent-revision.json`，原始 API 响应一并保存。等价性指这七个运行所需文件的内容身份，不扩展到未核验的上游文件。

```bash
# 完整模型资源；默认固定 huggingface_hub==0.36.2 下载客户端
bash setup/03_download_weights.sh

# 仅 GR00T + Cosmos，核心闭环无需安装 π0.5 参考环境或下载其权重
bash setup/03_download_weights.sh --core-only

# CPU 只读：验证当前本地资源，不下载、不加载模型
bash setup/03_download_weights.sh --verify-only
# 也可仅核验核心资源
bash setup/03_download_weights.sh --verify-only --core-only
```

脚本只下载清单中运行所需的文件，不下载全套 GR00T 训练状态。已有文件若散列不一致，会在下载前终止；外部 tokenizer/statistics 下载到临时文件，散列通过才创建目标，已有目标不被替换。显式替换实验资源时使用新目录和独立审阅过的清单，不把旧文件自动覆盖为新实验。

可以设置 `GR00T_HF_REVISION`、`COSMOS_HF_REVISION`、`PI05_HF_REVISION` 改变下载选择器，但这不会关闭已知字节验收。`PTQAD_MODEL_SOURCE_MANIFEST` 可指定另一个经审阅的身份清单；`PTQAD_WEIGHTS_DIR` 可指定下载目录；`PTQAD_HF_HUB_VERSION` 可指定下载客户端版本。非默认目录需显式配置训练、仿真和原生基准路径，不能假设 `setup/06_install_recovery.sh` 自动切换所有资源。

权重 SHA-256 只证明字节身份，不证明算法等价或闭环效果。当前实验仍需独立的量化验收、实际导出 checkpoint、完整评测与运行时源码清单。安装脚本的全新机器下载流程尚未实际重跑；本次完成的是原有文件验真与脚本静态/CPU 验证。
