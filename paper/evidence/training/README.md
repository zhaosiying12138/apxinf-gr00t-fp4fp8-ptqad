# v11 训练证据

本目录当前只保留 v11 W4A4 运行的轻量基座清单。运行完成后，`paper/materialize_final_evidence.py` 根据 `final_manifest.json` 整理发布证据，并调用 `paper/collect_training_costs.py` 核验、复制 QAD、continued-QAD 和 OPD 的训练请求、训练日志、运行清单与教师缓存元数据，绑定 SHA-256 并汇总成本。模型权重、观测张量和教师缓存张量留在本地。

`paper/extract_final_evidence.py` 负责从最终清单及其绑定的 held-out 配对结果提取数值，不复制训练日志。各阶段目录由运行清单定位，此处不硬编码具体 rN 后缀；发布包只纳入最终协议选中的证据。
