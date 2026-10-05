# 已完成恢复搜索的成本证据

`summary.json` 从本目录原始副本复算选定训练路径和全部候选成本。`evidence_manifest.json` 保留原始路径、文件长度、SHA-256与复制用途。

验证命令：`python3 paper/collect_search_costs.py --verify paper/evidence/search_costs`。验证只需公开副本，不读取模型、教师缓存或私有原目录。

本包不包含 held-out 结果；各计时范围分别相加，不能作为统一端到端耗时。模型、观测张量和教师缓存未复制；保留生产者记录的权重身份，采集时核对观测和缓存哈希。
