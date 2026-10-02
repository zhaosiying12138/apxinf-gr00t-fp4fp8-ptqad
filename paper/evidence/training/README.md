# v11 训练证据

本目录只保留当前 v11 W4A4 运行的轻量基座清单。QAD、continued-QAD、OPD 的训练请求、教师缓存和逐回合日志会在 v11 运行完成后由 `paper/extract_final_evidence.py` 从最终运行目录复制并绑定 SHA-256；运行目录由最终 `run_manifest.json` 明确记录，不在此文件中硬编码具体 rN 后缀。旧协议和旧 scope 的训练快照不纳入发布包。
