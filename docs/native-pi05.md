# π0.5 原生 NVFP4 工件准备

这条路径验证 APXInf 的原生 packed NVFP4 加载与推理。它使用 π0.5 的语言、动作与视觉投影，和 GR00T 的恢复训练、LIBERO 成功率是不同实验，分别报告结果。引擎仍保留 BF16 权重，因此 packed 文件大小不能直接解释为实际显存节省。

## 仅在 CPU 上重新打包

环境需要 Python、NumPy、safetensors。默认解释器为 `third_party/apxinf-robo/.venv/bin/python`；可以通过 `PTQAD_NATIVE_PYTHON` 覆盖。下面命令不导入 CUDA、不运行模型、不安装 wheel。入口会强制 `CUDA_VISIBLE_DEVICES` 为空，并将 OMP、MKL、OpenBLAS 线程数设为 2。

```bash
cd /absolute/path/to/fp4vla
RUN="$PWD/weights/native_pi05_new_run"
bash exp/prepare_native_pi05.sh pack \
  --base "$PWD/weights/pi05_libero_base" --out "$RUN" --chunk-rows 512
bash exp/prepare_native_pi05.sh model-overlay \
  --base "$PWD/weights/pi05_libero_base" --out "$RUN"
```

`--out` 必须是源模型目录之外的绝对路径。每个阶段拒绝覆盖已有输出；失败会保留日志和部分工件，重新尝试应使用新的运行目录。脚本针对语言 18 层、动作 18 层、视觉 27 层的 π0.5 源模型，要求完整生成 99 个投影张量：语言与动作各包含 QKV、gate/up，视觉包含 QKV。其他模型结构不能悄悄截断层数后沿用此命令。

转换器通过 safetensors 按需读取权重，用分块量化限制 CPU 临时内存。语言分支将 `1 + gamma` 归一化权重折叠进消费它的投影；动作与视觉分支不做这一折叠。输出包含 E2M1 数据 nibble、每 16 元素一个 E4M3 scale 和物理 scale 排列；此原生接口固定 `tscale = 1`。

**产物覆盖与执行覆盖分开统计。** 当前 `Fp4Blocks::vision` 委托 BF16 视觉路径；生成的 27 个视觉 QKV 产物不代表视觉算子已经使用 FP4。语言与动作分支按实际投影名称选择低精度产物，其余投影保留 BF16。99 个产物是打包完整性检查，不能据此声称 99 个投影都在一次推理中执行 FP4。

## 接受条件与溯源

`pack_input_manifest.json` 在计算前冻结源 checkpoint、实际转换源码 SHA-256、命令、CPU 环境与 Git HEAD。运行结束再次校验源码和源 checkpoint 未改变，然后检查完整张量集合、实际文件长度和总字节数。

`packed/producer_manifest.json` 只有完整验证通过才生成，记录输入 manifest、转换 manifest 和全部 198 个数据/scale 文件的 SHA-256。`packed/manifest.json` 中的 `orig_bytes` 是选定投影的 **FP32 源数据** 字节数；它既不是完整模型的字节数，也不是 BF16 对照的字节数。论文计算压缩率时必须明确分母。

第二阶段重新核验源 checkpoint 与 packed 文件，再建立新的 `model/` 目录：大权重以符号链接复用，小型配置、统计和 tokenizer 文件按原样复制，`model/fp4` 只指向本轮新工件。原模型目录中的 `fp4` 链接不受影响。`overlay_manifest.json` 记录新目录、源权重及所有复制的元数据哈希。

## 后续原生推理的边界

原生 wheel 的构建、安装、推理与计时均不在此脚本内。后续调用 `AutoPolicy` 时传入本轮 `model/`，再分别启动 BF16 和 `nvfp4_static` 的独立进程，避免两模型同时驻留。计时记录还需要绑定实际 wheel 与安装的扩展库哈希、GPU、输入、随机种子、预热和重复次数；重新打包成功本身不能证明速度提升或任务成功率。
