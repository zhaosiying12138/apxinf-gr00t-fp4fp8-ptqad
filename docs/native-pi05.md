# π0.5 原生 NVFP4 工件准备

这条路径验证 APXInf 的原生 packed NVFP4 加载与推理。它使用 π0.5 的语言、动作与视觉投影，和 GR00T 的恢复训练、LIBERO 成功率是不同实验，分别报告结果。引擎仍保留 BF16 权重，因此 packed 文件大小不能直接解释为实际显存节省。

## 仅在 CPU 上重新打包

环境需要 Python、NumPy、safetensors。默认解释器为 `third_party/apxinf-robo/.venv/bin/python`；可以通过 `PTQAD_NATIVE_PYTHON` 覆盖。下面命令不导入 CUDA、不运行模型、不安装 wheel。入口会强制 `CUDA_VISIBLE_DEVICES` 为空，并将 OMP、MKL、OpenBLAS 线程数设为 2。

```bash
cd /absolute/path/to/fp4vla
export PROJECT=$(pwd)
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

## 四道原生 graph 验收

打包完成后，还要分别验证算子布局、捕获重放、物理 padding 和完整模型。以下步骤针对已准备的原生开发环境；必须先结束其他训练、评测和 CUDA 作业。`PTQAD_GPU_EXCLUSIVE=1` 是调用者确认串行使用 GPU 的声明，不是自动检查或抢占其他进程。

候选 wheel、完整模型验收程序和算子基准应来自同一份已应用补丁的源码。先编译候选产物；这些编译命令不执行模型，也不代表 GPU 验收通过：

```bash
APX="$PROJECT/third_party/apxinf-robo/apxinf"
ENGINE_PY="$PROJECT/third_party/apxinf-robo/.venv/bin/python"
source "$PROJECT/third_party/apxinf-robo/.venv/bin/activate"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LIBRARY_PATH="$HOME/.cuda-stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export APXINF_CUDA_ARCH=sm_120 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=target/wheel
(cd "$APX"; maturin build --release --features cuda --auditwheel skip \
  -m crates/apxinf-py/Cargo.toml)
(cd "$APX"; cargo build --release -p apxinf-model --features cuda \
  --example pi05_fp4_graph_smoke)
make -C "$PROJECT/spike" fp4_opbench
```

`.cuda-stubs` 只用于链接阶段的 `LIBRARY_PATH`，不能放进运行时 `LD_LIBRARY_PATH`。实验机的真实 CUDA 库由系统动态链接配置解析；其他机器如需设置 `LD_LIBRARY_PATH`，应使用真实 CUDA runtime 库目录，例如 `/usr/local/cuda/lib64`，WSL 驱动来自真实的 `/usr/lib/wsl/lib`。`setup/02_build_engine.sh` 同时构建并安装 wheel；上述候选构建将安装留给验收入口。

正式验收使用仓内的串行入口；设置本轮 overlay，避免误用脚本的实验机默认模型目录：

```bash
cd "$PROJECT"
NATIVE_PI05_MODEL="$RUN/model" NATIVE_PY="$ENGINE_PY" PTQAD_GPU_EXCLUSIVE=1 \
  bash exp/run_native_graph_gates.sh "$PROJECT/results/native_graph_new_run"
```

输出目录必须不存在。脚本按下表执行四道检查，前三道还要求日志明确出现 **1 个测试通过、0 个失败**，防止过滤器没有选中测试却返回成功。任一步失败立即结束，保留日志；全部通过后才安装本次候选 wheel，并记录实际导入扩展的 SHA-256 和 getter 是否存在。

| 顺序 | 实际入口 | 检查范围 |
|---|---|---|
| 1 | `fp4_contract_rowmajor_and_tensor_scale` | 两种非方阵、正负数、零块与非单位张量 scale；逐元素对照 CPU 反量化乘积，核验输出布局。 |
| 2 | `fp4_graph_replay_bf16_and_distinct_scales` | 同形状投影使用不同 scale 值与不同 scale 缓冲指针；污染 arena 后捕获，并改变同一 BF16 输入缓冲，在四次重放中核对每个输出，包括全零输入。 |
| 3 | `fp4_activation_padding_zero_after_capture` | 每次重放前污染物理 scale tile，对 M=3、K=48 的全部 512 字节检查清零及有效值，覆盖行与 K 两个 padding 方向。 |
| 4 | `pi05_fp4_graph_smoke <overlay> 10` | 完整 π0.5 `nvfp4_static` 必须达到 `RequireGraph`；先取两组 eager 参考，释放 eager plan 后仅保留一份 graph arena，再按噪声索引 0、1、0 重放，要求动作有限且相对对应 eager 结果的最大绝对差不超过 0.01。 |

前三道在 APXInf checkout 中分别使用 `cargo test --release -p apxinf-cuda <上表名称> -- --ignored --nocapture`，第四道执行已编译的 `target/wheel/release/examples/pi05_fp4_graph_smoke`。共同环境是 `APXINF_CUDA_ARCH=sm_120`、`CARGO_BUILD_JOBS=1`、`CARGO_TARGET_DIR=target/wheel`；正式日志和退出状态由 wrapper 保留，不建议绕过 wrapper 串接会覆盖日志的命令。

[CPU 准备凭据](../paper/validation/native-graph-preparation/README.md) 记录编译、纯 CPU arena 预算检查及补丁应用检查；其中没有把 GPU 验收标为通过。验收结论只能来自本轮四道检查的实际日志。完整模型检查比较的是同一 NVFP4 路径的 eager/graph 一致性，不证明与 BF16 模型等价，也不证明 LIBERO 成功率或速度提升。

此实现将 BF16/F16 转换、在线量化中间缓冲与输出放入 `GraphWorkspace`，另复用 context 持有的 64 MiB scratch。scale tile 的完整清零是捕获图中的操作，重放时也必须执行。原 BF16 权重、packed 权重和 graph arena 仍共同占用显存。计划缓存按线程、设备、stream、形状与 scratch 容量组织，准备和捕获须在同一宿主线程；当前检查不覆盖多 GPU 或不匹配的低层 device/context。

## 后续原生推理的边界

CPU 打包脚本不负责原生 wheel 构建、安装、推理与计时；四道验收通过并安装候选 wheel 后，再进行独立计时。后续调用 `AutoPolicy` 时传入本轮 `model/`，再分别启动 BF16 和 `nvfp4_static` 的独立进程，避免两模型同时驻留。计时记录还需要绑定实际 wheel 与安装的扩展库哈希、GPU、输入、随机种子、预热和重复次数；重新打包成功本身不能证明速度提升或任务成功率。

GPU 空闲并确认原生 wheel 已安装后，以下两个命令必须串行运行。`--out` 拒绝已有文件；默认显式读取新模型目录下的 `norm_stats.json` 和 `paligemma_tokenizer.model`。

```bash
ENGINE_PY="$PROJECT/third_party/apxinf-robo/.venv/bin/python"
(cd /tmp; "$ENGINE_PY" "$PROJECT/exp/bench_engine.py" \
  --model-dir "$RUN/model" --variant bf16 --device cuda:0 \
  --warmup 10 --samples 30 --seed 7 --model-seed 0 \
  --out "$PROJECT/results/engine/pi05_bf16_new_run.json")
(cd /tmp; "$ENGINE_PY" "$PROJECT/exp/bench_engine.py" \
  --model-dir "$RUN/model" --variant nvfp4_static --device cuda:0 \
  --warmup 10 --samples 30 --seed 7 --model-seed 0 \
  --out "$PROJECT/results/engine/pi05_nvfp4_new_run.json")
```

本基准要求原生 `model_variant` 与请求完全一致，并核对 NVFP4 producer manifest、源 checkpoint 和每个 packed 文件的实际 SHA-256。结果记录实际导入的扩展 SO、Python 源文件、包版本和本地检查过的 Rust 源码散列。SO 哈希是已加载程序的身份；当前工作树源码散列不能单独证明该 SO 的构建来源，构建日志与 wheel 身份仍需一并保留。

只读属性 `policy.model_runner.execution_mode` 返回当前已有隐式 plan 的状态：π0.5 使用 `unprepared`、`graph` 或 `eager`，GR00T 的 graph 计划返回 `cuda-graph`，基准保留模型给出的原字符串；读取属性不会初始化 CUDA 资源、创建 plan 或执行推理。计时入口在 warmup 后和每个计时样本后读取该属性，查询本身位于计时区间之外，保存到 `execution_mode.after_warmup` 与 `execution_mode.after_each_timed_sample`。旧运行库未提供 getter 时记录 `null`，不能据此推断为 graph。

已有 JSON 可完全在 CPU 上查看实际模式：

```bash
python3 - "$PROJECT/results/engine/pi05_nvfp4_new_run.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
print(json.dumps(result["execution_mode"], indent=2))
PY
```

任何“graph 路径耗时”结论都要求 warmup 后与所有计时样本的实际模式均为 `graph`，同时输出有限且输入身份一致；缺少警告或看起来更快均不能替代模式记录。四道检查通过后，也需要用新 wheel 重新测量所有受影响原生 variant，包括 BF16。PyTorch 恢复训练不使用该扩展，其成功率与成本仍独立报告。

`model_ms` 是阻塞原生模型调用直到取得 CPU float32 动作的时间，包括输入传输、GPU 执行、同步与 D2H；`total_ms` 还包括策略的预处理、分词和动作后处理；外层 `wrapper_ms` 另记完整 Python 调用。三组数组按实际顺序保存，p50/p99 采用 NumPy 线性插值。源码检查确认 π0.5 的 host-return transfer 和 GR00T executor 在 D2H 前同步 CUDA，因此这不是只量 kernel 提交的时间，也不是 CUDA event 的纯 kernel 时间。

`nvidia-smi` 功耗或显存不可用时记录 `null` 和错误；已有值属于显式 `--telemetry-device` 选择的整卡计时阶段，可能包含桌面和其他进程。CUDA 可见设备重映射与 nvidia-smi 编号也要分别核对。任何动作非有限、variant 不符、来源校验失败或清理失败都会阻止成功 JSON 发布。
