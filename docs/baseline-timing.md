# PyTorch 参考入口耗时复现

这里只运行两条独立的 batch=1 合成输入基准，不训练模型、不启动 LIBERO。GR00T 基准复用核心恢复环境；π0.5 的 LeRobot 环境是可选参考环境，**核心量化、QAD、OPD 和闭环评测均不需要安装它**。每次只启动一个 GPU 模型进程，先退出训练、评测服务或其他基准，再执行下一个命令。

公开的 `baselines/` 只需要 `bench_gr00t_pt.py` 和 `bench_pi05_lerobot.py`。第二个文件复用第一个文件里的小型校验函数。不要复制 `baselines/Isaac-GR00T` 或任何 venv；GR00T 源码由固定上游 revision 加仓内补丁恢复。

## 1. GR00T：复用已恢复的模型环境

完成 `setup/06_install_recovery.sh` 后执行：

```bash
cd /absolute/path/to/fp4vla
source setup/recovery-env.sh
export PROJECT=$(pwd)
export PTQAD_BASE="$PROJECT/weights/GR00T-N1.7-LIBERO/libero_10"

(cd "$GR00T_REPO"; "$PTQAD_PYTHON" "$PROJECT/baselines/bench_gr00t_pt.py" \
  --checkpoint "$PTQAD_BASE" \
  --backbone "$GR00T_BACKBONE_MODEL" \
  --device cuda:0 --seed 7 --warmup 10 --samples 50 \
  --out "$PROJECT/results/baselines/gr00t_bf16_run01.json")
```

本机已在当前 `PTQAD_PYTHON` 下完成 `Gr00tPolicy` 的 CPU 导入检查。脚本用当前 package 和 checkpoint 加载完整 16/32/4 层模型，检查加载没有 missing/unexpected/mismatched keys，核对所有模型参数为 BF16。`--backbone` 保留本地路径中的 `nvidia/Cosmos-Reason2`，不解析成只有 snapshot hash 的路径；模型和 processor 都显式从该本地目录加载，不修改原 checkpoint。

计时入口是 `Gr00tPolicy.get_action`，包括 processor、视觉语言 backbone、动作头、动作传回 CPU、解码和接口检查。随机观察生成在计时外，模拟器和网络不在本基准内。当前 API 返回 `(action_dict, info)`，脚本逐键校验动作的形状与有限值。

## 2. 可选 π0.5 环境

本机实查环境为 Python **3.12.14**、`lerobot==0.6.1`、`torch==2.9.0+cu128`、`torchvision==0.24.0+cu128`、`transformers==5.5.4`。逐包版本共 70 项，见 [pi05-baseline-py312.txt](../setup/locks/pi05-baseline-py312.txt)；安装来源和检查范围见 [pi05-baseline-environment.json](../setup/locks/pi05-baseline-environment.json)。该环境与 GR00T 的 transformers 4.57.3 分开保存。

LeRobot 的安装记录为 `uv` 安装的 0.6.1 distribution，没有 `direct_url.json` 或 VCS commit。实查其 RECORD 列出的 529 个带哈希文件全部一致；因此这里能确认安装版本及文件身份，不能把某个 Git commit 当作已证实的安装来源。

下面是为新环境准备的命令，**尚未在空白环境执行验证**；它们不会覆盖已有基准环境。完整固定依赖列表使用 `--no-deps`，避免安装器另行解析成其他版本。

```bash
cd /absolute/path/to/fp4vla
export PROJECT=$(pwd)
export PI05_ENV="$PROJECT/.venv-pi05-baseline"
test ! -e "$PI05_ENV"
uv python install 3.12.14
uv venv --python 3.12.14 "$PI05_ENV"
uv pip install --python "$PI05_ENV/bin/python" --no-deps --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r setup/locks/pi05-baseline-py312.txt

CUDA_VISIBLE_DEVICES= "$PI05_ENV/bin/python" -c \
  'from lerobot.policies.pi05 import PI05Policy; print(PI05Policy.__module__)'
```

实查本机安装环境的 CPU package 导入和 checkpoint config 解析已通过；本轮未安装新环境、未在本任务内运行 GPU 基准。权重和 `paligemma_tokenizer.model` 由 `setup/03_download_weights.sh` 下载步骤准备。

## 3. π0.5：显式配置 BF16 后运行

```bash
"$PI05_ENV/bin/python" "$PROJECT/baselines/bench_pi05_lerobot.py" \
  --checkpoint "$PROJECT/weights/pi05_libero_base" \
  --device cuda:0 --seed 7 --warmup 10 --samples 30 \
  --out "$PROJECT/results/baselines/pi05_lerobot_bf16_run01.json"
```

脚本显式将 LeRobot config 设为 `dtype=bfloat16`，先在 CPU 构造并严格加载本地权重，再移到指定 CUDA 设备。LeRobot 0.6.1 的便捷 `from_pretrained` 会捕获加载异常；本入口复用该版本的官方 key remapper 后直接调用 `load_state_dict(strict=True)`，加载不完整就退出，不产生成功结果文件。

这里遵循 LeRobot 原生 BF16 配置：**视觉路径、projector、归一化及 action/time heads 仍保留 FP32**。脚本不会在最后对整模型无差别调用 `.bfloat16()`；JSON 记录实际 BF16/FP32 参数元素数，并在第一次未计时前向中观察 Linear 的输入、权重和输出 dtype。不能把这个配置描述成全模型纯 BF16。

计时入口是 `PI05Policy.predict_action_chunk`，包括其内部图像 resize/normalize、模型前向与动作去填充；输入随机数生成、SentencePiece 分词和初始 host-to-device 传输均在计时外。两张合成图像输入为 256×256，缺失的第三相机由 LeRobot 填充并设为无效 mask。动作按完整 chunk 推理，不能用有动作队列缓存的 `select_action` 代替。

这个参考输入采用 `BOS + 提示词 + EOS`。`predict_action_chunk` 不直接使用 batch 中的状态向量，真实环境 processor 的 state-to-text、状态归一化和动作反归一化不在本脚本里。输出记录 token 数与该边界；它是合成输入的参考入口耗时，不能冒充完整机器人控制周期，也不能在未对齐 token、相机和预处理后与 APXInf 接口宣称等价输入加速比。

## 4. 结果验收

两个脚本都强制新 `--out` 文件，使用 CUDA 同步包围被测调用。先进行一次带 dtype 观察 hook 的未计时前向，再完成指定 warmup；计时阶段没有这些 hook。逐次有限值检查在停止计时后进行，失败立即退出。

JSON `schema_version=2` 包含 `dtype_config`、`dtype_policy`、`parameter_dtype_elements`、`compute_dtype.linear_io_signatures`、`timing_scope`、输入/源码哈希、随机种子、warmup 和样本量。`lat_all_ms` 保留实际执行顺序和未取整毫秒数，p50/p99 使用 NumPy 的线性插值百分位算法。PyTorch 显存指标在 warmup 后重置峰值，包含模型常驻与被测调用的分配，不包括其他进程或桌面显存。当前脚本不把无法采集的功耗填写成零。

本轮 CPU 校验覆盖 tuple/有限值拒收、dtype hook 清理、BF16 config 覆盖及严格加载错误传播。完整 CUDA 加载与实际性能仍以随后新生成的 JSON 和运行日志为准。
