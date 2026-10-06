### 6. 原生 APXInf 验收（可选）

仅验证独立 CUDA 程序的编译时，先复制源码到新的 scratch 目录，避免覆盖已记录性能的二进制。以下命令对应[编译补充截图](docs/images/ubuntu-compile.png)；实拍环境为 `nvcc 13.3.73`、`sm_120`，三个目标均编译成功，没有执行 GPU 基准。它是论文17图之外的编译补充图，不代表空白系统安装验收或新增性能结果。

```bash
(
set -euo pipefail
SPIKE_BUILD="$PROJECT/paper/_build/spike-compile-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$(dirname "$SPIKE_BUILD")"
mkdir "$SPIKE_BUILD"
cp "$PROJECT/spike/Makefile" "$PROJECT/spike/fp4_gemm_bench.cu" \
  "$PROJECT/spike/fp4_opbench.cu" "$PROJECT/spike/fp8_probe.cu" "$SPIKE_BUILD/"
make -C "$SPIKE_BUILD" -j1 ARCH=sm_120 \
  APXINF_FP4_ADAPTER="$PROJECT/third_party/apxinf-robo/apxinf/crates/apxinf-cuda/adapters/cublaslt_fp4_adapter.cu" \
  fp4_gemm_bench fp4_opbench fp8_probe
)
```

完成前面的 APXInf 安装后，按下面命令编译候选 wheel、准备 π0.5 packed 模型，再串行验收和计时。需要完整权重下载（不带 `--core-only`）；GPU 此时必须空闲。每个输出目录/文件必须是新的，失败时保留现场并改用新的运行名称。实现与计时边界见 [原生 π0.5 说明](docs/native-pi05.md)。

<details>
<summary>展开：原生编译、四道验收、GEMM/单层与三条策略基准的完整命令</summary>

```bash
(
set -euo pipefail
cd "$PROJECT"
export BENCH_RUN="$PROJECT/results/independent_bench_new_run"
export RUN="$PROJECT/weights/native_pi05_new_run"
mkdir "$BENCH_RUN"
bash setup/03_download_weights.sh
export APX="$PROJECT/third_party/apxinf-robo/apxinf"
export ENGINE_PY="$PROJECT/third_party/apxinf-robo/.venv/bin/python"
source "$PROJECT/third_party/apxinf-robo/.venv/bin/activate"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LIBRARY_PATH="$HOME/.cuda-stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export APXINF_CUDA_ARCH=sm_120 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=target/wheel

# 1. 编译；尚未执行模型。
(
  cd "$APX"
  maturin build --release --features cuda --auditwheel skip \
    -m crates/apxinf-py/Cargo.toml
  cargo build --release -p apxinf-model --features cuda \
    --example pi05_fp4_graph_smoke
) > "$BENCH_RUN/build_engine.log" 2>&1
make -C "$PROJECT/spike" fp4_gemm_bench fp8_probe fp4_opbench \
  > "$BENCH_RUN/build_spike.log" 2>&1

# 2. CPU 打包及模型 overlay；两个阶段各自拒绝已有输出。
bash exp/prepare_native_pi05.sh pack \
  --base "$PROJECT/weights/pi05_libero_base" --out "$RUN" --chunk-rows 512
bash exp/prepare_native_pi05.sh model-overlay \
  --base "$PROJECT/weights/pi05_libero_base" --out "$RUN"

# 3. 四道 GPU 验收；全部通过才由入口安装候选 wheel。
NATIVE_PI05_MODEL="$RUN/model" NATIVE_PY="$ENGINE_PY" PTQAD_GPU_EXCLUSIVE=1 \
  bash exp/run_native_graph_gates.sh "$BENCH_RUN/native_graph"

# 4. 独立算子验收与计时。
./spike/fp8_probe > "$BENCH_RUN/fp8_probe.log" 2>&1
./spike/fp4_gemm_bench --verify --slayout=hw \
  > "$BENCH_RUN/gemm_verify.stdout" 2> "$BENCH_RUN/gemm_verify.log"
./spike/fp4_gemm_bench --slayout=hw --iters=50 \
  > "$BENCH_RUN/gemm.csv" 2> "$BENCH_RUN/gemm.log"
./spike/fp4_opbench --verify-only \
  > "$BENCH_RUN/opbench_verify.csv" 2> "$BENCH_RUN/opbench_verify.log"
./spike/fp4_opbench --warmup=10 --samples=30 \
  > "$BENCH_RUN/opbench.csv" 2> "$BENCH_RUN/opbench.log"

# 5. 三个独立模型进程依次退出，避免同时驻留。
cd /tmp
"$ENGINE_PY" "$PROJECT/exp/bench_engine.py" \
  --model-dir "$RUN/model" --variant bf16 --device cuda:0 --telemetry-device 0 \
  --warmup 10 --samples 30 --seed 7 --model-seed 0 \
  --out "$BENCH_RUN/pi05_bf16.json"
"$ENGINE_PY" "$PROJECT/exp/bench_engine.py" \
  --model-dir "$RUN/model" --variant nvfp4_static --device cuda:0 --telemetry-device 0 \
  --warmup 10 --samples 30 --seed 7 --model-seed 0 \
  --out "$BENCH_RUN/pi05_nvfp4.json"
"$ENGINE_PY" "$PROJECT/exp/bench_engine.py" \
  --model-dir "$BASE" --variant bf16 --device cuda:0 --telemetry-device 0 \
  --extra-kwarg "backbone=$GR00T_BACKBONE_MODEL" \
  --warmup 10 --samples 30 --seed 7 --model-seed 0 \
  --out "$BENCH_RUN/gr00t_bf16.json"
)
```

`.cuda-stubs` 只用于链接阶段的 `LIBRARY_PATH`，不要放入运行时 `LD_LIBRARY_PATH`。`PTQAD_GPU_EXCLUSIVE=1` 是串行使用 GPU 的声明，不会抢占其他作业。候选 wheel、完整模型验收程序与单层基准须来自同一份已应用项目补丁的 APXInf 源码。

</details>

### 7. PyTorch 参考基准（可选）

GR00T 复用恢复环境；π0.5 另用锁定的 LeRobot 环境。以下安装模板尚未在空白环境重新执行验证，锁文件记录的是实际测量环境；依赖来源与加载契约见 [PyTorch 基准说明](docs/baseline-timing.md)。同样逐条串行运行：

<details>
<summary>展开：两个 PyTorch 参考入口与 π0.5 独立依赖安装</summary>

```bash
(
set -euo pipefail
cd "$PROJECT"
source setup/recovery-env.sh
BASELINE_RUN="$PROJECT/results/pytorch_baseline_new_run"
mkdir "$BASELINE_RUN"
bash setup/03_download_weights.sh
(
  cd "$GR00T_REPO"
  "$PTQAD_PYTHON" "$PROJECT/baselines/bench_gr00t_pt.py" \
    --checkpoint "$BASE" --backbone "$GR00T_BACKBONE_MODEL" \
    --device cuda:0 --seed 7 --warmup 10 --samples 50 \
    --out "$BASELINE_RUN/gr00t_bf16.json"
)

PI05_ENV="$PROJECT/.venv-pi05-baseline"
test ! -e "$PI05_ENV"
uv python install 3.12.14
uv venv --python 3.12.14 "$PI05_ENV"
uv pip install --python "$PI05_ENV/bin/python" --no-deps --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r setup/locks/pi05-baseline-py312.txt
CUDA_VISIBLE_DEVICES= "$PI05_ENV/bin/python" -c \
  'from lerobot.policies.pi05 import PI05Policy; print(PI05Policy.__module__)'
"$PI05_ENV/bin/python" "$PROJECT/baselines/bench_pi05_lerobot.py" \
  --checkpoint "$PROJECT/weights/pi05_libero_base" \
  --device cuda:0 --seed 7 --warmup 10 --samples 30 \
  --out "$BASELINE_RUN/pi05_bf16.json"
)
```

</details>

