### 安装步骤

以下安装步骤以 x86_64 WSL2 Ubuntu 为目标。先在 Windows 安装支持该 Blackwell GPU 的 NVIDIA 驱动，并确认 WSL 中 `nvidia-smi` 可见 GPU；WSL 驱动按 [NVIDIA 官方指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)配置。恢复链需要系统 Python、Git LFS、curl、C/C++ 构建工具、EGL/OpenGL 和已安装的 conda。这里只提供锁定环境的重建步骤；本次证据没有覆盖一台空白系统的完整安装。

先安装 Ubuntu 通用依赖，再获取仓库：

```bash
sudo apt-get update
sudo apt-get install -y \
  ca-certificates curl git git-lfs bzip2 unzip \
  python3 python3-venv python3-dev \
  build-essential pkg-config cmake ninja-build libssl-dev \
  libegl1 libgl1 libglx0 libglvnd0 libosmesa6 libglfw3 \
  libx11-6 libxext6 libxrender1
git lfs install
nvidia-smi

git clone https://github.com/zhaosiying12138/apxinf-gr00t-fp4fp8-ptqad.git
cd apxinf-gr00t-fp4fp8-ptqad
```

若已有 conda，直接把 `CONDA_EXE` 指向其可执行文件。否则可选用 [Miniforge 官方 release](https://github.com/conda-forge/miniforge/releases)安装：下面先解析一次 release 标签，再从同一 release 下载 x86_64 安装器和 SHA-256 文件，校验通过才执行。保留打印的版本与校验文件；它只用于创建媒体环境，FFmpeg 等实际包仍由仓库的 explicit lock 固定。

```bash
(
set -euo pipefail
test ! -e "$HOME/miniforge3" || { echo 'Miniforge prefix already exists'; exit 1; }
MINIFORGE_STAGE="$(mktemp -d -t fp4vla-miniforge-XXXXXX)"
MINIFORGE_RELEASE="$(curl -fLsS --retry 3 \
  https://api.github.com/repos/conda-forge/miniforge/releases/latest | \
  python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_RELEASE"
cd "$MINIFORGE_STAGE"
curl -fL --retry 3 "$MINIFORGE_URL/Miniforge3-Linux-x86_64.sh" \
  -o Miniforge3-Linux-x86_64.sh
curl -fL --retry 3 "$MINIFORGE_URL/Miniforge3-Linux-x86_64.sh.sha256" \
  -o Miniforge3-Linux-x86_64.sh.sha256
sha256sum --check Miniforge3-Linux-x86_64.sh.sha256
printf 'Miniforge release: %s; installer evidence: %s\n' "$MINIFORGE_RELEASE" "$MINIFORGE_STAGE"
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
)
export CONDA_EXE="$HOME/miniforge3/bin/conda"
"$CONDA_EXE" --version
```

接下来从仓库根目录执行。大模型权重、Hessian、训练 checkpoint 和虚拟环境均不进入 Git：

```bash
export CONDA_EXE="${CONDA_EXE:-$HOME/miniforge3/bin/conda}"
test -x "$CONDA_EXE" || { echo 'Set CONDA_EXE to an installed conda'; exit 1; }
bash setup/01_install_dev_tools.sh
export PATH="$HOME/.local/bin:$PATH"
bash setup/03_download_weights.sh --core-only
CONDA_EXE="$CONDA_EXE" bash setup/06_install_recovery.sh
source setup/recovery-env.sh
```

`setup/06_install_recovery.sh` 会准备固定 revision 的 GR00T、训练环境 `.venv`、仿真环境 `.venv-libero`、媒体库和恢复补丁，并生成 `setup/recovery-env.sh`。脚本需要已安装的 conda；若其路径不同，修改 `CONDA_EXE`。安装后先检查环境和权重：

```bash
python3 setup/verify_weights.py --models gr00t cosmos
```

仅复现 Torch QDQ 恢复与闭环不需要编译 APXInf。需要原生引擎、CUDA 算子或编译截图时，另外安装包含 `nvcc` 和 cuBLASLt、支持 `sm_120` 的 CUDA toolkit，以及 Rust/Cargo。CUDA 按 [官方 WSL toolkit 安装说明](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2)选择 toolkit-only 安装；WSL 不安装 Linux 显示驱动。Rust 按 [官方 rustup 安装说明](https://www.rust-lang.org/tools/install)安装。参考编译截图使用 `nvcc 13.3.73`；PyTorch wheel 的 cu128 runtime 与系统 toolkit 分别核验，不要求字符串相同。

下面只检查已经安装的原生工具链，检查成功后才启动构建：

```bash
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
(
set -euo pipefail
nvcc --version
nvcc --list-gpu-code | grep -Fx sm_120
rustc --version
cargo --version
c++ --version
make --version
bash setup/05_restore_all.sh --with-engine
)
source setup/recovery-env.sh
```

`setup/05_restore_all.sh --with-engine` 会克隆固定 revision 的 APXInf-robo 并调用引擎构建脚本；已有完整权重时可追加 `--skip-download`。不要把本机 Hugging Face snapshot 直接改名成不含 `nvidia/Cosmos-Reason2` 的路径，GR00T 工厂会用该字符串选择 backbone。

该封装入口会依次调用 `setup/01_install_dev_tools.sh`、`setup/03_download_weights.sh`、`setup/06_install_recovery.sh`，并在 `--with-engine` 时调用 `setup/02_build_engine.sh` 和 `setup/00_env_report.sh`。`setup/06_install_recovery.sh` 会按锁文件创建训练环境与独立的 `.venv-libero`；因此本项目不再单独运行旧的 `setup/04_install_libero.sh`，避免把 LIBERO 依赖装入 APXInf/训练环境。

