# Windows / WSL Ubuntu 运行截图

本目录提供论文使用的终端采集工具，无需安装个人 Codex 技能。七个脚本从 `wsl-ubuntu-screenshot` 的 `scripts/` 逐字节复制，来源、长度和 SHA-256 见 [source_manifest.json](source_manifest.json)；没有修改行为，也未包含旧的提示符模拟脚本或实验样张。目录内的 `.gitattributes` 禁用这些脚本的自动换行转换，使 Windows 和 Linux checkout 保持相同字节与哈希。

## 环境要求

- Windows、Windows PowerShell 5.1、Microsoft Store 稳定版 Windows Terminal，以及名为 `Ubuntu` 的 WSL 发行版。`wt.exe`、`wsl.exe` 应可直接调用。
- 当前脚本使用 `%LOCALAPPDATA%\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json`。先正常打开一次 Terminal；Preview/便携版或其他发行版名称不属于这份原样脚本的配置范围。
- 已解锁的 `Default` 交互桌面、主屏幕实际分辨率 **3840×2400**。本论文确认的成品为去除任务栏后的 **3840×2280**；脚本不缩放图片，裁剪后的尺寸必须实际核验。
- WSL 的 Python 3，以及 Pillow、NumPy。可在独立虚拟环境安装 `python -m pip install Pillow numpy`，下方裁剪命令使用该环境的解释器。
- 真实执行脚本位于 WSL，使用 `set -euo pipefail`；输出指向新的 scratch 目录。截图期间 GPU 作业串行执行，桌面保持可见。

在 WSL 创建下文使用的截图环境：

```bash
python3 -m venv "$HOME/.venvs/fp4vla-capture"
"$HOME/.venvs/fp4vla-capture/bin/python" -m pip install Pillow numpy
```

## 采集与裁剪

先拍一次低成本真实命令的样张，人工确认分辨率、紫色主题和可读性后再批量采集。Bash 脚本只放要执行的真实命令；采集器会用 `bash -x` 显示命令并保存原始输出，不需要自行添加结束标记或等待逻辑。

在 Windows PowerShell 中，将以下路径换成本机位置。把整个工具目录复制到 Windows 本地盘后运行，七个脚本保持在同一目录：

```powershell
$Repo = '\\wsl.localhost\Ubuntu\home\<linux-user>\codebase\apxinf-gr00t-fp4fp8-ptqad'
$CaptureTools = Join-Path $env:LOCALAPPDATA ('fp4vla-capture-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
Copy-Item -LiteralPath (Join-Path $Repo 'setup\windows_capture') -Destination $CaptureTools -Recurse
$OutPng = Join-Path $env:USERPROFILE 'shot\new-run\sample.png'
$BashScript = '/home/<linux-user>/shot_scripts/sample.sh'
$CapturePython = '/home/<linux-user>/.venvs/fp4vla-capture/bin/python'

powershell.exe -ExecutionPolicy Bypass -File (Join-Path $CaptureTools 'capture_session.ps1') `
  -BashScript $BashScript -OutPng $OutPng -TimeoutSec 900
if ($LASTEXITCODE -ne 0) { throw 'Capture failed; inspect the raw log and failure record.' }

$CaptureToolsWsl = (& wsl.exe -d Ubuntu -- wslpath -a $CaptureTools.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot convert capture tool path.' }
$OutWsl = (& wsl.exe -d Ubuntu -- wslpath -a $OutPng.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot convert image path.' }
wsl.exe -d Ubuntu -- $CapturePython "$CaptureToolsWsl/crop_taskbar.py" $OutWsl
if ($LASTEXITCODE -ne 0) { throw 'Taskbar crop failed.' }
```

`crop_taskbar.py` 按底部行亮度识别任务栏，并原位替换 PNG；它只裁掉底部，不修图、不缩放。不同桌面布局不保证得到 2280 像素高，须检查实际图像和裁剪记录。成功运行留下：

| 文件 | 内容 |
|---|---|
| `sample.png` | 原始屏幕截取后仅裁掉任务栏的图像 |
| `sample.log` | Bash 命令与未过滤的标准输出、错误输出 |
| `sample.json` | 退出状态、截图时间、裁剪前哈希与黑屏检查 |
| `sample.png.window-binding.json` | 本次窗口的标题 token、HWND/PID、前台与光标记录 |
| `sample.crop.json` | 裁剪前后哈希、尺寸及裁剪矩形 |

保存这些文件及实际 Bash 脚本；脚本引用的 `common_*.sh` 等辅助文件也应保留。每次使用新文件名，采集器拒绝覆盖已有证据。查看完整 PNG，核对真实命令、最后输出、错误信息、弹窗和日志中的数字，再考虑登记。

## Terminal 配置的备份与恢复

`capture_session.ps1` 为每次运行建立独立的 `%TEMP%\wsl-shot-settings-<token>.json`，临时加入 `UbuntuPurpleShot` 紫色配置，最后通过 `finally` 恢复原设置。正常使用只调用此入口，不额外运行主题开关；已有 `.bak_shot` 不会被覆盖。

停电或强制结束进程可能来不及执行 `finally`。保留输出中显示的备份路径，确认对应本次会话后恢复：

```powershell
powershell.exe -ExecutionPolicy Bypass -File (Join-Path $CaptureTools 'wt_purple_off.ps1') `
  -BackupPath 'C:\Users\<windows-user>\AppData\Local\Temp\wsl-shot-settings-<token>.json'
```

恢复成功后脚本删除该备份。采集时工具还会最大化并聚焦本次 Terminal、暂移光标，并临时隐藏限定窗口类型的输入法/ToDesk 浮条，随后尝试恢复；不会停止远程桌面进程。启动前应自行确认桌面已解锁。窗口绑定与黑屏检查不等于人工内容验收；命令成功也不保证截图有效，截图失败后先检查图像与日志，不自动重跑昂贵实验。

## 独立复现的样张与清单

仓库中的 `paper/evidence/captures.json` 绑定作者已批准样张的哈希。原样张属于作者重建发布证据的输入，本目录不附带该旧实验画面。读者可直接查看仓库已发布截图；重新采集时，使用自己明确批准的样张和**独立复现副本**，保留原发布清单。

[`paper/record_capture.py`](../../paper/record_capture.py) 会登记到它所在副本的 `paper/evidence/captures.json`，没有另选清单的参数。以下命令在 WSL 的原仓库根目录执行，创建独立 checkout；样张经人工明确批准后，才初始化新清单：

```bash
export CAPTURE_REPRO="$HOME/codebase/fp4vla-capture-repro"
export CAPTURE_SAMPLE="/mnt/c/Users/<windows-user>/shot/new-run/sample.png"
test ! -e "$CAPTURE_REPRO" || exit 1
git worktree add --detach "$CAPTURE_REPRO" HEAD
# 使用已安装 Pillow 的解释器；此处以独立截图环境为例。
"$HOME/.venvs/fp4vla-capture/bin/python" - "$CAPTURE_REPRO" "$CAPTURE_SAMPLE" <<'PY'
from datetime import datetime, timezone
import hashlib, json, pathlib, sys
from PIL import Image
repro, sample = (pathlib.Path(arg).resolve(strict=True) for arg in sys.argv[1:])
if repro == pathlib.Path.cwd().resolve():
    raise ValueError("Use a separate reproduction checkout")
with Image.open(sample) as image:
    dimensions = list(image.size)
if dimensions != [3840, 2280]:
    raise ValueError("Approve a 3840 x 2280 sample for this publication layout")
manifest = repro / "paper/evidence/captures.json"
original = repro / "paper/_build/capture-repro-original/captures.json"
if original.exists():
    raise FileExistsError("Independent manifest was already initialized")
original.parent.mkdir(parents=True, exist_ok=True)
original.write_bytes(manifest.read_bytes())
# Preserve the original proof files outside the new capture namespace.
# New batches can then keep their own common_v11.sh and plan.json unchanged.
proofs = repro / "paper/evidence/captures"
if proofs.is_dir():
    proofs.rename(original.parent / "captures")
proofs.mkdir(parents=True, exist_ok=True)
record = {"version": 1, "approved_sample": {
    "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
    "confirmed_by_user": True,
    "confirmed_on": datetime.now(timezone.utc).date().isoformat(),
    "dimensions": dimensions}, "screenshots": []}
manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(manifest)
PY
```

后续从 `$CAPTURE_REPRO` 执行 `paper/record_capture.py`，传入同一 `--approved-sample`、本次图像及原始日志，并在实际看图后使用 `--visually-verified`。该流程只改独立副本；新清单中的截图不能冒充原论文截图。构建完整新发布包还需补齐所有图位与匹配的实验记录。

## 验收范围

工具校验命令退出状态、窗口身份、前台可见性、非黑屏像素及图像裁剪链。人工仍需核验可读性、内容与来源。它不验证训练是否收敛，也不由终端图推断成功率或延迟；这些结论来自完整实验 JSON 和逐回合日志。

本次公开复制仅进行了七文件哈希比对、PowerShell/Python 语法解析和 Bash 语法检查，没有为复制过程执行实验或新增截图。
