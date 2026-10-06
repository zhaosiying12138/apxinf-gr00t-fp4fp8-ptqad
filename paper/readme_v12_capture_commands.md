按生成的 `plan.json` 顺序逐张执行：`shot_bake`、`shot_collect`、`shot_qad`、`shot_rollout`、`shot_opdcache`、`shot_opd`、`shot_evalserver`。其中前两张只做 CPU 来源核验；QAD/OPD 截图分别执行 2/4 次短训练，正式 2,000 步结果由完整日志证明。

在 Windows PowerShell 中，设置实际 WSL 项目路径。下面使用作者运行位置；独立复现时替换为自己的路径。每次只改变 `$Figure`，执行并查看一张图片后再进行下一张。

```powershell
$WslProject = '/home/zhaosiying/codebase/fp4vla'
$Repo = '\\wsl.localhost\Ubuntu' + $WslProject.Replace('/','\')
$WslCaptureRoot = "$WslProject/results/reruns/rtn_w4a4_release_20261006_01/captures/w4a4_final"
$CaptureTools = Join-Path $env:LOCALAPPDATA 'fp4vla-capture-v12'
if (-not (Test-Path $CaptureTools)) {
  Copy-Item -LiteralPath (Join-Path $Repo 'setup\windows_capture') -Destination $CaptureTools -Recurse
}
$OutDir = Join-Path $env:USERPROFILE 'shot\ptqad_v12_final'
$CapturePython = '/home/zhaosiying/.venvs/fp4vla-capture/bin/python'
$Figure = 'shot_bake'
# v12 后缀使窗口记录与先前批次的文件名不会冲突。
$OutPng = Join-Path $OutDir ($Figure + '_v12.png')
powershell.exe -ExecutionPolicy Bypass -File (Join-Path $CaptureTools 'capture_session.ps1') `
  -BashScript "$WslCaptureRoot/$Figure.sh" -OutPng $OutPng -TimeoutSec 900
if ($LASTEXITCODE -ne 0) { throw 'Capture failed; inspect the raw log before retrying.' }
$ToolsWsl = (& wsl.exe -d Ubuntu -- wslpath -a $CaptureTools.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve capture tools.' }
$ImageWsl = (& wsl.exe -d Ubuntu -- wslpath -a $OutPng.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve captured image.' }
wsl.exe -d Ubuntu -- $CapturePython "$ToolsWsl/crop_taskbar.py" $ImageWsl
if ($LASTEXITCODE -ne 0) { throw 'Crop failed.' }
```

裁剪解释器的安装、桌面条件和独立样张清单见 [截图工具说明](setup/windows_capture/README.md)。作者沿用已批准的样张；其他机器应先确认自己的样张，并在独立复现副本中登记。检查命令、末尾输出、紫色主题、尺寸和遮挡后，才运行下面的登记命令。`CAPTURE_WINDOWS_DIR` 与 PowerShell 的 `$OutDir` 对应；`APPROVED_SAMPLE_PNG` 必须指向清单中已经批准的样张。

```bash
export CAPTURE_WINDOWS_DIR="/mnt/c/Users/Admin1/shot/ptqad_v12_final"
export APPROVED_SAMPLE_PNG="/mnt/c/Users/Admin1/shot/ptqad_20260929/sample_verified.png"
FIGURE=shot_bake
# 保存原始 plan 字节，并使用本批次独立名称。
if test ! -e "$CAPTURE_ROOT/plan_v12.json"; then
  cp "$CAPTURE_ROOT/plan.json" "$CAPTURE_ROOT/plan_v12.json"
fi
cmp "$CAPTURE_ROOT/plan.json" "$CAPTURE_ROOT/plan_v12.json"
python3 paper/record_capture.py \
  --figure "$FIGURE" \
  --image "$CAPTURE_WINDOWS_DIR/${FIGURE}_v12.png" \
  --log "$CAPTURE_WINDOWS_DIR/${FIGURE}_v12.log" \
  --script "$CAPTURE_ROOT/$FIGURE.sh" \
  --sidecar "$CAPTURE_WINDOWS_DIR/${FIGURE}_v12.json" \
  --crop-manifest "$CAPTURE_WINDOWS_DIR/${FIGURE}_v12.crop.json" \
  --support-file "$CAPTURE_ROOT/common_v12.sh" \
  --support-file "$CAPTURE_ROOT/plan_v12.json" \
  --support-file "$CAPTURE_WINDOWS_DIR/${FIGURE}_v12.png.window-binding.json" \
  --approved-sample "$APPROVED_SAMPLE_PNG" \
  --visually-verified
```

每次登记都校验截图与裁剪链，最后的发布校验还会核对 17 张图和本轮协议。截图短跑的分数和耗时不进入正式结果表。
