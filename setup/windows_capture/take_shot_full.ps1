param([string]$BashScript, [string]$OutPng, [int]$TimeoutSec = 900, [string]$MarkerPath = "$env:TEMP\shot_done")
$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath $OutPng) { throw "Refusing existing capture output: $OutPng" }
. (Join-Path $PSScriptRoot 'capture_validation.ps1')
$marker = $MarkerPath
Remove-Item -LiteralPath $marker -ErrorAction SilentlyContinue

if (-not ('WinShotV2' -as [type])) { Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;
public class WinShotV2 {
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int x,y; }
  [DllImport("kernel32.dll", SetLastError=true)] public static extern uint SetThreadExecutionState(uint flags);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll", SetLastError=true)] public static extern bool GetCursorPos(out POINT point);
  [DllImport("user32.dll", SetLastError=true)] public static extern bool SetCursorPos(int x,int y);
  public delegate bool EnumProc(IntPtr h, IntPtr p);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc fn, IntPtr p);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  public static IntPtr[] HideImeStatus() {
    var hidden = new List<IntPtr>();
    EnumWindows((h,p) => { var name = new StringBuilder(128); GetClassName(h,name,128);
      if (IsWindowVisible(h) && name.ToString()=="SoPY_Status") { hidden.Add(h); ShowWindow(h,0); }
      return true;
    }, IntPtr.Zero);
    return hidden.ToArray();
  }
}
"@
}
[WinShotV2]::SetProcessDPIAware() | Out-Null

# Bind to the new top-level Terminal HWND with this launch's unique title token.
if (-not ('WinShotBindingDraftV1' -as [type])) { Add-Type @"
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public class TerminalWindowEvidence {
  public long hwnd; public uint pid; public uint thread;
  public string owner; public string windowClass; public string title;
  public bool exists; public bool visible; public bool minimized;
  public int cloaked; public int left; public int top; public int right; public int bottom;
}
public static class WinShotBindingDraftV1 {
  public delegate bool EnumProc(IntPtr h, IntPtr p);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int left,top,right,bottom; }
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc fn, IntPtr p);
  [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT rect);
  [DllImport("user32.dll",CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr h,StringBuilder s,int n);
  [DllImport("user32.dll",CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h,StringBuilder s,int n);
  [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr h,int attribute,out int value,int size);
  public static TerminalWindowEvidence Describe(IntPtr h) {
    var value=new TerminalWindowEvidence(); value.hwnd=h.ToInt64(); value.exists=IsWindow(h);
    uint pid; value.thread=GetWindowThreadProcessId(h,out pid); value.pid=pid;
    var cls=new StringBuilder(256); GetClassName(h,cls,256); value.windowClass=cls.ToString();
    var title=new StringBuilder(1024); GetWindowText(h,title,1024); value.title=title.ToString();
    try { using(var proc=System.Diagnostics.Process.GetProcessById((int)pid)) { value.owner=proc.ProcessName; } }
    catch { value.owner="<unavailable>"; }
    value.visible=IsWindowVisible(h); value.minimized=IsIconic(h);
    int cloaked; value.cloaked=DwmGetWindowAttribute(h,14,out cloaked,4)==0 ? cloaked : -1;
    RECT rect; if(GetWindowRect(h,out rect)) { value.left=rect.left; value.top=rect.top; value.right=rect.right; value.bottom=rect.bottom; }
    return value;
  }
  public static TerminalWindowEvidence[] VisibleToDeskToolbars() {
    var windows=new List<TerminalWindowEvidence>();
    EnumWindows((h,p) => {
      var info=Describe(h);
      if(info.visible && info.cloaked==0 && info.windowClass=="H-SMILE-FRAME" &&
         info.owner=="ToDesk" && info.title=="") windows.Add(info);
      return true;
    },IntPtr.Zero);
    return windows.ToArray();
  }
  public static TerminalWindowEvidence[] TerminalWindows() {
    var windows=new List<TerminalWindowEvidence>();
    EnumWindows((h,p) => {
      var cls=new StringBuilder(256); GetClassName(h,cls,256);
      if(cls.ToString()=="CASCADIA_HOSTING_WINDOW_CLASS") {
        var info=Describe(h);
        if(String.Equals(info.owner,"WindowsTerminal",StringComparison.OrdinalIgnoreCase)) windows.Add(info);
      }
      return true;
    },IntPtr.Zero);
    return windows.ToArray();
  }
}
"@ }
$bindingPath = $OutPng + '.window-binding.json'
if (Test-Path -LiteralPath $bindingPath) { throw "Refusing existing HWND evidence: $bindingPath" }
$windowTitleToken = 'FP4VLA-' + [guid]::NewGuid().ToString('N')
$binding = [ordered]@{ status='not_started'; window_title_token=$windowTitleToken; before=@(); polls=@(); selected=$null }
function Save-WindowBindingEvidence {
  [IO.File]::WriteAllText($bindingPath, ($binding | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
}


# This is a temporary requirement of the current capture thread only. It does
# not change power/lock policies, prevent manual sleep, or bypass a lock screen.
$previousExecutionState = [WinShotV2]::SetThreadExecutionState([uint32]2147483651) # CONTINUOUS | DISPLAY | SYSTEM
if ($previousExecutionState -eq 0) { throw 'SetThreadExecutionState failed; capture not started.' }
try {
# Snapshot every existing Terminal top-level HWND, including minimized/cloaked ones.
$binding.before = @([WinShotBindingDraftV1]::TerminalWindows())
$beforeHandles = @($binding.before | ForEach-Object { $_.hwnd })
$binding.status = 'launching'
Save-WindowBindingEvidence
$proc = Start-Process -FilePath "wt.exe" -PassThru -ArgumentList @(
    "-w", "new", "new-tab", "--profile", "UbuntuPurpleShot", "--title", $windowTitleToken,
    "--suppressApplicationTitle", "wsl.exe", "-d", "Ubuntu", "--", "bash", $BashScript)
$binding.launcher_pid = $proc.Id  # wt.exe may be a short-lived broker; never use it as the HWND owner.
$windowWait = [Diagnostics.Stopwatch]::StartNew()
$selected = $null
$stableKey = $null
$stableCount = 0
while ($windowWait.Elapsed.TotalSeconds -lt 12) {
  $after = @([WinShotBindingDraftV1]::TerminalWindows())
  $added = @($after | Where-Object { $beforeHandles -notcontains $_.hwnd })
  $matching = @($added | Where-Object { $_.title -ceq $windowTitleToken })
  $binding.polls += [ordered]@{ elapsed_ms=$windowWait.ElapsedMilliseconds; added=$added; matching=$matching }
  if ($matching.Count -gt 1) { throw 'Multiple new Terminal HWNDs with this title token; refusing ambiguous binding.' }
  if ($matching.Count -eq 1 -and $matching[0].exists -and $matching[0].visible -and $matching[0].cloaked -eq 0) {
    $key = '{0}:{1}:{2}' -f $matching[0].hwnd,$matching[0].pid,$matching[0].thread
    if ($stableKey -eq $key) { $stableCount++ } else { $stableKey=$key; $stableCount=1 }
    if ($stableCount -ge 2) { $selected=$matching[0]; break }
  } else { $stableKey=$null; $stableCount=0 }
  Start-Sleep -Milliseconds 200
}
if (-not $selected) { throw 'No unique visible uncloaked new Terminal HWND with this title token; no MainWindowHandle fallback.' }
$binding.selected = $selected
$binding.status = 'bound'
Save-WindowBindingEvidence
$h = [IntPtr]$selected.hwnd
[WinShotV2]::ShowWindow($h, 9) | Out-Null    # SW_RESTORE
$binding.foreground_at_launch_return = [WinShotV2]::SetForegroundWindow($h)
Start-Sleep -Milliseconds 400
[WinShotV2]::ShowWindow($h, 3) | Out-Null    # SW_MAXIMIZE
Start-Sleep -Milliseconds 600

$sw = [Diagnostics.Stopwatch]::StartNew()
while (-not (Test-Path $marker) -and $sw.Elapsed.TotalSeconds -lt $TimeoutSec) { Start-Sleep -Milliseconds 500 }
if (-not (Test-Path -LiteralPath $marker)) { throw "TIMEOUT: no completion marker for $BashScript; no success screenshot saved." }
$runStatus = (Get-Content -Raw -LiteralPath $marker).Trim()
if ($runStatus -and $runStatus -ne "0") { throw "Command failed with exit status $runStatus; inspect its raw log before retrying." }
Start-Sleep -Milliseconds 1500

# keep the terminal foreground after the wait, then grab the WHOLE screen (PrtSc)
# Keep the same launch-bound HWND throughout. Never reselect by process age.
$current = [WinShotBindingDraftV1]::Describe($h)
if (-not $current.exists -or $current.pid -ne $selected.pid -or $current.thread -ne $selected.thread -or
    $current.windowClass -ne 'CASCADIA_HOSTING_WINDOW_CLASS' -or $current.owner -ne 'WindowsTerminal' -or
    $current.title -cne $windowTitleToken) {
  throw 'The launch-bound Terminal window disappeared or changed owner; no fallback allowed.'
}
$binding.foreground_before_capture_return = [WinShotV2]::SetForegroundWindow($h)
[WinShotV2]::ShowWindow($h, 3) | Out-Null
Start-Sleep -Milliseconds 500
$binding.capture_window = [WinShotBindingDraftV1]::Describe($h)
$binding.capture_foreground = [WinShotBindingDraftV1]::Describe([WinShotV2]::GetForegroundWindow())
$current = $binding.capture_window
if (-not $current.exists -or -not $current.visible -or $current.minimized -or $current.cloaked -ne 0 -or
    $current.pid -ne $selected.pid -or $current.thread -ne $selected.thread -or
    $current.windowClass -ne 'CASCADIA_HOSTING_WINDOW_CLASS' -or $current.owner -ne 'WindowsTerminal' -or
    $current.title -cne $windowTitleToken -or
    ($current.right-$current.left) -lt 800 -or ($current.bottom-$current.top) -lt 600 -or
    $binding.capture_foreground.hwnd -ne $selected.hwnd -or
    $binding.capture_foreground.pid -ne $selected.pid -or $binding.capture_foreground.thread -ne $selected.thread -or
    $binding.capture_foreground.owner -ne 'WindowsTerminal' -or
    $binding.capture_foreground.windowClass -ne 'CASCADIA_HOSTING_WINDOW_CLASS' -or
    $binding.capture_foreground.title -cne $windowTitleToken) {
  throw 'The bound Terminal is not a usable foreground window; no screenshot accepted.'
}
Save-WindowBindingEvidence
$hiddenImeWindows = @()
$hiddenRemoteToolbars = @()
$bmp = $null
$g = $null
$savedCursor = [WinShotV2+POINT]::new()
$parkedCursor = [WinShotV2+POINT]::new()
$cursorWasMoved = $false
try {
    Add-Type -AssemblyName System.Windows.Forms
    $captureBounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    # Only the two observed right-edge ToDesk geometries; never its process or session.
    $toolbarCandidates = @([WinShotBindingDraftV1]::VisibleToDeskToolbars() | Where-Object {
        $toolbarWidth = $_.right - $_.left
        $toolbarHeight = $_.bottom - $_.top
        $collapsed = $_.left -ge ($captureBounds.Right - 120) -and $_.right -le $captureBounds.Right -and
            $toolbarWidth -gt 0 -and $toolbarWidth -le 120 -and $toolbarHeight -gt 0 -and $toolbarHeight -le 200
        $expanded = [Math]::Abs($_.right - $captureBounds.Right) -le 16 -and
            $toolbarWidth -gt 0 -and $toolbarWidth -le 1100 -and $toolbarHeight -gt 0 -and $toolbarHeight -le 700 -and
            $_.top -ge ($captureBounds.Top + $captureBounds.Height / 2) -and $_.bottom -le ($captureBounds.Bottom + 16)
        $collapsed -or $expanded
    })
    if ($toolbarCandidates.Count -gt 1) { throw 'Ambiguous remote toolbar; refusing capture.' }
    foreach ($toolbar in $toolbarCandidates) {
        $hiddenRemoteToolbars += $toolbar
        [WinShotV2]::ShowWindow([IntPtr]$toolbar.hwnd, 0) | Out-Null
    }
    $binding.temporarily_hidden_remote_toolbars = $hiddenRemoteToolbars
    $hiddenImeWindows = [WinShotV2]::HideImeStatus()
    Start-Sleep -Milliseconds 500

    Add-Type -AssemblyName System.Drawing
    Add-Type -AssemblyName System.Windows.Forms
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    # Move away from title-bar controls so a hover tooltip cannot cover the terminal.
    if (-not [WinShotV2]::GetCursorPos([ref]$savedCursor)) { throw 'Cannot save cursor position; capture stopped.' }
    $parkedCursor.x = $bounds.Left + 200
    $parkedCursor.y = $bounds.Bottom - 150
    if ($parkedCursor.x -le ($current.left + 40) -or $parkedCursor.x -ge ($current.right - 40) -or
        $parkedCursor.y -le ($current.top + 120) -or $parkedCursor.y -ge ($current.bottom - 20)) {
        throw 'Cursor parking point is outside the expected terminal content area; capture stopped.'
    }
    if (-not [WinShotV2]::SetCursorPos($parkedCursor.x,$parkedCursor.y)) { throw 'Cannot park cursor; capture stopped.' }
    $cursorWasMoved = $true
    $binding.cursor_before = [ordered]@{ x=$savedCursor.x; y=$savedCursor.y }
    $binding.cursor_parked = [ordered]@{ x=$parkedCursor.x; y=$parkedCursor.y }
    Start-Sleep -Milliseconds 500
    $g.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
    $quality = Get-WslShotImageQuality -Bitmap $bmp
    if (-not $quality.accepted_not_black) {
        $rejected = $OutPng + '.rejected-' + [guid]::NewGuid().ToString('N') + '.png'
        $bmp.Save($rejected, [System.Drawing.Imaging.ImageFormat]::Png)
        throw "BLACK_CAPTURE_REJECTED: diagnostic image $rejected; near-black fraction=$($quality.near_black_fraction), mean luminance=$($quality.mean_luminance). No accepted image saved."
    }
    $bmp.Save($OutPng, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
    # An earlier cleanup error must not skip restoration of a temporarily hidden toolbar.
    try {
        if ($g) { $g.Dispose() }
    } finally {
        try {
            if ($bmp) { $bmp.Dispose() }
        } finally {
            try {
                foreach ($imeWindow in $hiddenImeWindows) { [WinShotV2]::ShowWindow($imeWindow,8) | Out-Null }
            } finally {
                try {
                    foreach ($toolbar in $hiddenRemoteToolbars) {
                        $toolbarNow = [WinShotBindingDraftV1]::Describe([IntPtr]$toolbar.hwnd)
                        if ($toolbarNow.exists -and $toolbarNow.pid -eq $toolbar.pid -and
                            $toolbarNow.thread -eq $toolbar.thread -and $toolbarNow.owner -eq 'ToDesk' -and
                            $toolbarNow.windowClass -eq 'H-SMILE-FRAME' -and $toolbarNow.title -ceq '') {
                            [WinShotV2]::ShowWindow([IntPtr]$toolbar.hwnd, 8) | Out-Null
                        }
                    }
                    $binding.remote_toolbar_restore_attempted = $true
                } finally {
                    if ($cursorWasMoved) {
                        $cursorNow = [WinShotV2+POINT]::new()
                        if ([WinShotV2]::GetCursorPos([ref]$cursorNow)) {
                            if ($cursorNow.x -eq $parkedCursor.x -and $cursorNow.y -eq $parkedCursor.y) {
                                $cursorRestored = [WinShotV2]::SetCursorPos($savedCursor.x,$savedCursor.y)
                                $binding.cursor_restore_attempted = $true
                                $binding.cursor_restore_succeeded = $cursorRestored
                            } else {
                                $binding.cursor_restore_skipped = 'Cursor moved after parking; preserve user position.'
                            }
                        } else {
                            $binding.cursor_restore_skipped = 'Cannot read current position; do not override it.'
                        }
                    }
                }
            }
        }
    }
}
Write-Output "saved $OutPng ($($bounds.Width)x$($bounds.Height)) FULLSCREEN"

    $binding.status = 'captured'
} catch {
    $binding.status = 'failed'
    $binding.error = $_.Exception.Message
    throw
} finally {
    try { Save-WindowBindingEvidence } catch { } # Diagnostics never block thread-state restoration.
    # Restore precisely the previous calling-thread execution requirement.
    if ([WinShotV2]::SetThreadExecutionState($previousExecutionState) -eq 0) {
        Write-Warning 'Could not restore the previous thread execution state; it is released when this capture process exits.'
    }
}
