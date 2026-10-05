param(
    [Parameter(Mandatory=$true)][string]$BashScript,
    [Parameter(Mandatory=$true)][string]$OutPng,
    [int]$TimeoutSec = 900
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot 'capture_validation.ps1')
$token = [guid]::NewGuid().ToString("N")
$marker = Join-Path $env:TEMP "wsl-shot-$token.done"
$launcher = Join-Path $env:TEMP "wsl-shot-$token.sh"
$rawLog = [IO.Path]::ChangeExtension($OutPng, '.log')
$sidecar = [IO.Path]::ChangeExtension($OutPng, '.json')
$failureSidecar = [IO.Path]::ChangeExtension($OutPng, ".failed-$token.json")
# Never let a previous image/sidecar masquerade as this execution's result.
foreach ($existing in @($OutPng, $rawLog, $sidecar)) {
    if (Test-Path -LiteralPath $existing) { throw "Refusing existing evidence output: $existing" }
}
$backup = Join-Path $env:TEMP "wsl-shot-settings-$token.json"
$toWsl = { param($value) (& wsl.exe -d Ubuntu --exec wslpath -a $value.Replace('\','/')).Trim() }
$runnerWsl = & $toWsl (Join-Path $PSScriptRoot 'run_raw.sh')
$markerWsl = & $toWsl $marker
$logWsl = & $toWsl $rawLog
$launcherWsl = & $toWsl $launcher
if (@($BashScript,$runnerWsl,$markerWsl,$logWsl) | Where-Object { $_ -match "['\r\n]" }) {
    throw 'Paths containing apostrophes or newlines are unsupported.'
}
[IO.File]::WriteAllText($launcher, "#!/usr/bin/env bash`nexec bash '$runnerWsl' '$BashScript' '$markerWsl' '$logWsl'`n", [Text.UTF8Encoding]::new($false))
$parent = Split-Path -Parent $OutPng
if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
$profileInstalled = $false
try {
    & (Join-Path $PSScriptRoot 'wt_purple_on.ps1') -BackupPath $backup
    $profileInstalled = $true
    & (Join-Path $PSScriptRoot 'take_shot_full.ps1') -BashScript $launcherWsl -OutPng $OutPng -TimeoutSec $TimeoutSec -MarkerPath $marker
    if (-not (Test-Path -LiteralPath $OutPng)) { throw 'Capture returned without an accepted image.' }
    if (-not (Test-Path -LiteralPath $marker) -or (Get-Content -Raw -LiteralPath $marker).Trim() -ne '0') {
        throw 'Missing or nonzero command completion status; no success metadata written.'
    }
    # Reopen and validate the exact persisted PNG, independently of the lower
    # capture routine's in-memory validation, before hashing or certifying it.
    $quality = Assert-WslShotImageFile -Path $OutPng
    $metadata = [ordered]@{
        capture_status = 'accepted_not_black'
        captured_at = (Get-Date).ToUniversalTime().ToString('o')
        script = $BashScript
        image = $OutPng
        raw_log = $rawLog
        exit_status = 0
        image_quality = $quality
        sha256_before_taskbar_crop = (Get-FileHash -Algorithm SHA256 -LiteralPath $OutPng).Hash.ToLowerInvariant()
        review_required = 'Inspect real terminal content, theme, popups and command identity before publication.'
    }
} catch {
    $failure = [ordered]@{
        capture_status = 'failed'
        image_accepted = $false
        failed_at = (Get-Date).ToUniversalTime().ToString('o')
        script = $BashScript
        intended_image = $OutPng
        raw_log = $rawLog
        reason = $_.Exception.Message
        command_completion_marker = $(if (Test-Path -LiteralPath $marker) { (Get-Content -Raw -LiteralPath $marker).Trim() } else { $null })
        note = 'Command completion and screen capture validity are independent; preserve the raw log. Do not rerun GPU work solely to repair a failed image.'
    }
    [IO.File]::WriteAllText($failureSidecar, ($failure | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
    throw
} finally {
    try {
        if ($profileInstalled) { & (Join-Path $PSScriptRoot 'wt_purple_off.ps1') -BackupPath $backup }
    } finally {
        Remove-Item -LiteralPath $launcher -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $marker -ErrorAction SilentlyContinue
    }
}
# Only a successful capture, file validation and profile restoration reach here.
$tempSidecar = "$sidecar.tmp-$token"
[IO.File]::WriteAllText($tempSidecar, ($metadata | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
[IO.File]::Move($tempSidecar, $sidecar)