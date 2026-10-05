param([string]$BackupPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json.bak_shot")
$ErrorActionPreference = "Stop"
$settings = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
Copy-Item -LiteralPath $BackupPath -Destination $settings -Force
Remove-Item -LiteralPath $BackupPath
Write-Output "settings restored"
