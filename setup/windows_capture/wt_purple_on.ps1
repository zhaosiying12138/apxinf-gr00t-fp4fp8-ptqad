param([string]$BackupPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json.bak_shot")
$ErrorActionPreference = "Stop"
$f = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
if (Test-Path -LiteralPath $BackupPath) {
    throw "A screenshot settings backup already exists. Restore it with wt_purple_off.ps1 before starting a new session."
}
Copy-Item -LiteralPath $f -Destination $BackupPath
$j = Get-Content $f -Raw | ConvertFrom-Json
# remove any previous shot profile, then add a purple Ubuntu one
$keep = @($j.profiles.list | Where-Object { $_.name -ne "UbuntuPurpleShot" })
$prof = [PSCustomObject]@{
    name = "UbuntuPurpleShot"
    commandline = "wsl.exe -d Ubuntu"
    colorScheme = "Ubuntu"
    background = "#300A24"
    startingDirectory = "%USERPROFILE%"
    historySize = 9999
}
$listType = $keep + @($prof)
if ($j.profiles.list -is [System.Array]) { $j.profiles.list = $listType }
else { $j.profiles.list = $listType }
$j | ConvertTo-Json -Depth 30 | Set-Content $f -Encoding UTF8
Write-Output "purple profile injected (backup at $BackupPath)"
