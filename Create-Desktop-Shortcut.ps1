# ============================================================
#  Creates a "VoxScribe" shortcut on your Desktop
# ============================================================
#  Run once: right-click this file > "Run with PowerShell"
#  (or:  powershell -ExecutionPolicy Bypass -File Create-Desktop-Shortcut.ps1)
#
#  The shortcut points at VoxScribe.bat in this repo folder, so it keeps
#  working after you move or update the repo - just re-run this script if
#  you relocate the folder. Safe to run again anytime; it overwrites the
#  existing shortcut.
# ============================================================

$ErrorActionPreference = 'Stop'

$repo    = Split-Path -Parent $MyInvocation.MyCommand.Path
$target  = Join-Path $repo 'VoxScribe.bat'
$icon    = Join-Path $repo 'Logo.ico'
$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop 'VoxScribe.lnk'

if (-not (Test-Path $target)) {
    throw "VoxScribe.bat not found next to this script ($target)"
}

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnkPath)
$sc.TargetPath       = $target
$sc.WorkingDirectory = $repo
$sc.Description       = 'VoxScribe - local audio recording and transcription'
$sc.WindowStyle      = 7   # start the launcher window minimized
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Save()

Write-Host "Created shortcut:" -ForegroundColor Green
Write-Host "  $lnkPath"
Write-Host "  -> $target"
