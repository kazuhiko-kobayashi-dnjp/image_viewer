param(
    [string]$VbsPath,
    [string]$WorkDir
)

$ErrorActionPreference = 'Stop'

$ws   = New-Object -ComObject WScript.Shell
$desk = $ws.SpecialFolders("Desktop")
if (-not $desk -or -not (Test-Path $desk)) {
    Write-Error "Desktop folder not found: '$desk'"
    exit 1
}
$lnk = Join-Path $desk "Image Viewer.lnk"
$sc  = $ws.CreateShortcut($lnk)
$sc.TargetPath   = 'C:\Windows\System32\wscript.exe'
$sc.Arguments    = '//nologo "' + $VbsPath + '"'
$sc.IconLocation = 'C:\Windows\System32\imageres.dll,20'
$sc.Description  = 'Image Viewer'
$sc.Save()
if (Test-Path $lnk) {
    Write-Host "Shortcut created: $lnk"
} else {
    Write-Error "Save() returned no error but file not found: $lnk"
    exit 1
}
