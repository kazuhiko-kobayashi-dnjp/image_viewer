param(
    [string]$VbsPath,
    [string]$WorkDir
)

$lnk = "$env:USERPROFILE\Desktop\Image Viewer.lnk"
$ws  = New-Object -ComObject WScript.Shell
$sc  = $ws.CreateShortcut($lnk)
$sc.TargetPath       = 'C:\Windows\System32\wscript.exe'
$sc.Arguments        = '//nologo "' + $VbsPath + '"'
$sc.WorkingDirectory = $WorkDir.TrimEnd('\')
$sc.IconLocation     = 'C:\Windows\System32\imageres.dll,20'
$sc.Description      = 'Image Viewer'
$sc.Save()
Write-Host "Shortcut saved: $lnk"
