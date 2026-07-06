@echo off
echo Installing dependencies for Image Viewer...
py -3 -m pip install PyQt5 numpy opencv-python
echo.

echo Creating desktop shortcut...
powershell -NoProfile -Command ^
  "$lnk = '%USERPROFILE%\Desktop\Image Viewer.lnk';" ^
  "$vbs = '%~dp0image_viewer.vbs';" ^
  "$ws  = New-Object -ComObject WScript.Shell;" ^
  "$sc  = $ws.CreateShortcut($lnk);" ^
  "$sc.TargetPath   = 'C:\Windows\System32\wscript.exe';" ^
  "$sc.Arguments    = '//nologo \"' + $vbs + '\"';" ^
  "$sc.WorkingDirectory = '%~dp0'.TrimEnd('\');" ^
  "$sc.IconLocation = 'C:\Windows\System32\imageres.dll,20';" ^
  "$sc.Description  = 'Image Viewer';" ^
  "$sc.Save();" ^
  "Write-Host 'Shortcut created:' $lnk"

echo.
echo Done.
echo - Shortcut placed on Desktop: "Image Viewer.lnk"
echo - Right-click it and choose "Pin to taskbar" to add it to the taskbar.
pause
