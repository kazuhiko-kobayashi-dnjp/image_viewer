@echo off
echo Installing dependencies for Image Viewer...
py -3 -m pip install PyQt5 numpy opencv-python
echo.

echo Creating desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_shortcut.ps1" -VbsPath "%~dp0image_viewer.vbs" -WorkDir "%~dp0"

echo.
echo Done.
echo - Shortcut placed on Desktop: "Image Viewer.lnk"
echo - Right-click it and choose "Pin to taskbar" to add it to the taskbar.
pause
