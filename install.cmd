@echo off
setlocal
cd /d "%~dp0"

title Kyven Portable Installer
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "KYVEN_INSTALL_EXIT=%ERRORLEVEL%"

echo.
if not "%KYVEN_INSTALL_EXIT%"=="0" echo Kyven installation failed. Read the error above.
pause
exit /b %KYVEN_INSTALL_EXIT%
