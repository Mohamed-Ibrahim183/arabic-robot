@echo off
REM Start Chatterbox-TTS-Server with CUDA 12.8 portable mode.
REM Keep this window open while using "Chatterbox TTS Server" in the GUI.

cd /d "%~dp0Chatterbox-TTS-Server"
if not exist "start.py" (
  echo Chatterbox-TTS-Server is missing.
  echo Clone it with:
  echo   git clone https://github.com/devnen/Chatterbox-TTS-Server.git
  pause
  exit /b 1
)

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
chcp 65001 >nul

echo Starting Chatterbox TTS Server on http://127.0.0.1:8004 ...
echo First run downloads Python + CUDA deps and can take a long time.
echo In the server Web UI, select Chatterbox Multilingual for Arabic.
echo.

python start.py --portable --nvidia-cu128 %*
if errorlevel 1 (
  echo.
  echo Launcher failed. Try: python start.py --portable --nvidia
  pause
)
