@echo off
setlocal enabledelayedexpansion
title 360extract setup
cd /d "%~dp0"

echo ============================================
echo  360extract setup
echo ============================================
echo.

rem ---- 1. Python --------------------------------------------------------
echo [1/4] Python 3.10+
set "PYLAUNCH="
where py >nul 2>&1
if not errorlevel 1 set "PYLAUNCH=py -3"
if not defined PYLAUNCH (
    where python >nul 2>&1
    if not errorlevel 1 set "PYLAUNCH=python"
)

if not defined PYLAUNCH (
    echo   Not found.
    where winget >nul 2>&1
    if errorlevel 1 (
        echo   winget is not available. Install Python 3.10+ yourself:
        echo     https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo   Installing Python via winget...
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo   winget install failed. Install Python 3.10+ yourself:
        echo     https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo.
    echo   Installed. Close this window and re-run setup.bat so PATH picks it up.
    pause
    exit /b 0
)
for /f "tokens=*" %%v in ('%PYLAUNCH% --version 2^>^&1') do echo   Found: %%v

rem ---- 2. Virtual environment + Python package ---------------------------
echo.
echo [2/4] Virtual environment and Python package
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo   Creating .venv ...
    %PYLAUNCH% -m venv .venv
    if errorlevel 1 (
        echo   Could not create the virtual environment.
        pause
        exit /b 1
    )
)

echo   Installing 360extract (with dev/test extras)...
"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip install --quiet -e ".[dev]"
if errorlevel 1 (
    echo   Install failed. Run "%PY% -m pip install -e .[dev]" to see the error.
    pause
    exit /b 1
)
echo   Done.

echo.
set /p WANTML="  Also install ML masking deps -- torch, ultralytics -- (y/N)? "
if /i "%WANTML%"=="y" (
    echo   Installing [ml] extras, this downloads a large torch wheel...
    "%PY%" -m pip install --quiet -e ".[ml]"
    if errorlevel 1 (
        echo   [ml] install failed. Run "%PY% -m pip install -e .[ml]" to see the error.
    ) else (
        echo   Done. For a CUDA build of torch instead of CPU-only, see the README's
        echo   "GPU" section under Masking.
    )
)

rem ---- 3. ffmpeg ----------------------------------------------------------
echo.
echo [3/4] ffmpeg 5.0+ with the v360 filter
"%PY%" -m threesixty.cli doctor >nul 2>&1
if errorlevel 1 (
    echo   Not found.
    where winget >nul 2>&1
    if not errorlevel 1 (
        echo   Installing ffmpeg via winget...
        winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
        if errorlevel 1 (
            echo   winget install failed. Get ffmpeg yourself:
            echo     https://ffmpeg.org/download.html
        ) else (
            echo   Installed. Close this window and re-run setup.bat so PATH picks it up.
            pause
            exit /b 0
        )
    ) else (
        echo   winget is not available. Get ffmpeg yourself:
        echo     https://ffmpeg.org/download.html
    )
) else (
    echo   OK.
)

rem ---- 4. COLMAP, Brush, SuperSplat ---------------------------------------
echo.
echo [4/4] COLMAP, Brush, SuperSplat
echo   These are not installed automatically: COLMAP needs a CUDA/no-CUDA choice,
echo   Brush is a single binary from a GitHub release, and SuperSplat is a web
echo   build the app points at rather than something on PATH. Full status:
echo.
"%PY%" -m threesixty.cli doctor
echo.
echo   Whatever is missing above, get it and either put it on PATH or set its
echo   location in the app: run 360extract-ui.bat, open System status, and set
echo   the path there. Download pages:
echo     COLMAP:     https://github.com/colmap/colmap/releases
echo     Brush:      https://github.com/ArthurBrussee/brush/releases
echo     SuperSplat: https://github.com/playcanvas/supersplat

echo.
echo ============================================
echo  Setup finished. Run 360extract-ui.bat to start the app.
echo ============================================
pause
