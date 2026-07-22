@echo off
setlocal
rem Launch the 360extract rig editor. Creates the virtualenv on first run.

cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo First run: creating virtual environment...
    where py >nul 2>&1
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 (
        echo.
        echo Could not create a virtual environment. Is Python 3.10+ installed and on PATH?
        pause
        exit /b 1
    )
    "%PY%" -m pip install --quiet --upgrade pip
    "%PY%" -m pip install --quiet -e .
    if errorlevel 1 (
        echo.
        echo Install failed. Run "%PY% -m pip install -e ." to see the error.
        pause
        exit /b 1
    )
)

rem Check ffmpeg before opening a browser onto a UI that cannot work.
"%PY%" -m threesixty.cli doctor >nul 2>&1
if errorlevel 1 (
    echo.
    echo No usable ffmpeg found. 360extract needs ffmpeg 5.0+ with the v360 filter.
    echo Details:
    echo.
    "%PY%" -m threesixty.cli doctor
    echo.
    pause
    exit /b 1
)

"%PY%" -m threesixty.cli ui %*
if errorlevel 1 pause
