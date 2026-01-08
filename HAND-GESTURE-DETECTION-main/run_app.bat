@echo off
rem Simple launcher for the hand gesture app.
rem If .venv exists, use it; otherwise fall back to system Python.

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
) else (
    set PYTHON_EXE=python
)

%PYTHON_EXE% hand_gesture_display.py

endlocal

