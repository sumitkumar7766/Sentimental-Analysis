@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  SentiMind: Restoring backend/venv on Windows
echo ========================================================

cd /d "%~dp0"

if not exist "venv_parts" (
    echo Error: Directory venv_parts not found!
    pause
    exit /b 1
)

echo [1/2] Combining 90MB chunks into backend_venv.tar.gz...
copy /b venv_parts\backend_venv.tar.gz.part_* backend_venv.tar.gz > nul

echo [2/2] Extracting archive to project root...
tar -xzf backend_venv.tar.gz -C ..
del /f /q backend_venv.tar.gz

echo ========================================================
echo  Done! backend/venv reconstructed successfully.
echo  Note: On Windows, to run natively use: python -m venv venv
echo ========================================================
pause
