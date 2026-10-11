@echo off
setlocal
cd /d "%~dp0"

echo ======================================
echo LegalFind Windows setup
echo ======================================

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found. Install Python 3.10 or newer from python.org,
    echo enable the Python Launcher option, then run this file again.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating project virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :failed
)

echo Installing project dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt -r WebAccess\requirements.txt pyinstaller
if errorlevel 1 goto :failed

if not exist "data\casehold-train.parquet" (
    echo Downloading the official CaseHOLD training parquet...
    ".venv\Scripts\python.exe" -m retrieval.download_dataset
    if errorlevel 1 goto :failed
)

echo Preparing the parquet data, database tables, and inverted indexes...
".venv\Scripts\python.exe" -m retrieval.initialize_instance
if errorlevel 1 goto :failed

echo Building Windows launcher...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onefile --name LegalFind --console launch_legalfind.py
if errorlevel 1 goto :failed

echo.
echo Database is ready. Starting LegalFind...
start /wait "LegalFind" "dist\LegalFind.exe"
exit /b %errorlevel%

:failed
echo.
echo Setup failed. Review the error above and run setup.bat again.
exit /b 1
