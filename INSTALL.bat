@echo off
echo ======================================
echo peppenwriter Installation
echo ======================================
echo.
REM Ensure Python 3.10 is used
where py > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: The 'py' launcher is not installed. Please install Python 3.10 from https://www.python.org/downloads/
    pause
    exit /b 1
)

py -3.10 --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3.10 is not installed. Please install Python 3.10 from https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10 from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Installing required packages...
py -3.10 -m pip install torch==2.0.0+cu118 tqdm numpy==1.23.5
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install required packages.
    echo Please run: pip install torch==2.0.0+cu118 tqdm numpy==1.23.5
    echo.
    pause
    exit /b 1
)

echo Creating desktop shortcut...
@echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\shortcut.vbs"
@echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\peppenwriter.lnk" >> "%TEMP%\shortcut.vbs"
@echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\shortcut.vbs"
@echo oLink.TargetPath = "pythonw.exe" >> "%TEMP%\shortcut.vbs"
@echo oLink.Arguments = "main.py" >> "%TEMP%\shortcut.vbs"
@echo oLink.WorkingDirectory = "%CD%\peppenwriter" >> "%TEMP%\shortcut.vbs"
@echo oLink.IconLocation = "%CD%\peppenwriter\icon.ico" >> "%TEMP%\shortcut.vbs"
@echo oLink.Description = "written insanity" >> "%TEMP%\shortcut.vbs"
@echo oLink.Save >> "%TEMP%\shortcut.vbs"
cscript //nologo "%TEMP%\shortcut.vbs"
del "%TEMP%\shortcut.vbs"

echo.
echo Installation complete! A shortcut has been created on your desktop.
echo.
pause
