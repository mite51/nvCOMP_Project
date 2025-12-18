@echo off
REM Quick test script for the WiX installer build
REM This checks for prerequisites without actually building

echo ============================================================================
echo nvCOMP WiX Installer - Quick Prerequisite Check
echo ============================================================================
echo.

REM Check for WiX
echo [1/4] Checking for WiX Toolset...
where candle.exe >nul 2>&1
if %ERRORLEVEL%==0 (
    echo   [OK] WiX Toolset found
    for /f "tokens=*" %%a in ('where candle.exe') do set WIX_PATH=%%a
    echo   Path: !WIX_PATH!
) else (
    echo   [MISSING] WiX Toolset not found
    echo   Download from: https://wixtoolset.org/releases/
)
echo.

REM Check for build output
echo [2/4] Checking for build output...
set BUILD_DIR=..\..\..\build_gui\bin\Release
if exist "%BUILD_DIR%\nvcomp-gui.exe" (
    echo   [OK] nvcomp-gui.exe found
) else (
    echo   [MISSING] nvcomp-gui.exe not found
    echo   Build it first: cd build_gui ^&^& cmake --build . --config Release
)
echo.

REM Check for Qt deployment
echo [3/4] Checking for Qt deployment...
if exist "%BUILD_DIR%\Qt6Core.dll" (
    echo   [OK] Qt dependencies found
) else (
    echo   [MISSING] Qt DLLs not found
    echo   windeployqt should run automatically during build
)
echo.

REM Check for GUIDs
echo [4/4] Checking for placeholder GUIDs...
findstr /C:"12345678-1234-1234-1234-123456789ABC" product.wxi >nul 2>&1
if %ERRORLEVEL%==0 (
    echo   [WARNING] Placeholder GUIDs detected in product.wxi
    echo   Generate unique GUIDs: powershell "[guid]::NewGuid()"
) else (
    echo   [OK] GUIDs appear to be customized
)
echo.

echo ============================================================================
echo Summary
echo ============================================================================
echo.
echo If all checks passed, you can build the installer:
echo   build_installer.bat Release
echo.
echo For more information, see README.md
echo.

pause

