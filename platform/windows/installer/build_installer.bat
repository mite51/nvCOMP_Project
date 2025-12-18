@echo off
REM ============================================================================
REM build_installer.bat - Automated WiX installer build script
REM 
REM This script compiles the WiX installer for nvCOMP.
REM Requires WiX Toolset 3.11 or later installed.
REM 
REM Usage:
REM   build_installer.bat [Release|Debug]
REM 
REM Environment:
REM   WIX - Path to WiX Toolset (auto-detected from registry if not set)
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set BUILD_CONFIG=%1
if "%BUILD_CONFIG%"=="" set BUILD_CONFIG=Release

echo ============================================================================
echo nvCOMP Installer Build Script
echo ============================================================================
echo.
echo Build Configuration: %BUILD_CONFIG%
echo.

REM ============================================================================
REM Find WiX Toolset
REM ============================================================================

if defined WIX (
    echo Using WiX from environment: %WIX%
) else (
    echo Detecting WiX Toolset installation...
    
    REM Try common installation paths
    set "WIX_PATHS=%ProgramFiles(x86)%\WiX Toolset v3.11"
    set "WIX_PATHS=%WIX_PATHS%;%ProgramFiles(x86)%\WiX Toolset v3.14"
    set "WIX_PATHS=%WIX_PATHS%;%ProgramFiles%\WiX Toolset v3.11"
    set "WIX_PATHS=%WIX_PATHS%;%ProgramFiles%\WiX Toolset v3.14"
    
    for %%p in (%WIX_PATHS%) do (
        if exist "%%p\bin\candle.exe" (
            set "WIX=%%p"
            echo Found WiX at: !WIX!
            goto :wix_found
        )
    )
    
    echo ERROR: WiX Toolset not found!
    echo.
    echo Please install WiX Toolset 3.11 or later from:
    echo https://wixtoolset.org/releases/
    echo.
    echo Or set the WIX environment variable to your WiX installation directory.
    echo.
    exit /b 1
)

:wix_found

set "CANDLE=%WIX%\bin\candle.exe"
set "LIGHT=%WIX%\bin\light.exe"

if not exist "%CANDLE%" (
    echo ERROR: candle.exe not found at: %CANDLE%
    exit /b 1
)

if not exist "%LIGHT%" (
    echo ERROR: light.exe not found at: %LIGHT%
    exit /b 1
)

echo WiX binaries located successfully
echo.

REM ============================================================================
REM Check for required files
REM ============================================================================

echo Checking for source files...
echo.

set "SOURCE_DIR=..\..\..\build_gui\bin\%BUILD_CONFIG%"
set "RESOURCE_DIR=..\..\gui\resources"

REM Check for main executable
if not exist "%SOURCE_DIR%\nvcomp-gui.exe" (
    echo ERROR: nvcomp-gui.exe not found in %SOURCE_DIR%
    echo.
    echo Please build the GUI application first:
    echo   cd build_gui
    echo   cmake --build . --config %BUILD_CONFIG%
    echo.
    exit /b 1
)

echo [OK] Found nvcomp-gui.exe
if exist "%SOURCE_DIR%\nvcomp-cli.exe" (
    echo [OK] Found nvcomp-cli.exe
) else (
    echo [WARNING] nvcomp-cli.exe not found - CLI feature will be excluded
)

REM Check for DLLs
if not exist "%SOURCE_DIR%\nvcomp_core.dll" (
    echo ERROR: nvcomp_core.dll not found
    exit /b 1
)
echo [OK] Found nvcomp_core.dll

if not exist "%SOURCE_DIR%\nvcomp64_5.dll" (
    echo ERROR: nvcomp64_5.dll not found
    exit /b 1
)
echo [OK] Found nvcomp64_5.dll

if not exist "%SOURCE_DIR%\Qt6Core.dll" (
    echo ERROR: Qt6Core.dll not found - did windeployqt run?
    exit /b 1
)
echo [OK] Found Qt dependencies

echo.
echo All required files found.
echo.

REM ============================================================================
REM Generate unique GUIDs if needed
REM ============================================================================

echo Checking GUIDs in product.wxi...
findstr /C:"12345678-1234-1234-1234-123456789ABC" product.wxi >nul
if %ERRORLEVEL%==0 (
    echo.
    echo WARNING: product.wxi contains placeholder GUIDs!
    echo.
    echo You should generate unique GUIDs and update product.wxi before building.
    echo Run this PowerShell command to generate new GUIDs:
    echo   [guid]::NewGuid^(^)
    echo.
    echo Press Ctrl+C to cancel or any key to continue anyway...
    pause >nul
)

REM ============================================================================
REM Create output directory
REM ============================================================================

set "OUTPUT_DIR=output"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Output directory: %OUTPUT_DIR%
echo.

REM ============================================================================
REM Compile WiX source files (candle)
REM ============================================================================

echo ============================================================================
echo Step 1: Compiling WiX source files...
echo ============================================================================
echo.

"%CANDLE%" -nologo ^
    -arch x64 ^
    -ext WixUIExtension ^
    -dBuildConfiguration=%BUILD_CONFIG% ^
    -out "%OUTPUT_DIR%\\" ^
    installer.wxs ^
    components.wxs ^
    ui.wxs

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Compilation failed!
    exit /b %ERRORLEVEL%
)

echo.
echo Compilation successful.
echo.

REM ============================================================================
REM Link WiX object files (light)
REM ============================================================================

echo ============================================================================
echo Step 2: Linking installer package...
echo ============================================================================
echo.

REM Read version from product.wxi
for /f "tokens=2 delims==" %%v in ('findstr /C:"ProductVersion" product.wxi') do (
    set VERSION_LINE=%%v
)
REM Extract version number (remove quotes and spaces)
set VERSION_LINE=%VERSION_LINE:"=%
set VERSION_LINE=%VERSION_LINE: =%
set VERSION_LINE=%VERSION_LINE:?^>=%
for /f "tokens=1" %%v in ("%VERSION_LINE%") do set PRODUCT_VERSION=%%v

set "MSI_NAME=nvCOMP-%PRODUCT_VERSION%-x64.msi"

"%LIGHT%" -nologo ^
    -ext WixUIExtension ^
    -cultures:en-us ^
    -out "%OUTPUT_DIR%\%MSI_NAME%" ^
    "%OUTPUT_DIR%\installer.wixobj" ^
    "%OUTPUT_DIR%\components.wixobj" ^
    "%OUTPUT_DIR%\ui.wixobj" ^
    -sval

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Linking failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo Build Complete!
echo ============================================================================
echo.
echo Installer created: %OUTPUT_DIR%\%MSI_NAME%
echo.

REM ============================================================================
REM Display file information
REM ============================================================================

for %%f in ("%OUTPUT_DIR%\%MSI_NAME%") do (
    set SIZE=%%~zf
    set /a SIZE_MB=!SIZE! / 1048576
    echo Size: !SIZE_MB! MB
)

echo.
echo To install:
echo   %OUTPUT_DIR%\%MSI_NAME%
echo.
echo To install silently:
echo   msiexec /i %OUTPUT_DIR%\%MSI_NAME% /quiet /qn /norestart
echo.
echo To uninstall:
echo   msiexec /x %OUTPUT_DIR%\%MSI_NAME% /quiet /qn /norestart
echo.

REM ============================================================================
REM Optional: Sign the MSI
REM ============================================================================

if defined SIGNTOOL_PATH (
    echo ============================================================================
    echo Code Signing
    echo ============================================================================
    echo.
    
    if defined CERT_THUMBPRINT (
        echo Signing installer with certificate...
        "%SIGNTOOL_PATH%\signtool.exe" sign /sha1 %CERT_THUMBPRINT% /t http://timestamp.digicert.com /fd SHA256 "%OUTPUT_DIR%\%MSI_NAME%"
        
        if !ERRORLEVEL! neq 0 (
            echo WARNING: Code signing failed!
        ) else (
            echo Installer signed successfully.
        )
    ) else (
        echo CERT_THUMBPRINT not set - skipping code signing
    )
    echo.
)

endlocal
exit /b 0

