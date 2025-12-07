@echo off
REM nvCOMP CLI Folder Compression Tests
REM Tests for folder compression feature and original zstd decompression issue

setlocal enabledelayedexpansion

echo ========================================
echo nvCOMP CLI Folder & Archive Tests
echo ========================================
echo.

set TEST_COUNT=0
set PASS_COUNT=0
set FAIL_COUNT=0

REM Create output directory
if not exist output mkdir output

set EXE=..\build\Release\nvcomp_cli.exe
if not exist %EXE% (
    set EXE=..\build\x64\Release\nvcomp_cli.exe
)
if not exist %EXE% (
    echo ERROR: nvcomp_cli.exe not found. Build the project first.
    exit /b 1
)

REM Verify sample_folder exists
if not exist sample_folder (
    echo ERROR: sample_folder not found. Create sample_folder directory with test files.
    exit /b 1
)

REM ============================================================================
REM Folder Compression Tests
REM ============================================================================

echo.
echo ========================================
echo Folder Compression Tests
echo ========================================

call :run_folder_test "Folder GPU LZ4" lz4 ""
call :run_folder_test "Folder GPU Zstd" zstd ""
call :run_folder_test "Folder CPU LZ4" lz4 "--cpu"
call :run_folder_test "Folder CPU Zstd" zstd "--cpu"

REM ============================================================================
REM Archive Listing Tests
REM ============================================================================

echo.
echo ========================================
echo Archive Listing Tests
echo ========================================

call :run_list_test "List GPU Zstd Archive" zstd ""
call :run_list_test "List GPU LZ4 Archive" lz4 ""
call :run_list_test "List CPU Zstd Archive" zstd "--cpu"

REM ============================================================================
REM GPU Zstd Round-Trip Test (Original Issue)
REM ============================================================================

echo.
echo ========================================
echo GPU Zstd Round-Trip Test
echo (Tests the original decompression issue)
echo ========================================

call :run_roundtrip_test "GPU Zstd Full Round-Trip" zstd

REM ============================================================================
REM Algorithm Auto-Detection Tests
REM ============================================================================

echo.
echo ========================================
echo Algorithm Auto-Detection Tests
echo ========================================

call :run_autodetect_list_test "Auto-detect LZ4 (List)" lz4 ""
call :run_autodetect_list_test "Auto-detect Snappy (List)" snappy ""
call :run_autodetect_list_test "Auto-detect Zstd (List)" zstd ""
call :run_autodetect_decompress_test "Auto-detect LZ4 (Decompress)" lz4 ""
call :run_autodetect_decompress_test "Auto-detect Zstd (Decompress)" zstd ""
call :run_autodetect_decompress_test "Auto-detect LZ4 CPU (Decompress)" lz4 "--cpu"

REM ============================================================================
REM Summary
REM ============================================================================

echo.
echo ========================================
echo Test Summary
echo ========================================
echo Total tests: %TEST_COUNT%
echo Passed: %PASS_COUNT%
echo Failed: %FAIL_COUNT%
echo ========================================

if %FAIL_COUNT% gtr 0 (
    echo.
    echo SOME TESTS FAILED!
    exit /b 1
) else (
    echo.
    echo ALL TESTS PASSED!
    exit /b 0
)

REM ============================================================================
REM Helper Functions
REM ============================================================================

:run_folder_test
set TEST_NAME=%~1
set ALGO=%~2
set FLAGS=%~3
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing folder...
%EXE% -c sample_folder output\folder.%ALGO% %ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Folder compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Listing archive...
%EXE% -l output\folder.%ALGO% %ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Archive listing failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Decompressing folder...
if exist output\folder_restored_%ALGO% rmdir /s /q output\folder_restored_%ALGO%
%EXE% -d output\folder.%ALGO% output\folder_restored_%ALGO% %ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Folder decompression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Verifying folder structure...
if not exist output\folder_restored_%ALGO%\PineTools.com_files (
    echo   FAILED: Folder structure not preserved
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED
set /a PASS_COUNT+=1
goto :eof

:run_list_test
set TEST_NAME=%~1
set ALGO=%~2
set FLAGS=%~3
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing folder for listing test...
%EXE% -c sample_folder output\list_test.%ALGO% %ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Listing archive contents...
%EXE% -l output\list_test.%ALGO% %ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Archive listing failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED
set /a PASS_COUNT+=1
goto :eof

:run_roundtrip_test
set TEST_NAME=%~1
set ALGO=%~2
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   This test reproduces the original issue:
echo   "Using CPU decompression (zstd)..."
echo   "Error: Zstd CPU decompression failed"
echo.

echo   Step 1: Compressing folder with GPU...
%EXE% -c sample_folder output\roundtrip.%ALGO% %ALGO%
if errorlevel 1 (
    echo   FAILED: GPU Compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Step 2: Listing archive (this was failing before)...
%EXE% -l output\roundtrip.%ALGO% %ALGO%
if errorlevel 1 (
    echo   FAILED: Archive listing failed - ORIGINAL ISSUE REPRODUCED!
    echo   This is the error that was reported.
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Step 3: Decompressing (this was also failing)...
if exist output\roundtrip_restored rmdir /s /q output\roundtrip_restored
%EXE% -d output\roundtrip.%ALGO% output\roundtrip_restored %ALGO%
if errorlevel 1 (
    echo   FAILED: Decompression failed - ORIGINAL ISSUE REPRODUCED!
    echo   This is the error that was reported.
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Step 4: Verifying decompressed folder...
if not exist output\roundtrip_restored\PineTools.com_files (
    echo   FAILED: Folder structure not preserved
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED - Original issue FIXED!
echo   GPU zstd compression now works correctly with listing and decompression.
set /a PASS_COUNT+=1
goto :eof

:run_autodetect_list_test
set TEST_NAME=%~1
set ALGO=%~2
set FLAGS=%~3
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing folder...
%EXE% -c sample_folder output\autodetect.%ALGO% %ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Listing archive WITHOUT algorithm parameter...
%EXE% -l output\autodetect.%ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Auto-detection listing failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED - Algorithm auto-detected successfully!
set /a PASS_COUNT+=1
goto :eof

:run_autodetect_decompress_test
set TEST_NAME=%~1
set ALGO=%~2
set FLAGS=%~3
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing folder...
%EXE% -c sample_folder output\autodetect.%ALGO% %ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Decompressing WITHOUT algorithm parameter...
if exist output\autodetect_restored rmdir /s /q output\autodetect_restored
%EXE% -d output\autodetect.%ALGO% output\autodetect_restored %FLAGS%
if errorlevel 1 (
    echo   FAILED: Auto-detection decompression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Verifying folder structure...
if not exist output\autodetect_restored\PineTools.com_files (
    echo   FAILED: Folder structure not preserved
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED - Algorithm auto-detected and decompressed successfully!
set /a PASS_COUNT+=1
goto :eof

