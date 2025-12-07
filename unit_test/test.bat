@echo off
REM nvCOMP CLI Test Suite for Windows
REM Tests: 15 total

setlocal enabledelayedexpansion

echo ========================================
echo nvCOMP CLI Test Suite
echo ========================================
echo.

set TEST_COUNT=0
set PASS_COUNT=0
set FAIL_COUNT=0

REM Create test input file (if it doesn't exist)
if not exist sample.txt (
    echo Creating test input file...
    echo This is a test file for nvCOMP CLI compression and decompression. > sample.txt
    echo It contains some repetitive text to test compression ratios. >> sample.txt
    echo This is a test file for nvCOMP CLI compression and decompression. >> sample.txt
    echo It contains some repetitive text to test compression ratios. >> sample.txt
    for /L %%i in (1,1,10) do (
        echo Line %%i: The quick brown fox jumps over the lazy dog. >> sample.txt
    )
)

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

REM ============================================================================
REM GPU Batched <-> GPU Batched (LZ4, Snappy, Zstd)
REM ============================================================================

echo.
echo ========================================
echo GPU Batched Tests
echo ========================================

call :run_test "GPU Batched LZ4" lz4 ""
call :run_test "GPU Batched Snappy" snappy ""
call :run_test "GPU Batched Zstd" zstd ""

REM ============================================================================
REM GPU Manager <-> GPU Manager (GDeflate, ANS, Bitcomp)
REM ============================================================================

echo.
echo ========================================
echo GPU Manager Tests
echo ========================================

call :run_test "GPU Manager GDeflate" gdeflate ""
call :run_test "GPU Manager ANS" ans ""
call :run_test "GPU Manager Bitcomp" bitcomp ""

REM ============================================================================
REM CPU <-> CPU (LZ4, Snappy, Zstd)
REM ============================================================================

echo.
echo ========================================
echo CPU Tests
echo ========================================

call :run_test "CPU LZ4" lz4 "--cpu"
call :run_test "CPU Snappy" snappy "--cpu"
call :run_test "CPU Zstd" zstd "--cpu"

REM ============================================================================
REM GPU Batched -> CPU (Cross-compatibility)
REM ============================================================================

echo.
echo ========================================
echo GPU to CPU Cross-compatibility Tests
echo ========================================

call :run_cross_test "GPU->CPU LZ4" lz4
call :run_cross_test "GPU->CPU Snappy" snappy
call :run_cross_test "GPU->CPU Zstd" zstd

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
    exit /b 1
) else (
    echo All tests passed!
    exit /b 0
)

REM ============================================================================
REM Helper Functions
REM ============================================================================

:run_test
set TEST_NAME=%~1
set ALGO=%~2
set FLAGS=%~3
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing...
%EXE% -c sample.txt output\test.%ALGO% %ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Decompressing...
if exist output\restored rmdir /s /q output\restored
%EXE% -d output\test.%ALGO% output\restored %ALGO% %FLAGS%
if errorlevel 1 (
    echo   FAILED: Decompression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Verifying...
fc /b sample.txt output\restored\sample.txt > nul
if errorlevel 1 (
    echo   FAILED: Files do not match
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED
set /a PASS_COUNT+=1
goto :eof

:run_cross_test
set TEST_NAME=%~1
set ALGO=%~2
set /a TEST_COUNT+=1

echo.
echo [Test %TEST_COUNT%] %TEST_NAME%
echo   Compressing with GPU...
%EXE% -c sample.txt output\test.%ALGO% %ALGO%
if errorlevel 1 (
    echo   FAILED: GPU Compression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Decompressing with CPU...
if exist output\restored rmdir /s /q output\restored
%EXE% -d output\test.%ALGO% output\restored %ALGO% --cpu
if errorlevel 1 (
    echo   FAILED: CPU Decompression failed
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   Verifying...
fc /b sample.txt output\restored\sample.txt > nul
if errorlevel 1 (
    echo   FAILED: Files do not match
    set /a FAIL_COUNT+=1
    goto :eof
)

echo   PASSED
set /a PASS_COUNT+=1
goto :eof

