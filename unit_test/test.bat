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
REM CLI Timing Harness (hands-off baseline for the CLI-vs-GUI comparison)
REM ============================================================================
REM
REM This block runs the CLI twice and prints the per-phase stats summary. The
REM identical summary is rendered by the GUI in onWorkerFinished, so a human
REM (or a watching CI job) can eyeball that the GUI's numbers match the CLI's
REM within the +/-10%% acceptance window from the plan.
REM
REM Set NVCOMP_TIMING_INPUT to point at a real, larger file (e.g. a 1 GB blob)
REM to make the comparison meaningful. If unset we fall back to sample.txt
REM which is too small to be representative but at least keeps the script
REM self-contained.

if "%NVCOMP_TIMING_INPUT%"=="" (
    set TIMING_INPUT=sample.txt
) else (
    set TIMING_INPUT=%NVCOMP_TIMING_INPUT%
)

echo.
echo ========================================
echo CLI Timing Harness ^(CLI-vs-GUI parity check^)
echo Input: %TIMING_INPUT%
echo ========================================

if not exist "%TIMING_INPUT%" (
    echo   Skipping: timing input %TIMING_INPUT% not found.
    goto :after_timing
)

echo.
echo --- CLI run #1 ^(warm-up, GPU LZ4^) ---
%EXE% -c "%TIMING_INPUT%" output\timing.lz4 lz4
echo.
echo --- CLI run #2 ^(measured, GPU LZ4^) ---
%EXE% -c "%TIMING_INPUT%" output\timing.lz4 lz4
echo.
echo --- CLI run #3 ^(--verbose, smoke-checks per-file output toggling^) ---
echo Expect one "Adding:" line per input file ^(or one "Added:" if it's a single file^).
%EXE% -c "%TIMING_INPUT%" output\timing.lz4 lz4 --verbose
echo.
echo Now run the GUI on the same file and compare the per-phase stats summary
echo it shows in the completion dialog with the "=== Compression stats ===" block
echo printed above. They should match within ~10%% per the parity acceptance.

:after_timing

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

