#!/bin/bash
# nvCOMP CLI Test Suite for Linux
# Tests: 15 total

set -e

echo "========================================"
echo "nvCOMP CLI Test Suite"
echo "========================================"
echo

TEST_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

# Create test input file (if it doesn't exist)
if [ ! -f "sample.txt" ]; then
    echo "Creating test input file..."
    {
        echo "This is a test file for nvCOMP CLI compression and decompression."
        echo "It contains some repetitive text to test compression ratios."
        echo "This is a test file for nvCOMP CLI compression and decompression."
        echo "It contains some repetitive text to test compression ratios."
        for i in {1..10}; do
            echo "Line $i: The quick brown fox jumps over the lazy dog."
        done
    } > sample.txt
fi

# Create output directory
mkdir -p output

# Find executable
EXE="../build/nvcomp_cli"
if [ ! -f "$EXE" ]; then
    EXE="../build/Release/nvcomp_cli"
fi
if [ ! -f "$EXE" ]; then
    echo "ERROR: nvcomp_cli not found. Build the project first."
    exit 1
fi

# Helper function to run a test
run_test() {
    local TEST_NAME=$1
    local ALGO=$2
    local FLAGS=$3
    
    ((TEST_COUNT++))
    
    echo
    echo "[Test $TEST_COUNT] $TEST_NAME"
    
    # Compress
    echo "  Compressing..."
    if ! $EXE -c sample.txt output/test.$ALGO $ALGO $FLAGS; then
        echo "  FAILED: Compression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Decompress
    echo "  Decompressing..."
    rm -rf output/restored
    if ! $EXE -d output/test.$ALGO output/restored $ALGO $FLAGS; then
        echo "  FAILED: Decompression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Verify
    echo "  Verifying..."
    if ! cmp -s sample.txt output/restored/sample.txt; then
        echo "  FAILED: Files do not match"
        ((FAIL_COUNT++))
        return 1
    fi
    
    echo "  PASSED"
    ((PASS_COUNT++))
    return 0
}

# Helper function for cross-compatibility tests
run_cross_test() {
    local TEST_NAME=$1
    local ALGO=$2
    
    ((TEST_COUNT++))
    
    echo
    echo "[Test $TEST_COUNT] $TEST_NAME"
    
    # Compress with GPU
    echo "  Compressing with GPU..."
    if ! $EXE -c sample.txt output/test.$ALGO $ALGO; then
        echo "  FAILED: GPU Compression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Decompress with CPU
    echo "  Decompressing with CPU..."
    rm -rf output/restored
    if ! $EXE -d output/test.$ALGO output/restored $ALGO --cpu; then
        echo "  FAILED: CPU Decompression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Verify
    echo "  Verifying..."
    if ! cmp -s sample.txt output/restored/sample.txt; then
        echo "  FAILED: Files do not match"
        ((FAIL_COUNT++))
        return 1
    fi
    
    echo "  PASSED"
    ((PASS_COUNT++))
    return 0
}

# ============================================================================
# GPU Batched <-> GPU Batched (LZ4, Snappy, Zstd)
# ============================================================================

echo
echo "========================================"
echo "GPU Batched Tests"
echo "========================================"

run_test "GPU Batched LZ4" "lz4" "" || true
run_test "GPU Batched Snappy" "snappy" "" || true
run_test "GPU Batched Zstd" "zstd" "" || true

# ============================================================================
# GPU Manager <-> GPU Manager (GDeflate, ANS, Bitcomp)
# ============================================================================

echo
echo "========================================"
echo "GPU Manager Tests"
echo "========================================"

run_test "GPU Manager GDeflate" "gdeflate" "" || true
run_test "GPU Manager ANS" "ans" "" || true
run_test "GPU Manager Bitcomp" "bitcomp" "" || true

# ============================================================================
# CPU <-> CPU (LZ4, Snappy, Zstd)
# ============================================================================

echo
echo "========================================"
echo "CPU Tests"
echo "========================================"

run_test "CPU LZ4" "lz4" "--cpu" || true
run_test "CPU Snappy" "snappy" "--cpu" || true
run_test "CPU Zstd" "zstd" "--cpu" || true

# ============================================================================
# GPU Batched -> CPU (Cross-compatibility)
# ============================================================================

echo
echo "========================================"
echo "GPU to CPU Cross-compatibility Tests"
echo "========================================"

run_cross_test "GPU->CPU LZ4" "lz4" || true
run_cross_test "GPU->CPU Snappy" "snappy" || true
run_cross_test "GPU->CPU Zstd" "zstd" || true

# ============================================================================
# Summary
# ============================================================================

echo
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total tests: $TEST_COUNT"
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "========================================"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
else
    echo "All tests passed!"
    exit 0
fi

