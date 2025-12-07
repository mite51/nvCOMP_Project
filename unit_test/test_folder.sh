#!/bin/bash
# nvCOMP CLI Folder Compression Tests
# Tests for folder compression feature and original zstd decompression issue

set -e

echo "========================================"
echo "nvCOMP CLI Folder & Archive Tests"
echo "========================================"
echo

TEST_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

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

# Verify sample_folder exists
if [ ! -d "sample_folder" ]; then
    echo "ERROR: sample_folder not found. Create sample_folder directory with test files."
    exit 1
fi

# Helper function to run folder tests
run_folder_test() {
    local TEST_NAME=$1
    local ALGO=$2
    local FLAGS=$3
    
    ((TEST_COUNT++))
    
    echo
    echo "[Test $TEST_COUNT] $TEST_NAME"
    
    # Compress folder
    echo "  Compressing folder..."
    if ! $EXE -c sample_folder output/folder.$ALGO $ALGO $FLAGS; then
        echo "  FAILED: Folder compression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # List archive
    echo "  Listing archive..."
    if ! $EXE -l output/folder.$ALGO $ALGO $FLAGS; then
        echo "  FAILED: Archive listing failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Decompress folder
    echo "  Decompressing folder..."
    rm -rf output/folder_restored_$ALGO
    if ! $EXE -d output/folder.$ALGO output/folder_restored_$ALGO $ALGO $FLAGS; then
        echo "  FAILED: Folder decompression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Verify folder structure
    echo "  Verifying folder structure..."
    if [ ! -d "output/folder_restored_$ALGO/PineTools.com_files" ]; then
        echo "  FAILED: Folder structure not preserved"
        ((FAIL_COUNT++))
        return 1
    fi
    
    echo "  PASSED"
    ((PASS_COUNT++))
    return 0
}

# Helper function for listing tests
run_list_test() {
    local TEST_NAME=$1
    local ALGO=$2
    local FLAGS=$3
    
    ((TEST_COUNT++))
    
    echo
    echo "[Test $TEST_COUNT] $TEST_NAME"
    
    # Compress folder
    echo "  Compressing folder for listing test..."
    if ! $EXE -c sample_folder output/list_test.$ALGO $ALGO $FLAGS; then
        echo "  FAILED: Compression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # List archive
    echo "  Listing archive contents..."
    if ! $EXE -l output/list_test.$ALGO $ALGO $FLAGS; then
        echo "  FAILED: Archive listing failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    echo "  PASSED"
    ((PASS_COUNT++))
    return 0
}

# Helper function for round-trip test
run_roundtrip_test() {
    local TEST_NAME=$1
    local ALGO=$2
    
    ((TEST_COUNT++))
    
    echo
    echo "[Test $TEST_COUNT] $TEST_NAME"
    echo "  This test reproduces the original issue:"
    echo '  "Using CPU decompression (zstd)..."'
    echo '  "Error: Zstd CPU decompression failed"'
    echo
    
    # Step 1: Compress with GPU
    echo "  Step 1: Compressing folder with GPU..."
    if ! $EXE -c sample_folder output/roundtrip.$ALGO $ALGO; then
        echo "  FAILED: GPU Compression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Step 2: List (this was failing before)
    echo "  Step 2: Listing archive (this was failing before)..."
    if ! $EXE -l output/roundtrip.$ALGO $ALGO; then
        echo "  FAILED: Archive listing failed - ORIGINAL ISSUE REPRODUCED!"
        echo "  This is the error that was reported."
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Step 3: Decompress (this was also failing)
    echo "  Step 3: Decompressing (this was also failing)..."
    rm -rf output/roundtrip_restored
    if ! $EXE -d output/roundtrip.$ALGO output/roundtrip_restored $ALGO; then
        echo "  FAILED: Decompression failed - ORIGINAL ISSUE REPRODUCED!"
        echo "  This is the error that was reported."
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Step 4: Verify
    echo "  Step 4: Verifying decompressed folder..."
    if [ ! -d "output/roundtrip_restored/PineTools.com_files" ]; then
        echo "  FAILED: Folder structure not preserved"
        ((FAIL_COUNT++))
        return 1
    fi
    
    echo "  PASSED - Original issue FIXED!"
    echo "  GPU zstd compression now works correctly with listing and decompression."
    ((PASS_COUNT++))
    return 0
}

# Helper function for auto-detection listing tests
run_autodetect_list_test() {
    local TEST_NAME=$1
    local ALGO=$2
    local FLAGS=$3
    
    ((TEST_COUNT++))
    
    echo
    echo "[Test $TEST_COUNT] $TEST_NAME"
    
    # Compress folder
    echo "  Compressing folder..."
    if ! $EXE -c sample_folder output/autodetect.$ALGO $ALGO $FLAGS; then
        echo "  FAILED: Compression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # List archive WITHOUT algorithm parameter
    echo "  Listing archive WITHOUT algorithm parameter..."
    if ! $EXE -l output/autodetect.$ALGO $FLAGS; then
        echo "  FAILED: Auto-detection listing failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    echo "  PASSED - Algorithm auto-detected successfully!"
    ((PASS_COUNT++))
    return 0
}

# Helper function for auto-detection decompression tests
run_autodetect_decompress_test() {
    local TEST_NAME=$1
    local ALGO=$2
    local FLAGS=$3
    
    ((TEST_COUNT++))
    
    echo
    echo "[Test $TEST_COUNT] $TEST_NAME"
    
    # Compress folder
    echo "  Compressing folder..."
    if ! $EXE -c sample_folder output/autodetect.$ALGO $ALGO $FLAGS; then
        echo "  FAILED: Compression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Decompress WITHOUT algorithm parameter
    echo "  Decompressing WITHOUT algorithm parameter..."
    rm -rf output/autodetect_restored
    if ! $EXE -d output/autodetect.$ALGO output/autodetect_restored $FLAGS; then
        echo "  FAILED: Auto-detection decompression failed"
        ((FAIL_COUNT++))
        return 1
    fi
    
    # Verify folder structure
    echo "  Verifying folder structure..."
    if [ ! -d "output/autodetect_restored/PineTools.com_files" ]; then
        echo "  FAILED: Folder structure not preserved"
        ((FAIL_COUNT++))
        return 1
    fi
    
    echo "  PASSED - Algorithm auto-detected and decompressed successfully!"
    ((PASS_COUNT++))
    return 0
}

# ============================================================================
# Folder Compression Tests
# ============================================================================

echo
echo "========================================"
echo "Folder Compression Tests"
echo "========================================"

run_folder_test "Folder GPU LZ4" "lz4" "" || true
run_folder_test "Folder GPU Zstd" "zstd" "" || true
run_folder_test "Folder CPU LZ4" "lz4" "--cpu" || true
run_folder_test "Folder CPU Zstd" "zstd" "--cpu" || true

# ============================================================================
# Archive Listing Tests
# ============================================================================

echo
echo "========================================"
echo "Archive Listing Tests"
echo "========================================"

run_list_test "List GPU Zstd Archive" "zstd" "" || true
run_list_test "List GPU LZ4 Archive" "lz4" "" || true
run_list_test "List CPU Zstd Archive" "zstd" "--cpu" || true

# ============================================================================
# GPU Zstd Round-Trip Test (Original Issue)
# ============================================================================

echo
echo "========================================"
echo "GPU Zstd Round-Trip Test"
echo "(Tests the original decompression issue)"
echo "========================================"

run_roundtrip_test "GPU Zstd Full Round-Trip" "zstd" || true

# ============================================================================
# Algorithm Auto-Detection Tests
# ============================================================================

echo
echo "========================================"
echo "Algorithm Auto-Detection Tests"
echo "========================================"

run_autodetect_list_test "Auto-detect LZ4 (List)" "lz4" "" || true
run_autodetect_list_test "Auto-detect Snappy (List)" "snappy" "" || true
run_autodetect_list_test "Auto-detect Zstd (List)" "zstd" "" || true
run_autodetect_decompress_test "Auto-detect LZ4 (Decompress)" "lz4" "" || true
run_autodetect_decompress_test "Auto-detect Zstd (Decompress)" "zstd" "" || true
run_autodetect_decompress_test "Auto-detect LZ4 CPU (Decompress)" "lz4" "--cpu" || true

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
    echo
    echo "SOME TESTS FAILED!"
    exit 1
else
    echo
    echo "ALL TESTS PASSED!"
    exit 0
fi

