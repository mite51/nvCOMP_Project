#!/bin/bash
# nvCOMP CLI Archive Listing Tests
# Tests for `nvcomp_cli -l`: streaming metadata listing without extraction
# (single-file, multi-volume, CPU decode, listing via a non-first volume).

echo "========================================"
echo "nvCOMP CLI Archive Listing Tests"
echo "========================================"
echo ""

TEST_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

mkdir -p output

EXE="../build/nvcomp_cli"
if [ ! -f "$EXE" ]; then
    echo "ERROR: nvcomp_cli not found. Build the project first."
    exit 1
fi

# ============================================================================
# Fixture tree: nested dirs, empty file, name with spaces, one multi-chunk
# file (>128KB so its data spans GPU sub-batch feeds).
# ============================================================================

SRC="output/list_src"
rm -rf "$SRC"
mkdir -p "$SRC/sub one/deep" "$SRC/sub two"
head -c 300000 /dev/urandom > "$SRC/big.bin"
seq 1 5000 > "$SRC/sub one/numbers.txt"
printf 'hello nvcomp' > "$SRC/sub one/deep/greeting.txt"
: > "$SRC/sub two/empty.dat"
EXPECTED_FILES=4

# Paths as stored in the archive (relative, forward slashes)
EXPECTED_PATHS=(
    "big.bin"
    "sub one/numbers.txt"
    "sub one/deep/greeting.txt"
    "sub two/empty.dat"
)

# ============================================================================
# Helper: run -l and validate paths + file-count footer
# ============================================================================

check_listing() {
    local TEST_NAME="$1"
    local ARCHIVE="$2"
    local EXTRA_FLAGS="$3"
    ((TEST_COUNT++))

    echo ""
    echo "[Test $TEST_COUNT] $TEST_NAME"

    local OUT
    OUT=$($EXE -l "$ARCHIVE" $EXTRA_FLAGS 2>&1)
    if [ $? -ne 0 ]; then
        echo "  FAILED: -l exited nonzero"
        echo "$OUT" | head -5
        ((FAIL_COUNT++))
        return
    fi

    for P in "${EXPECTED_PATHS[@]}"; do
        if ! echo "$OUT" | grep -qF "$P"; then
            echo "  FAILED: path missing from listing: $P"
            ((FAIL_COUNT++))
            return
        fi
    done

    if ! echo "$OUT" | grep -qE "Total: $EXPECTED_FILES file"; then
        echo "  FAILED: expected 'Total: $EXPECTED_FILES file(s)' footer"
        echo "$OUT" | tail -3
        ((FAIL_COUNT++))
        return
    fi

    # Spot-check a known size (numbers.txt: seq 1 5000)
    local NUM_SIZE
    NUM_SIZE=$(stat -c %s "$SRC/sub one/numbers.txt")
    local NUM_KB
    NUM_KB=$(awk "BEGIN{printf \"%.2f\", $NUM_SIZE/1024}")
    if ! echo "$OUT" | grep "numbers.txt" | grep -q "$NUM_KB KB"; then
        echo "  FAILED: numbers.txt size mismatch (expected $NUM_KB KB)"
        echo "$OUT" | grep "numbers.txt"
        ((FAIL_COUNT++))
        return
    fi

    echo "  PASSED"
    ((PASS_COUNT++))
}

# ============================================================================
# Tests
# ============================================================================

# Single-file archives, one per cross-compatible algorithm
for ALGO in lz4 zstd snappy; do
    rm -f "output/list_single.$ALGO"
    $EXE -c "$SRC" "output/list_single.$ALGO" $ALGO --no-volumes > /dev/null || {
        echo "ERROR: compression failed ($ALGO)"; exit 1; }
    check_listing "Single-file listing ($ALGO)" "output/list_single.$ALGO"
done

# CPU decode path
check_listing "Single-file listing (lz4, --cpu)" "output/list_single.lz4" "--cpu"

# Multi-volume archive (small volumes => several)
rm -f output/list_mv.vol*.lz4
$EXE -c "$SRC" "output/list_mv.lz4" lz4 --volume-size 100KB > /dev/null || {
    echo "ERROR: multi-volume compression failed"; exit 1; }
VOLS=$(ls output/list_mv.vol*.lz4 2>/dev/null | wc -l)
if [ "$VOLS" -lt 2 ]; then
    echo "ERROR: expected multiple volumes, found $VOLS"
    exit 1
fi
check_listing "Multi-volume listing (vol001)" "output/list_mv.vol001.lz4"

# Listing must work when pointed at any volume, not just the first
LAST_VOL=$(ls output/list_mv.vol*.lz4 | sort | tail -1)
check_listing "Multi-volume listing (last volume path)" "$LAST_VOL"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total tests: $TEST_COUNT"
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "========================================"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "ALL TESTS PASSED!"
    exit 0
else
    echo "SOME TESTS FAILED!"
    exit 1
fi
