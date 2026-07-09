#!/usr/bin/env bash
# Archive v2 metadata round-trip: POSIX permissions (incl. executable bit) and
# modification times must survive compress -> decompress, across the GPU
# pipeline, the CPU path, multi-volume splits, and 1MB-feed boundary splits.
set -u
cd "$(dirname "$0")"
EXE=${EXE:-../build/nvcomp_cli}
WORK=output/meta_test
PASS=0; FAIL=0

make_tree() {
    local big=${1:-0}
    rm -rf "$WORK/src"
    mkdir -p "$WORK/src/bin" "$WORK/src/cfg"
    if [[ $big == 1 ]]; then
        # Force a multi-volume split at --volume-size 100MB; mid-file split
        # exercises metadata on a file spanning volumes.
        head -c 250000000 /dev/urandom > "$WORK/src/huge.dat"
        chmod 751 "$WORK/src/huge.dat"
        touch -d "2023-01-02 03:04:05" "$WORK/src/huge.dat"
    fi
    printf '#!/bin/sh\necho hi\n' > "$WORK/src/bin/run.sh";  chmod 755 "$WORK/src/bin/run.sh"
    head -c 300000 /dev/urandom  > "$WORK/src/bin/tool";     chmod 700 "$WORK/src/bin/tool"
    printf 'secret\n'            > "$WORK/src/cfg/private";  chmod 600 "$WORK/src/cfg/private"
    printf 'readonly\n'          > "$WORK/src/cfg/frozen";   chmod 444 "$WORK/src/cfg/frozen"
    : > "$WORK/src/cfg/empty";                               chmod 640 "$WORK/src/cfg/empty"
    # File large enough to span 1MB feeds in the stressor run.
    head -c 3000000 /dev/urandom > "$WORK/src/big.dat";      chmod 664 "$WORK/src/big.dat"
    touch -d "2024-03-15 12:34:56" "$WORK/src/bin/run.sh" "$WORK/src/cfg/frozen" "$WORK/src/big.dat"
}

# meta <dir> -> "relpath mode mtime" lines, sorted
meta() {
    (cd "$1" && find . -type f -exec stat -c "%n %a %Y" {} \; | sort)
}

check() {
    local label=$1 algo=$2 big=$3; shift 3
    make_tree "$big"
    local exp; exp=$(meta "$WORK/src")
    rm -rf "$WORK/arch"* "$WORK/out"
    if ! "$EXE" -c "$WORK/src" "$WORK/arch.$algo" "$algo" "$@" > "$WORK/c.log" 2>&1; then
        echo "FAIL ($label): compress error"; FAIL=$((FAIL+1)); return
    fi
    local input="$WORK/arch.$algo"
    [[ -f "$WORK/arch.vol001.$algo" ]] && input="$WORK/arch.vol001.$algo"
    if ! "$EXE" -d "$input" "$WORK/out" "$algo" > "$WORK/d.log" 2>&1; then
        echo "FAIL ($label): decompress error"; FAIL=$((FAIL+1)); return
    fi
    local got; got=$(meta "$WORK/out")
    if ! diff -rq "$WORK/src" "$WORK/out" > /dev/null 2>&1; then
        echo "FAIL ($label): content mismatch"; FAIL=$((FAIL+1)); return
    fi
    if [[ "$exp" == "$got" ]]; then
        echo "PASS: $label"; PASS=$((PASS+1))
    else
        echo "FAIL ($label): metadata mismatch"
        diff <(echo "$exp") <(echo "$got") | head -10
        FAIL=$((FAIL+1))
    fi
}

mkdir -p "$WORK"
check "gpu-zstd"                 zstd 0
check "gpu-lz4"                  lz4 0
check "cpu-zstd"                 zstd 0 --cpu
check "gpu-manager-gdeflate"     gdeflate 0
NVCOMP_SUBBATCH_MB=1 check "gpu-zstd-1MB-feeds" zstd 0
check "gpu-multi-volume"         lz4 1 --volume-size 100MB

rm -rf "$WORK"
echo "---"
echo "metadata round-trip: $PASS passed, $FAIL failed"
exit $((FAIL > 0))
