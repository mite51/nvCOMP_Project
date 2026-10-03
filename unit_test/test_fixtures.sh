#!/usr/bin/env bash
# Backward-compatibility gate: archives created by the PRE-optimization binary
# (unit_test/fixtures/, supplied separately) must decompress
# byte-exactly with the current build.
set -u
cd "$(dirname "$0")/.."
EXE=${EXE:-build/nvcomp_cli}
FIX=${FIX:-unit_test/fixtures}
OUT=unit_test/output/fixture_check
PASS=0; FAIL=0

[[ -d $FIX ]] || { echo "no fixtures at $FIX (supply pre-release archives or set FIX)"; exit 1; }
FIX=$(cd "$FIX" && pwd)
mkdir -p "$OUT"

check_tree() {
    local file=$1 algo=$2 flags=${3:-}
    local dir=$OUT/$(basename "$file" | tr . _)
    rm -rf "$dir"
    if ! "$EXE" -d "$FIX/$file" "$dir" "$algo" $flags > "$dir.log" 2>&1; then
        echo "FAIL (decompress error): $file"; FAIL=$((FAIL+1)); return
    fi
    if (cd "$dir" && sha256sum -c "$FIX/manifest_tree.sha256" --quiet 2>/dev/null); then
        echo "PASS: $file"; PASS=$((PASS+1))
    else
        echo "FAIL (content mismatch): $file"; FAIL=$((FAIL+1))
    fi
    rm -rf "$dir"
}

for a in lz4 snappy zstd; do
    check_tree fixture_gpu_tree.$a $a
    check_tree fixture_cpu_tree.$a $a
    check_tree fixture_gpu_tree.$a $a --cpu   # GPU-made archive on CPU path
done
for a in gdeflate ans bitcomp; do
    check_tree fixture_mgr_tree.$a $a
done

# multi-volume
dir=$OUT/mv; rm -rf "$dir"
if "$EXE" -d "$FIX/fixture_mv.vol001.lz4" "$dir" lz4 > "$OUT/mv.log" 2>&1 \
   && [[ "$(sha256sum "$dir/med_mixed.bin" | cut -d' ' -f1)" == "$(cut -d' ' -f1 "$FIX/manifest_mv.sha256")" ]]; then
    echo "PASS: fixture_mv (4 volumes)"; PASS=$((PASS+1))
else
    echo "FAIL: fixture_mv"; FAIL=$((FAIL+1))
fi
rm -rf "$dir"

echo "---"
echo "fixture compat: $PASS passed, $FAIL failed"
exit $((FAIL > 0))
