#!/usr/bin/env bash
# Benchmark harness for the nvCOMP optimization pass.
#
# Usage:
#   bench/bench.sh [--quick] [--label <name>] [--exe <path>] [--algos "lz4 zstd"]
#
# Per run it records: wall clock (compress + decompress), the tool's own phase
# stats (read/prepare/compute/write), peak host RSS (/usr/bin/time -v), peak
# process VRAM (nvidia-smi polling), round-trip verification, into a CSV under
# bench/results/.
set -u

cd "$(dirname "$0")/.."
ROOT=$PWD
DATA=$ROOT/bench/data
RESULTS=$ROOT/bench/results
SCRATCH=$ROOT/bench/scratch
EXE=$ROOT/build/nvcomp_cli
LABEL="run"
QUICK=0
ALGOS="lz4 snappy zstd gdeflate"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK=1; shift;;
        --label) LABEL=$2; shift 2;;
        --exe)   EXE=$2; shift 2;;
        --algos) ALGOS=$2; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

[[ -x $EXE ]] || { echo "missing exe: $EXE" >&2; exit 1; }
mkdir -p "$RESULTS" "$SCRATCH"
CSV=$RESULTS/${LABEL}_$(date +%Y%m%d_%H%M%S).csv
echo "label,algo,asset,mode,op,ok,wall_s,read_s,prepare_s,compute_s,write_s,speed_mbps,ratio,peak_rss_mb,peak_vram_mb" > "$CSV"

# --- helpers -----------------------------------------------------------------

# Poll peak VRAM used by compute apps while a command runs.
# run_timed <logprefix> <cmd...>  -> sets WALL, RSS_MB, VRAM_MB, RC, LOG
run_timed() {
    local prefix=$1; shift
    LOG=$SCRATCH/$prefix.log
    local vramlog=$SCRATCH/$prefix.vram
    : > "$vramlog"
    (
        while :; do
            nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1
            sleep 0.1
        done >> "$vramlog"
    ) &
    local poller=$!
    /usr/bin/time -v -o "$SCRATCH/$prefix.time" "$@" > "$LOG" 2>&1
    RC=$?
    kill $poller 2>/dev/null; wait $poller 2>/dev/null
    WALL=$(awk -F': ' '/Elapsed \(wall clock\)/{split($2,t,":");
        if (length(t)==3) print t[1]*3600+t[2]*60+t[3]; else print t[1]*60+t[2]}' "$SCRATCH/$prefix.time")
    RSS_MB=$(awk -F': ' '/Maximum resident set size/{printf "%.0f", $2/1024}' "$SCRATCH/$prefix.time")
    VRAM_MB=$(sort -n "$vramlog" 2>/dev/null | tail -1)
    VRAM_MB=${VRAM_MB:-0}
    rm -f "$vramlog"
}

# Parse the tool's stats block from $LOG -> READ_S PREP_S COMP_S WRITE_S SPEED RATIO
parse_stats() {
    READ_S=$(awk '/^  Read /{print $3}' "$LOG" | tail -1);   READ_S=${READ_S:-}
    PREP_S=$(awk '/^  Prepare /{print $3}' "$LOG" | tail -1); PREP_S=${PREP_S:-}
    COMP_S=$(awk '/^  Compute /{print $3}' "$LOG" | tail -1); COMP_S=${COMP_S:-}
    WRITE_S=$(awk '/^  Write /{print $3}' "$LOG" | tail -1);  WRITE_S=${WRITE_S:-}
    SPEED=$(awk '/^  Speed /{print $3}' "$LOG" | tail -1);    SPEED=${SPEED:-}
    RATIO=$(awk '/^  Ratio /{sub("x","",$3); print $3}' "$LOG" | tail -1); RATIO=${RATIO:-}
}

# verify <src> <extracted_dir>
verify() {
    local src=$1 out=$2
    if [[ -d $src ]]; then
        # folder archives extract their contents directly into the target dir
        diff -rq "$src" "$out" > /dev/null 2>&1
    else
        cmp -s "$src" "$out/$(basename "$src")"
    fi
}

bench_one() {
    local algo=$1 asset=$2 mode=$3 extra=$4
    local name=$(basename "$asset")
    local arch=$SCRATCH/${name}.${algo}
    local outdir=$SCRATCH/extract_${name}_${algo}
    rm -rf "$arch" "$SCRATCH/${name}".vol*."$algo" "$outdir"

    echo "--- $algo / $name / $mode"

    run_timed "c_${name}_${algo}" "$EXE" -c "$asset" "$arch" "$algo" $extra
    local cwall=$WALL crss=$RSS_MB cvram=$VRAM_MB crc=$RC
    parse_stats
    local cok=ok; [[ $crc -ne 0 ]] && cok=FAIL
    echo "$LABEL,$algo,$name,$mode,compress,$cok,$cwall,$READ_S,$PREP_S,$COMP_S,$WRITE_S,$SPEED,$RATIO,$crss,$cvram" >> "$CSV"

    # decompress from vol001 if volumes were created
    local dinput=$arch
    [[ -f ${arch%.$algo}.vol001.$algo ]] && dinput=${arch%.$algo}.vol001.$algo
    if [[ $crc -eq 0 && -f $dinput ]]; then
        run_timed "d_${name}_${algo}" "$EXE" -d "$dinput" "$outdir" "$algo"
        local dok=FAIL
        [[ $RC -eq 0 ]] && verify "$asset" "$outdir" && dok=ok
        parse_stats
        echo "$LABEL,$algo,$name,$mode,decompress,$dok,$WALL,$READ_S,$PREP_S,$COMP_S,$WRITE_S,$SPEED,$RATIO,$RSS_MB,$VRAM_MB" >> "$CSV"
        [[ $dok == FAIL ]] && echo "!!! round-trip FAILED: $algo $name (log: $LOG)"
    else
        echo "$LABEL,$algo,$name,$mode,decompress,SKIP,,,,,,,,," >> "$CSV"
        [[ $crc -ne 0 ]] && echo "!!! compress FAILED: $algo $name (log: $SCRATCH/c_${name}_${algo}.log)"
    fi
    rm -rf "$arch" "$SCRATCH/${name}".vol*."$algo" "$outdir"
}

# --- matrix ------------------------------------------------------------------

if [[ $QUICK -eq 1 ]]; then
    ASSETS="$DATA/single/med_mixed.bin $DATA/tree_small"
else
    ASSETS="$DATA/single/med_text.bin $DATA/single/med_random.bin $DATA/single/med_mixed.bin $DATA/single/large_mixed.bin $DATA/tree_med $DATA/tree_large"
fi

for algo in $ALGOS; do
    for asset in $ASSETS; do
        [[ -e $asset ]] || { echo "missing asset: $asset (run gen_dataset.py)"; continue; }
        bench_one "$algo" "$asset" "default" ""
    done
done

echo
echo "results: $CSV"
column -t -s, "$CSV"
