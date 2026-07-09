#!/usr/bin/env python3
"""Deterministic benchmark dataset generator for the nvCOMP optimization pass.

Generates into bench/data/ (git-ignored):
  single/  tiny.txt edge_64k.bin edge_64k_plus1.bin small_text.txt small_random.bin
           med_text.bin med_random.bin med_mixed.bin large_mixed.bin
  tree_small/ tree_med/ tree_large/

All content is seeded (numpy PCG64) so datasets are reproducible across runs.
Idempotent: files/trees that already exist at the right size are skipped.
"""

import os
import sys
import numpy as np

MB = 1024 * 1024
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEED = 20260707

WRITE_BLOCK = 64 * MB


def log(msg):
    print(msg, flush=True)


def build_text_corpus(rng, size=4 * MB):
    """~4MB block of word-based text: compressible (~3-6x) but not degenerate."""
    try:
        with open("/usr/share/dict/words", "rb") as f:
            words = [w for w in f.read().split() if w.isalpha()]
    except OSError:
        words = [b"lorem", b"ipsum", b"dolor", b"sit", b"amet", b"consectetur",
                 b"adipiscing", b"elit", b"sed", b"do", b"eiusmod", b"tempor"]
    idx = rng.integers(0, len(words), size // 6)
    out = bytearray()
    line = bytearray()
    lineno = 0
    for i in idx:
        line += words[int(i)] + b" "
        if len(line) >= 72:
            out += b"%07d " % lineno + line + b"\n"
            lineno += 1
            if len(out) >= size:
                break
            line = bytearray()
    return bytes(out[:size])


def write_random(path, size, rng):
    if os.path.exists(path) and os.path.getsize(path) == size:
        log(f"  skip {path}")
        return
    with open(path, "wb") as f:
        left = size
        while left > 0:
            n = min(WRITE_BLOCK, left)
            f.write(rng.bytes(n))
            left -= n
    log(f"  wrote {path} ({size / MB:.1f} MB random)")


def write_text(path, size, corpus):
    if os.path.exists(path) and os.path.getsize(path) == size:
        log(f"  skip {path}")
        return
    with open(path, "wb") as f:
        left = size
        block_no = 0
        while left > 0:
            header = b"==== block %09d " % block_no + b"=" * 40 + b"\n"
            block = header + corpus
            n = min(len(block), left)
            f.write(block[:n])
            left -= n
            block_no += 1
    log(f"  wrote {path} ({size / MB:.1f} MB text)")


def write_mixed(path, size, corpus, rng, slice_mb=1):
    """Alternate 1MB text and 1MB random slices -> realistic ~1.5-2.5x ratio."""
    if os.path.exists(path) and os.path.getsize(path) == size:
        log(f"  skip {path}")
        return
    s = slice_mb * MB
    text_block = (corpus * ((s // len(corpus)) + 2))
    with open(path, "wb") as f:
        left = size
        i = 0
        while left > 0:
            if i % 2 == 0:
                off = (i // 2 * 4096) % len(corpus)
                chunk = text_block[off:off + s]
            else:
                chunk = rng.bytes(s)
            n = min(len(chunk), left)
            f.write(chunk[:n])
            left -= n
            i += 1
    log(f"  wrote {path} ({size / MB:.1f} MB mixed)")


def make_tree(root, spec, corpus, rng):
    """spec: list of (count, min_size, max_size, kind) where kind in text|random|mixed."""
    marker = os.path.join(root, ".complete")
    if os.path.exists(marker):
        log(f"  skip {root}")
        return
    os.makedirs(root, exist_ok=True)
    dirs = [root]
    for d in ["docs", "src", "src/lib", "assets", "assets/blobs", "logs"]:
        p = os.path.join(root, d)
        os.makedirs(p, exist_ok=True)
        dirs.append(p)
    text_pool = corpus * 4  # 16MB
    rand_pool = rng.bytes(16 * MB)
    fid = 0
    total = 0
    for count, lo, hi, kind in spec:
        for _ in range(count):
            size = int(rng.integers(lo, hi + 1))
            d = dirs[int(rng.integers(0, len(dirs)))]
            ext = {"text": ".txt", "random": ".bin", "mixed": ".dat"}[kind]
            path = os.path.join(d, f"f{fid:05d}{ext}")
            fid += 1
            with open(path, "wb") as f:
                left = size
                while left > 0:
                    if kind == "text":
                        pool = text_pool
                    elif kind == "random":
                        pool = rand_pool
                    else:
                        pool = text_pool if (left // MB) % 2 == 0 else rand_pool
                    off = int(rng.integers(0, len(pool) - MB))
                    n = min(MB, left, len(pool) - off)
                    f.write(pool[off:off + n])
                    left -= n
            total += size
    open(marker, "w").write("ok\n")
    log(f"  wrote {root} ({fid} files, {total / MB:.0f} MB)")


def main():
    os.makedirs(os.path.join(DATA, "single"), exist_ok=True)
    rng = np.random.default_rng(SEED)
    corpus = build_text_corpus(rng)
    single = os.path.join(DATA, "single")

    log("== single files ==")
    p = os.path.join(single, "tiny.txt")
    if not os.path.exists(p):
        open(p, "wb").write(b"tiny data\n")
    write_mixed(os.path.join(single, "edge_64k.bin"), 4 * MB, corpus, rng)
    write_mixed(os.path.join(single, "edge_64k_plus1.bin"), 4 * MB + 1, corpus, rng)
    write_text(os.path.join(single, "small_text.txt"), 8 * MB, corpus)
    write_random(os.path.join(single, "small_random.bin"), 8 * MB, rng)
    write_text(os.path.join(single, "med_text.bin"), 512 * MB, corpus)
    write_random(os.path.join(single, "med_random.bin"), 512 * MB, rng)
    write_mixed(os.path.join(single, "med_mixed.bin"), 512 * MB, corpus, rng)
    write_mixed(os.path.join(single, "large_mixed.bin"), 6 * 1024 * MB, corpus, rng)

    log("== trees ==")
    make_tree(os.path.join(DATA, "tree_small"),
              [(40, 1024, 256 * 1024, "text"), (20, 4096, MB, "random")],
              corpus, rng)
    make_tree(os.path.join(DATA, "tree_med"),
              [(1200, 10 * 1024, 500 * 1024, "text"),
               (600, 100 * 1024, 2 * MB, "mixed"),
               (200, 500 * 1024, 5 * MB, "random")],
              corpus, rng)
    make_tree(os.path.join(DATA, "tree_large"),
              [(8000, 4 * 1024, 64 * 1024, "text"),
               (14, 64 * MB, 256 * MB, "mixed"),
               (6, 256 * MB, 512 * MB, "random")],
              corpus, rng)
    log("done")


if __name__ == "__main__":
    sys.exit(main())
