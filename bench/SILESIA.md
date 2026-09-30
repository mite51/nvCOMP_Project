# Silesia SDK comparison

`silesia.py` benchmarks two nvCOMP CLI builds on the
[Silesia compression corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia).
The corpus contains 12 real files totaling 211,938,580 bytes: text, HTML/XML,
source code, executables, databases, a PDF and medical images. The files are
historical; this is a varied, reproducible workload rather than a representative
sample of every modern application.

See [the recorded 5.1 vs 5.3 results](SILESIA_RESULTS.md) for the Windows RTX 4090 run.

## Run

Requires Python 3.9+, two built CLI executables, and a working NVIDIA GPU/driver.
Each executable must stay beside its own matching core and SDK runtime libraries.
The script uses only the Python standard library and works on Windows and Linux.

From the repository root, using the preserved upgrade baseline:

```powershell
python bench/silesia.py --download --baseline output/nvcomp51-baseline/nvcomp_cli.exe --candidate out/nvcomp53/Release/nvcomp_cli.exe
```

Use your actual binary paths on Linux. To download and verify the corpus without
running benchmarks:

```powershell
python bench/silesia.py --download --prepare-only
```

To run only the combined folder workload:

```powershell
python bench/silesia.py --baseline output/nvcomp51-baseline/nvcomp_cli.exe --candidate out/nvcomp53/Release/nvcomp_cli.exe --scope folder
```

Other options: `--algorithms lz4 zstd`, `--repeats 5`, `--warmups 1`,
`--timeout 60`, `--data-dir PATH`, and `--results-dir PATH`. Defaults test all
six algorithms on each of the 12 files and the complete folder: 78 cases,
936 round trips including warm-ups, and 1,872 CLI process invocations.

## Method

- Download the author's ZIP. Check its recorded SHA-256, then verify all 12
  uncompressed sizes and the MD5 checksums published on the corpus page. Record
  SHA-256 hashes of the verified inputs in the run metadata.
- Keep the original files and bytes. The folder case uses the application's
  normal archive packing; it does not repeat or manufacture extra data.
- Keep the normal algorithm settings, 64 KiB chunks and `--no-volumes` for both
  builds. The CPU codec dependencies and application code are the same in the
  preserved 5.1 and upgraded 5.3 builds.
- Shuffle case order with a fixed seed. Alternate which build runs first in each
  pair. Run one warm-up and five measured repetitions per build/case, sequentially
  to avoid the two builds competing for GPU or disk resources.
- Measure complete CLI elapsed time, including process startup, GPU setup,
  transfers, disk I/O and shutdown. Record the application's phase timings as
  supplementary data; these are rounded and are not isolated kernel timings.
- Verify the exact extracted filenames and SHA-256 contents after every run,
  outside the timed section. Fresh extraction directories prevent stale outputs
  from passing verification. Compressed size includes the application's headers.
- Use medians of successful measured runs, with minimum/maximum and success
  counts. A decompressor crash is a failed run even if it wrote correct output.
  Such a run can still establish compression size/time when compression exited
  successfully and extracted content verified; it cannot establish a successful
  decompression time. The overall command exits nonzero if any process or content
  check fails, including warm-ups.

This is a warm-cache application benchmark: it does not flush operating-system
caches or measure peak memory. Small differences may reflect CPU/disk/desktop GPU
activity. For a file-by-file corpus aggregate, sum median times and compressed
bytes, then divide total input bytes by those sums; do not average per-file ratios
or throughput. The combined-folder measurement is a separate workload.

## Outputs

Every invocation creates a unique `bench/results/silesia_<UTC timestamp>/` with:

- `metadata.json`: corpus hashes, executable/runtime hashes, GPU/driver details,
  environment overrides and invocation settings.
- `samples.jsonl`: every warm-up and measured result, exit codes, verification
  results, elapsed times, sizes and reported phase timings.
- `summary.json` and `summary.csv`: per-file and folder medians, ranges, ratios,
  throughput and success counts for each build.
- `logs/`: complete compression and decompression process output.

Generated corpus and results are ignored by Git. Temporary archives/extractions
are removed after the run; the source corpus, logs and measurements are retained.
