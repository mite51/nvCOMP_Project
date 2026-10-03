# Compression investigation and performance improvements

Session: October 2–3, 2026. This is the consolidated record of the investigation,
CPU comparisons, implemented improvements and unsuccessful experiments. The
benchmark scripts, datasets, raw result dumps and separate reports formerly in
this directory were removed during the requested cleanup. This summary records
historical measurements; it is not a reproducible benchmark harness.

## Outcome and retained changes

The application became substantially faster, but **CPU zstd remained faster for
the tested end-to-end workloads**. GPU kernel throughput alone does not include
input reading, allocation, transfers, packing and writing. The retained changes
are:

1. **Bulk binary input reads.** Replace the streaming compressor's large-file
   `std::ifstream` reads and the shared small-file/mmap fallback with bulk
   `fread`, capped at 64 MiB per call. On the tested MSVC library, the old path
   split reads into 4,095-byte library calls. Native wide Windows filenames,
   failed-open handling and short-read errors are preserved.
2. **Separate completion and writer workers.** The caller submits GPU work;
   reader, completion and ordered writer threads overlap the remaining stages.
   They share at most three reusable slots. Single-slot jobs use just the caller
   and reader. Callbacks stay on the caller thread, and failures stop/join all
   workers, drain outstanding transfers and clean up partial output.
3. **Stream the first volume directly to disk.** Reserve its manifest/table
   prefix and patch it after writing, avoiding two large whole-volume RAM
   buffers. Serialized volume sizes now include the first-volume prefix.
4. **Optional reuse between jobs.** `NVCOMP_REUSE_BUFFERS=1`, set before jobs
   start, retains one matching workspace in the same process. It helped repeated
   small jobs, so it remains opt-in; fresh CLI runs cannot share allocations.

Default Zstd batches remain **64 MiB**, divided into 64 KiB chunks, with a
maximum of **three slots** (reduced for small inputs or insufficient VRAM).
Each slot owns host/device buffers, scratch space, a CUDA stream and a completion
event. Host worker count, GPU streams and kernel launch count are different
quantities. Three slots bound batches in flight; they do not establish how many
kernels execute simultaneously. Codec settings and archive versions are unchanged.

## Measurement conditions

- Windows 11, i9-12900K (16 physical/24 logical cores), RTX 4090 (24 GiB),
  driver 591.86, about 96 GiB RAM; nvCOMP 5.3.0.16, CUDA 13.0.88,
  MSVC 19.38.33145 and CMake 4.0.2.
- Native Windows zstd 1.5.7 and 7-Zip Extra 26.03. `zstd -T0` selected 16
  compression workers; 7-Zip used multithreaded LZMA2 (`-mmt=on`, `-ms=on`).
- Silesia: 12 verified files totaling 211,938,580 bytes. Generated mixed file:
  6,442,450,944 bytes, SHA-256
  `16a6889e45b3611d120e5de0d35c45f9361ca34304dba81124b097b7619e2290`.
  The historical Isaac Sim dataset was unavailable; these are not reruns of it.
- Competitors ran sequentially on the same machine/filesystem. Fresh-process
  timings include startup and I/O. Folder zstd timings include tar packing.
  Writes are buffered, not timed through durable storage completion. Normal
  desktop activity and occasional write delays remain sources of variation.
- Every completed benchmark archive was extracted and content-hash checked
  outside compression timing. CPU time and host RSS were recorded separately.
  GPU allocation totals below exclude context/driver memory; total process VRAM
  was unavailable under WDDM. Host RAM and VRAM must not be conflated.

## Measured improvement at each stage

Each row is its own same-session comparison: three measured runs after one
warm-up, using the 6 GiB input and 512 MiB logical volumes. Do not chain timings
from different campaigns into a single controlled speedup claim.

| Change | Before, median s | After, median s | CPU zstd -T0 -1 in that comparison |
|---|---:|---:|---:|
| Bulk input reads | 8.687 | 5.651 | 2.113 |
| Separate completion/writer workers | 5.575 | 2.776 | 1.938 |
| Stream first volume | 2.860 | 2.510 | 2.014 |

For the final writer comparison, GPU times ranged from 2.509–3.031 s and CPU
zstd from 1.999–2.100 s. The all-core CPU comparison was run immediately after
the GPU sweep: an initial accidental `-T1` case was retained in the investigation
but excluded from the all-core comparison. The single-thread CPU result was not
used to claim a GPU advantage.

Streaming the first volume reduced median sampled peak host RSS from **1.10 GiB
to 0.50 GiB**. Silesia's fresh-process change was small: 0.434 to 0.418 s, with
overlapping ranges; CPU zstd took 0.234 s. Output ratios stayed approximately
1.656x GPU / 1.673x CPU for the large file and 2.870x / 2.891x for Silesia.
Earlier campaigns included slow baseline trials (16.386 s for bulk-read testing,
23.793 s for worker testing); those were retained rather than discarded.

## Reusing buffers between jobs

These are repeated C API calls in a resident process, after one cold call,
using the same new writer with reuse disabled/enabled. They exclude process
startup and DLL loading and must not be compared directly with fresh CLI times.

| Input | Reallocate each job, median [min, max] s | Reuse, median [min, max] s |
|---|---:|---:|
| Silesia | 0.244 [0.242, 0.244] | 0.116 [0.112, 0.122] |
| 6 GiB mixed file | 2.771 [2.745, 2.924] | 2.701 [2.584, 2.786] |

Repeated small jobs improved by **52%**. The large-file ranges overlapped, so
the 2.5% median difference is not a dependable large-file improvement.
Diagnostic calls reduced warm setup from 0.106 s to about 0.000034 s.

The default Zstd workspace accounts for **2.42 GiB of GPU buffers and 384 MiB
of host buffers** on this SDK. Reuse keeps this memory allocated between jobs;
it reduces allocation overhead rather than the working set. At most one idle
workspace is cached process-wide, bounded to 4 GiB device and 1 GiB host memory.
Concurrent jobs exclusively own their workspaces. A mismatch releases the old
workspace before sizing against free VRAM; failed jobs discard their buffers.

Release idle buffers with `nvcomp_core::clearCompressionBufferCache()` or
`nvcomp_clear_compression_buffer_cache()` before CUDA reset/library unload or
when the memory is needed elsewhere. Active jobs continue, but their existing
leases cannot repopulate a cleared cache. Subsequent jobs may cache again.
`compressionBufferCacheDeviceBytes()` reports known idle device allocations.

## Experiments not retained

At fixed 64 MiB batches, the same executable tested 3/6/8/12 slots on the 6 GiB
input, with reuse off, event tracing off and randomized order. Three measured
runs followed one warm-up per setting. Actual allocation depth was checked.

| Slots | Wall time, median [min, max] s | GPU buffers GiB | Host buffers MiB |
|---|---:|---:|---:|
| 3 | **2.688 [2.660, 2.893]** | 2.42 | 384 |
| 6 | 2.818 [2.803, 2.824] | 4.85 | 768 |
| 8 | 2.904 [2.885, 3.065] | 6.46 | 1,024 |
| 12 | 3.139 [3.075, 9.949] | 9.70 | 1,536 |

Pipeline medians stayed around 2.4 s, while median time outside the pipeline
rose from 0.304 to 0.427/0.532/0.698 s. The 9.949 s trial spent 9.190 s in the
writer; the measurements do not identify the underlying cause of that delay.
More slots did not establish a throughput benefit and consumed more memory.
The slot-count override and its configuration reporting were therefore removed.
Earlier batch-size sweeps also did not justify changing the 64 MiB default.

The temporary CUDA event profiler, Manager allocation instrumentation, benchmark
harnesses and standalone investigation prototypes were removed at wrap-up.
Existing production controls unrelated to these experiments remain unchanged.

## Fair CPU baselines (before these optimizations)

Initial comparison: five measured repetitions after a warm-up. Compression
times include complete application lifetimes; ratios include archive overhead.
Numeric levels are not equivalent between different codecs. These results
describe the original application, not the retained optimized implementation.

| Tool | Silesia compress s / ratio | 6 GiB compress s / ratio |
|---|---:|---:|
| GPU Zstd, original | 0.478 / 2.870x | 8.027 / 1.656x |
| CPU zstd -T0 -1 | 0.222 / 2.891x | 2.797 / 1.673x |
| CPU zstd -T0 -3 | 0.251 / 3.201x | 2.728 / 1.809x |
| CPU zstd -T0 -6 | 0.421 / 3.455x | 3.095 / 1.901x |
| 7-Zip LZMA2 level 1 | 0.837 / 3.584x | 36.391 / 1.714x |
| 7-Zip LZMA2 level 5 | 23.013 / 4.273x | 161.969 / 1.993x |

CPU zstd won the roughly matched-ratio speed comparison. 7-Zip traded time
for smaller output. GPU Zstd used fewer CPU-seconds, which is useful offloading
but does not prove an elapsed-time win. The earlier stock-zip comparison used
single-threaded Deflate and did not establish superiority over these tools.

## GDeflate memory and DirectStorage

The production Manager path is unchanged and still buffers whole volumes.
With nvCOMP 5.3, a 2.5 GiB volume requests approximately 2.5 GiB input, 5.01 GiB
output capacity and 18.76 GiB workspace: **26.3 GiB**, about 10.5 times the
volume size. That is allocation demand, not a measured resident peak: the
workspace request failed on the 24 GiB GPU. A 512 MiB volume requests about
5.25 GiB across those buffers; `--volume-size 512MB` completed and verified the
6 GiB job. Whole-archive host buffering remains a separate limitation.

Successful repeated-operation probes did not show an accumulating scratch leak.
Recovery from a Manager allocation failure in a long-lived application was not
established; exiting a failed CLI process does not prove safe in-process recovery.
The old generic 2.1x memory estimate is insufficient for GDeflate compression.

DirectStorage rejected nvCOMP Manager native/RAW containers and bare low-level
tiles. Unchanged low-level GDeflate tile payloads repackaged with Microsoft's
TileStream framing decoded correctly: **256 GPU requests** across driver and
shader backends passed content checks. This demonstrates payload compatibility,
not direct readability of existing `.gdeflate` archives. No production exporter
or Manager memory redesign was added. The prototype source was removed during
cleanup; a future exporter still needs asset metadata and compatible framing.

Reference formats: [Microsoft TileStream](https://github.com/microsoft/DirectStorage/blob/main/GDeflate/GDeflate/TileStream.h)
and [reference compressor](https://github.com/microsoft/DirectStorage/blob/main/GDeflate/GDeflate/GDeflateCompress.cpp).

## Validation and limits

Completed campaigns verified 672 initial CPU/GPU comparison round trips, 88
bulk-reader investigation round trips, 32 worker-pipeline comparisons, 56
buffer-reuse/writer comparisons and 16 slot-sweep trials. Warm-ups are included
in those verification counts, but excluded from headline medians. Diagnostic
and resident-process timings are kept separate from fresh CLI comparisons.

The retained worker regression suite covers 17 successful round trips, three
batched codecs, partial chunks, single/multiple volumes, simultaneous jobs,
caller-thread callbacks, four failure/recovery scenarios, manifest sizes and
cache release. The standard C API suite covers 20 checks. Larger-slot tests were
removed with the unsupported tuning override. Cleanup validation is recorded in
the changelog.

Linux, multiple GPUs, low-VRAM hardware and the unavailable Isaac Sim workload
were not tested in this session. A pre-existing non-ASCII extraction-path issue
was reproduced with the unchanged baseline and left outside this optimization.
These measurements do not establish an intrinsic GPU codec limit or prove that
the GPU cannot outperform CPU zstd under different workloads or I/O arrangements.
