"""Download/verify Silesia and compare two nvCOMP CLI builds (Python 3.9+).

Uses the standard library. Each executable must retain its matching runtime DLLs.
Raw samples, logs, corpus hashes, build hashes and summaries are saved per run.
"""

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia'
URL = 'https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip'
# ZIP SHA-256 recorded from the author's download; individual MD5s and sizes
# below are published on the corpus page and checked independently.
ZIP_SHA256 = '0626e25f45c0ffb5dc801f13b7c82a3b75743ba07e3a71835a41e3d9f63c77af'
FILES = {
    'dickens': (10192446, '88334708559f6db57d79096bc0aca07e'),
    'mozilla': (51220480, 'c7789a2097f1ff944b0c737430a339b3'),
    'mr': (9970564, '38e623e3093b7bf2003ca4b1bbc19927'),
    'nci': (33553445, '31f85bc8706f3c921104e7c169e2e2e1'),
    'ooffice': (6152192, '573c4ae915e36631d8f2dcffb9b9b66d'),
    'osdb': (10085684, 'e734b0c48e6a982adfb5802da3032ecd'),
    'reymont': (6627202, 'd8f54d78105079775f32d76dc55fc671'),
    'samba': (21606400, '154eaea7ea70e89f6339ff0abf4112ca'),
    'sao': (7251944, '79e95a22e18cd82b7e42bf91b380d30b'),
    'webster': (41458703, '474931ad907ac27bf962c75ded46c069'),
    'xml': (5345280, '9b09c0c80104adb8aae910b7d7db003e'),
    'x-ray': (8474240, '9baec32ad14ec3eff487d254382cb91c'),
}
ALGORITHMS = ('lz4', 'snappy', 'zstd', 'gdeflate', 'ans', 'bitcomp')


def digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare(data, download):
    corpus = data / 'silesia'
    if not corpus.exists():
        archive = data / 'silesia.zip'
        if not archive.exists():
            if not download:
                raise RuntimeError('Corpus missing; run with --download or provide --data-dir')
            data.mkdir(parents=True, exist_ok=True)
            print(f'Downloading {URL}', flush=True)
            partial = archive.with_suffix('.zip.part')
            with urllib.request.urlopen(URL, timeout=120) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out)
            if digest(partial) != ZIP_SHA256:
                raise RuntimeError('Downloaded ZIP checksum mismatch')
            partial.replace(archive)
        if digest(archive) != ZIP_SHA256:
            raise RuntimeError('ZIP checksum mismatch')
        with zipfile.ZipFile(archive) as bundle:
            if sorted(bundle.namelist()) != sorted(FILES):
                raise RuntimeError('Unexpected ZIP members')
            corpus.mkdir()
            # Write only explicitly named files; never follow archive paths.
            for name, (size, _) in FILES.items():
                if bundle.getinfo(name).file_size != size:
                    raise RuntimeError(f'Unexpected ZIP member size: {name}')
                with bundle.open(name) as src, (corpus / name).open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    if sorted(p.name for p in corpus.iterdir()) != sorted(FILES):
        raise RuntimeError('Corpus directory must contain exactly the 12 Silesia files')
    manifest = {}
    for name, (size, md5) in FILES.items():
        path = corpus / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != size or digest(path, 'md5') != md5:
            raise RuntimeError(f'Corpus verification failed: {name}')
        manifest[name] = {'bytes': size, 'md5': md5, 'sha256': digest(path)}
    return corpus, manifest


def build_identity(exe):
    files = [exe]
    files += [p for name in ('nvcomp_core.dll', 'nvcomp64_5.dll', 'nvcomp_cpu64_5.dll')
              if (p := exe.parent / name).exists()]
    return {'executable': str(exe), 'sha256': {p.name: digest(p) for p in files}}


def summarize(samples, repeats):
    summaries = []
    keys = sorted({(s['algorithm'], s['asset'], s['build']) for s in samples})
    for algo, asset, build in keys:
        rows = [s for s in samples if (s['algorithm'], s['asset'], s['build']) == (algo, asset, build)
                and not s['warmup']]
        compressed = [s for s in rows if s['compress_exit'] == 0 and s['verified']]
        decompressed = [s for s in compressed if s['decompress_exit'] == 0]
        summary = dict(algorithm=algo, asset=asset, build=build, successful_runs=len(decompressed),
                       expected_runs=repeats, failed_runs=len(rows) - len(decompressed),
                       compress_successful_runs=len(compressed), decompress_successful_runs=len(decompressed))
        for metric, good in (('compress_s', compressed), ('archive_bytes', compressed),
                             ('decompress_s', decompressed)):
            if good:
                values = [s[metric] for s in good]
                summary[metric] = statistics.median(values)
                summary[metric + '_min'] = min(values)
                summary[metric + '_max'] = max(values)
        if compressed:
            summary['input_bytes'] = compressed[0]['input_bytes']
            summary['ratio'] = summary['input_bytes'] / summary['archive_bytes']
            for operation in ('compress', 'decompress'):
                if operation + '_s' in summary:
                    summary[operation + '_mib_s'] = summary['input_bytes'] / 1048576 / summary[operation + '_s']
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'bench/data')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'bench/results')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument('--scope', choices=('files', 'folder', 'both'), default='both')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 1 or args.timeout <= 0:
        parser.error('repeats, warmups and timeout must be positive')
    corpus, manifest = prepare(args.data_dir.resolve(), args.download)
    print(f'Verified {len(manifest)} corpus files, {sum(v["bytes"] for v in manifest.values()):,} bytes', flush=True)
    if args.prepare_only:
        return
    if not args.baseline or not args.candidate:
        parser.error('--baseline and --candidate are required for comparison')
    executables = {'baseline': args.baseline.resolve(strict=True), 'candidate': args.candidate.resolve(strict=True)}
    if os.name == 'nt':
        # Suppress crash dialogs so failed legacy processes cannot block the run.
        ctypes.windll.kernel32.SetErrorMode(3)
    env = {key.upper(): value for key, value in os.environ.items()} if os.name == 'nt' else os.environ.copy()
    run_dir = args.results_dir.resolve() / datetime.now(timezone.utc).strftime('silesia_%Y%m%dT%H%M%S_%fZ')
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'logs').mkdir()
    scratch = run_dir / 'scratch'
    scratch.mkdir()
    metadata = {'source': SOURCE, 'download_url': URL, 'zip_sha256': ZIP_SHA256,
                'corpus': manifest, 'platform': platform.platform(), 'python': sys.version,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'builds': {name: build_identity(exe) for name, exe in executables.items()},
                'environment': {k: v for k, v in env.items() if k.startswith('NVCOMP_') or k == 'CUDA_VISIBLE_DEVICES'}}
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,utilization.gpu',
                              '--format=csv'], capture_output=True, text=True, timeout=10)
        metadata['gpu'] = gpu.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        metadata['gpu'] = 'unavailable'
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Results: {run_dir}', flush=True)

    def invoke(exe, label, arguments):
        with (run_dir / 'logs' / f'{label}.log').open('wb') as log:
            start = time.perf_counter()
            try:
                result = subprocess.run([str(exe), *map(str, arguments)], env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
                status = result.returncode
            except subprocess.TimeoutExpired:
                status = 'timeout'
            elapsed = time.perf_counter() - start
        text = (run_dir / 'logs' / f'{label}.log').read_text(errors='replace')
        phases = {name.lower(): float(value) for name, value in
                  re.findall(r'^\s*(Read|Prepare|Compute|Write|Total)\s*:\s*([\d.]+) s', text, re.M)}
        return elapsed, status, phases

    assets = list(FILES) if args.scope != 'folder' else []
    if args.scope != 'files':
        assets.append('corpus-folder')
    cases = [(algo, asset) for algo in args.algorithms for asset in assets]
    random.Random(530).shuffle(cases)
    samples = []
    try:
        with (run_dir / 'samples.jsonl').open('w', encoding='utf-8') as raw:
            for index, (algo, asset) in enumerate(cases):
                source = corpus if asset == 'corpus-folder' else corpus / asset
                expected = manifest if asset == 'corpus-folder' else {asset: manifest[asset]}
                for iteration in range(args.warmups + args.repeats):
                    order = list(executables)
                    if (iteration + index) % 2:
                        order.reverse()
                    for build in order:
                        exe = executables[build]
                        slot = scratch / build
                        slot.mkdir(exist_ok=True)
                        archive = slot / f'archive.{algo}'
                        restored = slot / 'restored'
                        # Both paths are owned scratch outputs, never corpus/user inputs.
                        if restored.exists():
                            if not restored.resolve().is_relative_to(scratch.resolve()):
                                raise RuntimeError('Unsafe scratch cleanup path')
                            shutil.rmtree(restored)
                        archive.unlink(missing_ok=True)
                        label = f'{algo}-{asset}-{iteration}-{build}'
                        cs, ce, cp = invoke(exe, label + '-compress', ['-c', source, archive, algo, '--no-volumes'])
                        ds, de, dp, verified = None, 'not-run', {}, False
                        compressed_size = archive.stat().st_size if archive.exists() else None
                        if ce == 0 and compressed_size:
                            ds, de, dp = invoke(exe, label + '-decompress', ['-d', archive, restored, algo])
                            actual = {p.relative_to(restored).as_posix(): digest(p)
                                      for p in restored.rglob('*') if p.is_file()}
                            verified = actual == {name: info['sha256'] for name, info in expected.items()}
                        sample = dict(algorithm=algo, asset=asset, build=build, iteration=iteration,
                                      warmup=iteration < args.warmups, first_in_pair=build == order[0],
                                      input_bytes=sum(v['bytes'] for v in expected.values()), archive_bytes=compressed_size,
                                      compress_s=cs, decompress_s=ds, compress_exit=ce, decompress_exit=de,
                                      compress_phases=cp, decompress_phases=dp, verified=verified)
                        samples.append(sample)
                        raw.write(json.dumps(sample) + '\n')
                        raw.flush()
                        if ce != 0 or de != 0 or not verified:
                            print(f'FAIL {label}: compress={ce}, decompress={de}, content={verified}', flush=True)
                print(f'[{index + 1}/{len(cases)}] {algo} / {asset}', flush=True)
    finally:
        summaries = summarize(samples, args.repeats)
        (run_dir / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
        if summaries:
            columns = sorted({k for row in summaries for k in row})
            with (run_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(summaries)
        # Keep logs and measurements; remove only this run's verified scratch subtree.
        if scratch.resolve().parent != run_dir.resolve():
            raise RuntimeError('Unsafe scratch cleanup path')
        shutil.rmtree(scratch)
    failures = sum(s['compress_exit'] != 0 or s['decompress_exit'] != 0 or not s['verified'] for s in samples)
    print(f'{len(samples)} round trips; {failures} failed. Results: {run_dir}', flush=True)
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
