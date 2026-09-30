"""Compare archives across SDK builds using real CLI processes (GPU required).

Run with --phase baseline before upgrading, then --phase compare afterwards.
Keep each executable beside its own nvcomp_core and SDK runtime libraries.
Requires Python 3.9+; uses only the standard library.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


ALGORITHMS = ('lz4', 'snappy', 'zstd', 'gdeflate', 'ans', 'bitcomp')
CPU_ALGORITHMS = ALGORITHMS[:3]


def manifest(root):
    return {
        p.relative_to(root).as_posix(): {
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'mtime': int(p.stat().st_mtime),
        }
        for p in sorted(root.rglob('*')) if p.is_file()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('baseline', 'compare'), required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=30, help='Per-process timeout in seconds')
    args = parser.parse_args()
    baseline = args.baseline.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True) if args.candidate else None
    if args.phase == 'compare' and candidate is None:
        parser.error('--candidate is required for compare')
    work = args.work.resolve()
    records = []
    failures = []
    checks = 0
    if os.name == 'nt':
        # Let failed child processes terminate without a Windows crash dialog.
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    env = {key.upper(): value for key, value in os.environ.items()} if os.name == 'nt' else os.environ.copy()

    def run(exe, label, *arguments, allow_failure=False):
        start = time.perf_counter()
        try:
            result = subprocess.run([str(exe), *map(str, arguments)],
                                    capture_output=True, timeout=args.timeout, env=env)
            output = result.stdout + result.stderr
            status = result.returncode
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or b'') + (error.stderr or b'')
            status = 'timeout'
        elapsed = time.perf_counter() - start
        (work / 'logs' / f'{label}.log').write_bytes(output)
        records.append({'label': label, 'seconds': elapsed, 'exit_code': status})
        if status:
            failures.append(f'{label}: CLI exit {status}')
            if not allow_failure:
                raise RuntimeError(f'{label}: CLI exit {status}; see logs')

    def check(exe, label, archive, expected, cpu=False):
        nonlocal checks
        output = work / 'extracted' / label
        if output.exists():
            raise RuntimeError(f'Refusing to reuse extraction output: {output}')
        # Batched archives support detection; native manager archives need an algorithm.
        algo = archive.suffix[1:]
        algorithm_arg = [] if algo in CPU_ALGORITHMS else [algo]
        run(exe, label, '-d', archive, output, *algorithm_arg, *(['--cpu'] if cpu else []),
            allow_failure=True)
        actual = manifest(output)
        if actual != expected:
            failures.append(f'{label}: extracted content, paths or modification times differ')
            print(f'FAIL content/metadata: {label}', flush=True)
        else:
            checks += 1
            print(f'PASS content/metadata: {label}', flush=True)

    if args.phase == 'baseline':
        work.mkdir(parents=True, exist_ok=False)
        (work / 'logs').mkdir()
        source = work / 'source'
        (source / 'nested space').mkdir(parents=True)
        rng = random.Random(530)
        for size in (0, 1, 4, 65535, 65536, 65537, 3 * 1024 * 1024 + 7):
            # Tiny files, exact chunk boundaries and an incompressible spanning file.
            data = rng.randbytes(size)
            path = source / 'nested space' / f'input-{size}.bin'
            path.write_bytes(data)
            os.utime(path, (1710506096, 1710506096))
        path = source / 'repeated.txt'
        path.write_bytes(b'nvCOMP SDK compatibility\n' * 10000)
        os.utime(path, (1710506096, 1710506096))
        expected = manifest(source)
        (work / 'manifest.json').write_text(json.dumps(expected, indent=2), encoding='utf-8')
    else:
        source = work / 'source'
        expected = json.loads((work / 'manifest.json').read_text(encoding='utf-8'))
        if manifest(source) != expected:
            raise AssertionError('Baseline input has changed')

    writer = baseline if args.phase == 'baseline' else candidate
    version = 'baseline' if args.phase == 'baseline' else 'candidate'
    archive_dir = work / version
    archive_dir.mkdir(exist_ok=False)
    try:
        for multi in (False, True):
            for algo in ALGORITHMS:
                for cpu in (False, True) if algo in CPU_ALGORITHMS else (False,):
                    case = f'{"multi" if multi else "tree"}-{algo}-{"cpu" if cpu else "gpu"}'
                    archive = archive_dir / f'{case}.{algo}'
                    flags = ['--volume-size', '1MB'] if multi else ['--no-volumes']
                    if cpu:
                        flags.append('--cpu')
                    run(writer, f'{version}-compress-{case}', '-c', source, archive, algo, *flags)
                    if multi:
                        archive = archive_dir / f'{case}.vol001.{algo}'
                        if len(list(archive_dir.glob(f'{case}.vol*.{algo}'))) < 2:
                            raise AssertionError(f'{case}: test did not produce multiple volumes')
                    # The current listing implementation only supports batched codecs.
                    if algo in CPU_ALGORITHMS:
                        run(writer, f'{version}-list-{case}', '-l', archive)
                    check(writer, f'{version}-self-{case}', archive, expected)
                    if args.phase == 'compare':
                        old_archive = work / 'baseline' / archive.name
                        check(candidate, f'old-to-new-{case}', old_archive, expected)
                        check(baseline, f'new-to-old-{case}', archive, expected)
                    if algo in CPU_ALGORITHMS:
                        check(writer, f'{version}-cpu-read-{case}', archive, expected, cpu=True)
                        if args.phase == 'compare':
                            check(candidate, f'old-to-new-cpu-{case}', old_archive, expected, cpu=True)
                            check(baseline, f'new-to-old-cpu-{case}', archive, expected, cpu=True)
    finally:
        (work / f'{version}-timings.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        (work / f'{version}-failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    print(f'{checks} content and metadata checks passed; logs and fixtures: {work}')
    if failures:
        print('\n'.join(failures))
        sys.exit(1)


if __name__ == '__main__':
    main()
