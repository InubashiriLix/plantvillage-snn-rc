#!/usr/bin/env python3
"""Audit tracked paths, sizes, text secrets/personal paths and Markdown links."""
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    problems = []
    total, largest = 0, (0, '')
    forbidden = {'.pt', '.pth', '.ckpt', '.npy', '.npz', '.joblib', '.pkl', '.pickle', '.zip', '.gz', '.tgz', '.tar', '.xz', '.zst', '.bz2', '.7z', '.pyc', '.safetensors', '.memmap'}
    sensitive = re.compile(rb'(?:/(?:home|Users)/[^/\s]+/|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for name in filter(None, paths):
        p = ROOT / name
        if not p.is_file():
            problems.append(f'missing tracked file: {name}')
            continue
        size = p.stat().st_size
        total += size
        largest = max(largest, (size, name))
        if size > 50 * 1024**2:
            problems.append(f'file exceeds 50 MiB: {name}')
        if p.suffix in forbidden or any(part in {'data', '__pycache__', '.pytest_cache', '.venv', 'tmp', 'output', '.agents', '.codex'} or part.startswith('artifacts') for part in Path(name).parts[:-1]):
            problems.append(f'generated or private file tracked: {name}')
        if re.search(r'_inputs.*\.csv$', name) or p.name == 'samples.csv':
            problems.append(f'complete hardware CSV tracked: {name}')
        blob = p.read_bytes()
        if sensitive.search(blob):
            problems.append(f'secret or personal absolute path: {name}')
        if p.suffix == '.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)', blob.decode()):
                if target.startswith(('http:', 'https:', 'mailto:', '#')):
                    continue
                target = unquote(target.split('#')[0])
                if not (p.parent / target).exists():
                    problems.append(f'broken Markdown link: {name}: {target}')
    if problems:
        print('\n'.join(problems), file=sys.stderr)
        return 1
    print(f'Audit passed: {len(list(filter(None, paths)))} tracked files; {total / 1024**2:.2f} MiB; largest {largest[1]} ({largest[0] / 1024**2:.2f} MiB).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
