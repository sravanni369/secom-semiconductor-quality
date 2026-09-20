"""Cooperative frozen-checker guard. No git changes and no security-sandbox claim."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(root, name):
    p = (root / name).resolve()
    if not p.is_relative_to(root) or p == root:
        raise ValueError('Contract path escapes experiment root: ' + name)
    return p


def run(contract_path, hypothesis, lesson, baseline=False):
    contract_path = contract_path.resolve()
    root = contract_path.parent
    c = json.loads(contract_path.read_text(encoding='utf-8'))
    if c['mode'] not in ('verify', 'optimize') or c['direction'] not in ('min', 'max'):
        raise ValueError('Invalid mode or metric direction')
    if not c['protected'] or not c['editable'] or not c['baselines']:
        raise ValueError('Protected files, editable files and baselines are required')
    if {local(root, x) for x in c['protected']} & {local(root, x) for x in c['editable']}:
        raise ValueError('Protected and editable files overlap')
    if not isinstance(c['command'], list) or not all(isinstance(x, str) for x in c['command']):
        raise ValueError('Command must be an argument list')
    if '{result}' not in c['command']:
        raise ValueError('Evaluator command needs a separate {result} argument')
    timeout = float(c['timeout_seconds'])
    margin = float(c['min_improvement'])
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(margin) or margin < 0:
        raise ValueError('Invalid time budget or improvement margin')
    if isinstance(c['max_runs'], bool) or not isinstance(c['max_runs'], int) or c['max_runs'] < 1:
        raise ValueError('max_runs must be a positive integer')
    expected = {local(root, k): v for k, v in c['protected'].items()}
    candidate = {k: digest(local(root, k)) for k in c['editable']}
    state = local(root, '.kloop')
    state.mkdir(exist_ok=True)
    lock = state / 'running.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write('Do not remove while an evaluator is running.')
    try:
        history = state / 'results.tsv'
        if history.exists():
            with history.open(encoding='utf-8', newline='') as f:
                previous = list(csv.DictReader(f, delimiter='\t'))
        else:
            previous = []
        fingerprint = digest(contract_path)
        if previous and any(r['contract_sha256'] != fingerprint for r in previous):
            raise ValueError('Contract changed; start a new experiment directory')
        if len(previous) >= c['max_runs']:
            raise ValueError('Run budget exhausted')
        if previous and previous[-1]['verdict'] == 'ERROR':
            raise ValueError('Previous error requires a reviewed new experiment')
        if c['mode'] == 'optimize':
            for i, record in enumerate(previous):
                if record['verdict'] not in ('BASELINE', 'KEEP'):
                    continue
                old = local(root, f'.kloop/run-{i:03}')
                review = json.loads((old / 'review.json').read_text(encoding='utf-8'))
                if not isinstance(review, dict) or review.get('approved') is not True or review.get('record_sha256') != digest(old / 'record.json'):
                    raise ValueError('Independent approval missing or rejected; do not continue')
        kept = [float(r['score']) for r in previous if r['verdict'] in ('BASELINE', 'KEEP')]
        if c['mode'] == 'optimize' and ((not kept and not baseline) or (kept and baseline)):
            raise ValueError('Explicit baseline required exactly once before optimization')
        run_dir = local(root, f'.kloop/run-{len(previous):03}')
        run_dir.mkdir()
        result = run_dir / 'result.json'
        verdict, score, error = 'ERROR', None, ''
        started = time.monotonic()
        try:
            if any(not p.is_file() or digest(p) != sha for p, sha in expected.items()):
                raise ValueError('Protected file hash mismatch before evaluation')
            command = [str(result) if x == '{result}' else x for x in c['command']]
            with (run_dir / 'stdout.txt').open('w', encoding='utf-8') as stdout, (run_dir / 'stderr.txt').open('w', encoding='utf-8') as stderr:
                proc = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            if any(not p.is_file() or digest(p) != sha for p, sha in expected.items()):
                raise ValueError('Protected file hash mismatch after evaluation')
            if digest(contract_path) != fingerprint:
                raise ValueError('Contract changed during evaluation')
            if any(digest(local(root, k)) != sha for k, sha in candidate.items()):
                raise ValueError('Candidate changed during evaluation')
            if proc.returncode != 0:
                raise ValueError(f'Evaluator exited {proc.returncode}')
            measured = json.loads(result.read_text(encoding='utf-8'))
            if not isinstance(measured, dict):
                raise ValueError('Evaluator result must be a JSON object')
            if measured.get('checks_passed') is not True or isinstance(measured.get('score'), bool):
                raise ValueError('Failed gates or invalid score')
            score = float(measured['score'])
            if not math.isfinite(score):
                raise ValueError('Nonfinite score')
            if c['mode'] == 'verify':
                verdict = 'VERIFIED'
            elif not kept:
                verdict = 'BASELINE'
            else:
                best = min(kept) if c['direction'] == 'min' else max(kept)
                gain = best - score if c['direction'] == 'min' else score - best
                verdict = 'KEEP' if gain > margin else 'DISCARD'
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            error = f'{type(exc).__name__}: {exc}'
        row = dict(utc=datetime.now(timezone.utc).isoformat(), cycle=len(previous),
                   contract_sha256=fingerprint, candidate_sha256=json.dumps(candidate, sort_keys=True),
                   score=score, seconds=round(time.monotonic()-started, 3), verdict=verdict,
                   hypothesis=hypothesis, lesson=lesson, error=error)
        with history.open('a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row), delimiter='\t')
            if not previous:
                w.writeheader()
            w.writerow(row)
        (run_dir/'record.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        print(json.dumps(row))
        return 2 if verdict == 'ERROR' else 1 if verdict == 'DISCARD' else 0
    finally:
        lock.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('contract', type=Path)
    p.add_argument('--hypothesis', required=True)
    p.add_argument('--lesson', required=True, help='Initial lesson/context; follow-up reviews are separate records')
    p.add_argument('--baseline', action='store_true')
    args = p.parse_args()
    try:
        raise SystemExit(run(args.contract, args.hypothesis, args.lesson, args.baseline))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'STOP: {exc}')
        raise SystemExit(2)
