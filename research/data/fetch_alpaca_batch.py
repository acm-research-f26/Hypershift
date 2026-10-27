"""Resume the historical 15-minute snapshot with a shared request budget.

Uses the exact request/chunk contract of fetch_alpaca_15min.py. Network I/O only
is parallel; every HTTP attempt passes one shared rate limiter.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import io
import json
from pathlib import Path
import threading
import time

import pandas as pd
import requests

import fetch_alpaca_15min as acquisition


class Pacer:
    def __init__(self, rpm=190):
        self.interval = 60 / rpm
        self.next_start = 0.
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            scheduled = max(now, self.next_start)
            self.next_start = scheduled + self.interval
        time.sleep(max(0., scheduled - time.monotonic()))


class PacedSession:
    def __init__(self, pacer):
        self.pacer, self.session = pacer, requests.Session()

    def get(self, *args, **kwargs):
        self.pacer.acquire()
        return self.session.get(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--requests-per-minute', type=int, default=190)
    parser.add_argument('--start', default='2018-01-01')
    parser.add_argument('--end', default='2026-01-01')
    parser.add_argument('--feed', choices=['sip', 'iex'], default='sip')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 2 <= args.requests_per_minute <= 190:
        parser.error('Use 1..16 workers and 2..190 requests/minute.')
    command = ['--dry-run', '--start', args.start, '--end', args.end, '--feed', args.feed]
    if args.out:
        command += ['--out', str(args.out)]
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        acquisition.main(command)
    planned = json.loads(captured.getvalue())
    plan, out = planned['plan'], Path(planned['output_directory'])
    headers = acquisition.auth_headers(acquisition.read_env(acquisition.ROOT / '.env'))
    lock = out / 'REQUEST.json'
    if lock.exists():
        if json.loads(lock.read_text()) != plan:
            raise acquisition.AlpacaError('Existing snapshot request changed; refusing to mix inputs.')
    elif out.exists() and any(out.iterdir()):
        raise acquisition.AlpacaError('Nonempty output without matching request lock.')
    out.mkdir(parents=True, exist_ok=True)
    acquisition.write_json(lock, plan)
    acquisition.write_json(out / 'BATCH_IMPLEMENTATION.json',
        dict(source=str(Path(__file__).name), sha256=acquisition.digest(Path(__file__)),
             workers=args.workers, requests_per_minute=args.requests_per_minute))
    acquisition.write_json(out / 'manifest.json', dict(**plan, download_status='in_progress'))
    pacer, local = Pacer(args.requests_per_minute), threading.local()
    jobs = [(s, a, b) for s in plan['symbols'] for a, b in acquisition.chunks(
        acquisition.date_bound(args.start), acquisition.date_bound(args.end))]
    def work(job):
        if not hasattr(local, 'client'):
            local.client = acquisition.AlpacaClient(headers, rpm=1000, session=PacedSession(pacer))
        symbol, start, end = job
        return acquisition.save_chunk(local.client, out, symbol, start, end, args.feed, 'raw')
    records, failed = [], None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception:
                failed = 'A download chunk failed; incomplete snapshot requires investigation. Details suppressed.'
                for pending in futures:
                    pending.cancel()
                break
            records.append(record)
            acquisition.write_json(out / 'PROGRESS.json',
                dict(completed_chunks=len(records), planned_chunks=len(jobs),
                     rows=sum(r['rows'] for r in records),
                     last_symbol=record['query']['symbol'], last_chunk=record['query']['start']))
            if len(records) % 8 == 0:
                print(f'{len(records)}/{len(jobs)} chunks; {sum(r["rows"] for r in records):,} bars', flush=True)
    if failed:
        acquisition.write_json(out / 'manifest.json',
            dict(**plan, download_status='interrupted_or_failed', completed_chunks=len(records), message=failed))
        raise acquisition.AlpacaError(failed)
    records.sort(key=lambda r: (r['query']['symbol'], r['query']['start']))
    rows = [dict(symbol=r['query']['symbol'], start=r['query']['start'],
                 end_exclusive=r['query']['end_exclusive'], rows=r['rows'], status=r['status'],
                 first_bar=r['first_bar'], last_bar=r['last_bar'], file=r['file'], sha256=r['sha256'])
            for r in records]
    coverage = pd.DataFrame(rows)
    coverage.to_csv(out / 'coverage.csv', index=False)
    acquisition.write_json(out / 'manifest.json', dict(**plan,
        download_status='completed_with_review_required', completed_chunks=len(records),
        rows=int(coverage.rows.sum()), empty_chunks=int((coverage.rows == 0).sum()), chunks=records,
        limitations=['Unreviewed price-only snapshot; corporate actions, identities and sessions require review.',
                     'All sessions retained. Missing bars remain absent. Raw bars are not total returns.']))
    print(f'Completed {len(records)} chunks; {coverage.rows.sum():,} bars: {out}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (acquisition.AlpacaError, OSError, ValueError):
        raise SystemExit('Download stopped; completed chunks remain resumable. Review manifest and progress.') from None
