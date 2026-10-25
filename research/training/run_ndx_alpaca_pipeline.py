"""Finish reviewed feature construction, augmented fits, evaluation and audit.

The separately running resumable downloader and daily-control trainer are allowed
to finish first. This is a finite local pipeline, not a scheduled automation.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs/nasdaq100_alpaca_20261019'
RAW = ROOT/'data/nasdaq_monthly_pit/alpaca_15min/sip_raw_20180101_20260101'
FEATURES = ROOT/'data/nasdaq_monthly_pit/alpaca_features_20261019'


def state(stage, **details):
    value = dict(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **details)
    temporary = RUN/'PIPELINE_PROGRESS.json.tmp'
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(RUN/'PIPELINE_PROGRESS.json')
    print(json.dumps(value), flush=True)


def run(script, *arguments):
    state(script)
    subprocess.run([sys.executable, str(ROOT/script), *arguments], cwd=ROOT, check=True)


def main():
    deadline = time.monotonic()+6*3600
    while True:
        manifest = json.loads((RAW/'manifest.json').read_text())
        status = manifest.get('download_status')
        if status=='completed_with_review_required':
            break
        if status=='interrupted_or_failed':
            raise RuntimeError('Download stopped; completed cache preserved. Review acquisition before continuing.')
        progress = json.loads((RAW/'PROGRESS.json').read_text())
        state('waiting_for_download', **progress)
        if time.monotonic()>deadline:
            raise RuntimeError('Download wait exceeded six hours; cache is resumable.')
        time.sleep(30)
    if not (FEATURES/'REVIEW.json').exists():
        run('prepare_ndx_alpaca.py')
    review = json.loads((FEATURES/'REVIEW.json').read_text())
    if review['status']!='passed_research_input_gate':
        raise RuntimeError('Data gate failed; do not fit augmented models.')
    while True:
        complete = len(list((RUN/'daily_control').glob('*/*_record.json')))
        if complete==140:
            break
        state('waiting_for_daily_controls', complete=complete, planned=140)
        if time.monotonic()>deadline:
            raise RuntimeError('Daily controls are incomplete; resume trainer before continuing.')
        time.sleep(30)
    if not (RUN/'FITS_COMPLETE_LOCK.json').exists():
        run('run_ndx_alpaca_training.py', '--view', 'intraday_augmented', '--workers', '2')
    if not (RUN/'completion.json').exists():
        run('report_ndx_alpaca.py')
    if not (RUN/'AUDIT.json').exists():
        run('audit_ndx_alpaca.py')
    run('publish_ndx_alpaca.py')
    state('completed_audited_publication', report=str(ROOT/'docs/nasdaq_alpaca_20261019/REPORT.md'))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        # No credentials are read by this orchestrator; retain a short stage failure.
        state('stopped_requires_investigation', error_type=type(error).__name__)
        raise
