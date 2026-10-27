"""Strict monthly company ranking from independently reviewed historical inputs.

This module validates and ranks supplied snapshots. It does not certify a
vendor's historical coverage or turn current ticker lists into historical data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

DATASET = 'NASDAQ_monthly_top100'
SNAPSHOT_COLUMNS = ['formation_date', 'company_id', 'security_id', 'ticker',
                    'exchange', 'security_type', 'company_market_cap_usd',
                    'adv20_usd', 'cap_available_date', 'listing_available_date',
                    'identity_available_date', 'source']
COVERAGE_COLUMNS = ['formation_date', 'expected_security_count',
                    'historical_listing_complete', 'missing_company_caps',
                    'publication_timing_reviewed', 'issuer_identity_reviewed']


def required_columns(frame, names, label):
    missing = sorted(set(names)-set(frame.columns))
    if missing:
        raise ValueError(f'{label}: missing columns {missing}')


def rank_months(snapshots, coverage, calendar, months, count=100):
    """Use the last session BEFORE a month, including financials and common ADRs.

    Snapshots cover every NASDAQ common security eligible on the formation date.
    Company caps repeat across classes and are never summed twice. One class is
    chosen using already-observed ADV, with stable ID tie breaks. Missing cap or
    identity anywhere in the candidate snapshot blocks ranking rather than
    silently dropping a potentially large issuer.
    """
    required_columns(snapshots, SNAPSHOT_COLUMNS, 'snapshots')
    required_columns(coverage, COVERAGE_COLUMNS, 'coverage')
    calendar = pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError('Calendar must be unique and increasing')
    frame, inventory = snapshots.copy(), coverage.copy()
    date_columns = ['formation_date', 'cap_available_date',
                    'listing_available_date', 'identity_available_date']
    for col in date_columns:
        frame[col] = pd.to_datetime(frame[col], errors='raise')
        if frame[col].isna().any():
            raise ValueError(f'Missing {col}')
    inventory['formation_date'] = pd.to_datetime(inventory.formation_date)
    if inventory.formation_date.duplicated().any():
        raise ValueError('Duplicate coverage attestations')
    result=[]
    for month in pd.PeriodIndex(months, freq='M'):
        before = calendar[calendar < month.start_time]
        if not len(before):
            raise ValueError(f'No prior session for {month}')
        formation = before[-1]
        rows = frame[frame.formation_date == formation].copy()
        audit = inventory[inventory.formation_date == formation]
        if len(audit) != 1 or rows.empty:
            raise ValueError(f'Missing complete formation snapshot: {formation.date()}')
        audit = audit.iloc[0]
        for flag in ['historical_listing_complete', 'publication_timing_reviewed', 'issuer_identity_reviewed']:
            if not (isinstance(audit[flag], (bool,np.bool_)) and bool(audit[flag])):
                raise ValueError(f'{formation.date()}: {flag} not verified')
        if int(audit.missing_company_caps) != 0 or len(rows) != int(audit.expected_security_count):
            raise ValueError(f'{formation.date()}: incomplete candidate/cap coverage')
        if rows.security_id.duplicated().any():
            raise ValueError('Duplicate security identity in a snapshot')
        for col in ['company_id','security_id','ticker','source']:
            if rows[col].isna().any() or rows[col].astype(str).str.strip().eq('').any():
                raise ValueError(f'Missing {col}')
        if not rows.exchange.eq('XNAS').all() or not rows.security_type.isin(['CS','ADRC']).all():
            raise ValueError('Snapshot must contain NASDAQ common stock/common ADRs only')
        for col in date_columns[1:]:
            if (rows[col] > formation).any():
                raise ValueError(f'{formation.date()}: future {col}')
        for col in ['company_market_cap_usd','adv20_usd']:
            rows[col] = pd.to_numeric(rows[col],errors='raise')
            if not np.isfinite(rows[col]).all() or (rows[col] < 0).any():
                raise ValueError(f'Invalid {col}')
        if (rows.company_market_cap_usd <= 0).any():
            raise ValueError('Company cap must be positive')
        if (rows.groupby('company_id').company_market_cap_usd.nunique() != 1).any():
            raise ValueError('Multiple classes must share a reconciled COMPANY cap')
        # Security representation never depends on later prices or availability.
        companies = (rows.sort_values(['adv20_usd','security_id'],ascending=[False,True],kind='stable')
                     .drop_duplicates('company_id'))
        companies = companies.sort_values(['company_market_cap_usd','company_id'],
                                          ascending=[False,True],kind='stable')
        if len(companies) < count:
            raise ValueError(f'Fewer than {count} companies on {formation.date()}')
        chosen = companies.head(count).copy()
        chosen.insert(0,'month',str(month))
        chosen.insert(1,'effective_from',month.start_time)
        chosen.insert(2,'effective_to',month.end_time.normalize())
        chosen.insert(3,'rank',np.arange(1,count+1))
        result.append(chosen)
    return pd.concat(result,ignore_index=True)


def membership_on_signals(members, security_ids, signal_dates, execution_dates):
    """Known monthly membership for next-close orders, keyed by stable security IDs.

    Orders crossing month-end may use the new month's list only when its
    formation/publication dates are no later than the signal. No membership is
    backfilled into an earlier training or validation date.
    """
    required_columns(members, ['effective_from','effective_to','formation_date',
                              'security_id','cap_available_date','listing_available_date',
                              'identity_available_date'], 'members')
    frame=members.copy()
    for col in ['effective_from','effective_to','formation_date','cap_available_date',
                'listing_available_date','identity_available_date']:
        frame[col]=pd.to_datetime(frame[col])
    signals, executions = pd.DatetimeIndex(signal_dates), pd.DatetimeIndex(execution_dates)
    if len(signals) != len(executions) or (executions <= signals).any():
        raise ValueError('Each execution must follow its signal')
    ids={security:i for i,security in enumerate(security_ids)}
    if len(ids) != len(security_ids):
        raise ValueError('Duplicate panel security IDs')
    mask=np.zeros((len(signals),len(ids)),dtype=bool)
    for i,(signal,execution) in enumerate(zip(signals,executions)):
        rows=frame[(frame.effective_from <= execution)&(frame.effective_to >= execution)]
        if rows.empty or rows.security_id.duplicated().any():
            raise ValueError(f'Missing/duplicate monthly members for {execution.date()}')
        for col in ['formation_date','cap_available_date','listing_available_date','identity_available_date']:
            if rows[col].isna().any() or (rows[col] > signal).any():
                raise ValueError(f'Unknown-at-signal {col}')
        for security in rows.security_id:
            if security not in ids:
                raise ValueError(f'Selected security missing from panel: {security}')
            mask[i,ids[security]]=True
    return mask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('data/nasdaq_monthly_pit/input'))
    parser.add_argument('--out',type=Path,default=Path('data/nasdaq_monthly_pit/membership'))
    parser.add_argument('--start',default='2017-01')
    parser.add_argument('--end',default='2025-12')
    args=parser.parse_args()
    files=['candidate_snapshots.csv','coverage.csv','calendar.csv']
    missing=[name for name in files if not (args.input/name).is_file()]
    if missing:
        parser.error('Historical data gate: missing '+', '.join(missing)+
                     '. See docs/nasdaq_monthly_pit/DATA_REQUIREMENTS.md. No training performed.')
    rows=rank_months(pd.read_csv(args.input/files[0],dtype={'company_id':str,'security_id':str}),
                    pd.read_csv(args.input/files[1]),
                    pd.to_datetime(pd.read_csv(args.input/files[2])['date']),
                    pd.period_range(args.start,args.end,freq='M'))
    if args.out.exists():
        parser.error('Choose a fresh immutable output directory')
    args.out.mkdir(parents=True)
    rows.to_csv(args.out/'members.csv',index=False)
    (args.out/'manifest.json').write_text(json.dumps(dict(dataset=DATASET,
        status='membership_ranked_prices_and_events_still_require_review',
        months=int(rows.month.nunique()),companies_per_month=100,
        input_hashes={name:hashlib.sha256((args.input/name).read_bytes()).hexdigest() for name in files}),indent=2),encoding='utf-8')
    print(f'Ranked {rows.month.nunique()} months; this does not authorize training without price/event validation.')


if __name__=='__main__':main()
