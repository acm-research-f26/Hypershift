"""Versioned after-hours/premarket summaries; never edits the regular-session study."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pit_research_engine import load_data
from prepare_ndx_alpaca import RAW,CALENDAR,IDENTITY,calendar_frame,filter_identity,sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/nasdaq_monthly_pit/extended_hours_features_20261024'
FIELDS=['return','range','within_bar_realized_vol','close_to_vwap','log_dollar_volume',
        'last_hour_return','late_volume_share','coverage','first_trade_delay','last_trade_age']


def windows(calendar,segment):
    dates=pd.Series(calendar.index,index=calendar.index).dt.strftime('%Y-%m-%d')
    local=lambda clock:pd.to_datetime(dates+' '+clock).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    if segment=='after_hours':return calendar.close_utc,local('20:00')
    if segment=='premarket':return local('04:00'),local('09:15')
    raise ValueError('Unknown extended session')


def summarize(frame,calendar,segment):
    """Only positive-trade bars with fully elapsed intervals enter a window."""
    start,end=windows(calendar,segment)
    columns=['date','observed','latest_bar_end']+FIELDS
    if frame.empty:return pd.DataFrame(columns=columns)
    f=frame.copy();f['timestamp']=pd.to_datetime(f.timestamp_utc,utc=True,format='ISO8601')
    f['end']=pd.to_datetime(f.bar_end_utc,utc=True,format='ISO8601')
    f['date']=f.timestamp.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    f=f[f.date.isin(calendar.index)]
    f['start']=f.date.map(start);f['cutoff']=f.date.map(end)
    on_grid=(f.timestamp.dt.minute%15==0)&(f.timestamp.dt.second==0)&((f.end-f.timestamp)==pd.Timedelta(minutes=15))
    active=f[(f.timestamp>=f.start)&(f.end<=f.cutoff)&on_grid&(f.volume>0)&(f.trade_count>0)].copy()
    if active.timestamp.duplicated().any():raise ValueError('Duplicate active extended-hours bars')
    if active.empty:return pd.DataFrame(columns=columns)
    active=active.sort_values('timestamp')
    active['squared_return']=np.log(active.close/active.open)**2
    active['weighted_price']=active.vwap*active.volume
    active['dollar_notional']=active.close*active.volume
    group=active.groupby('date',sort=True)
    d=group.agg(observed=('timestamp','size'),first_start=('timestamp','first'),last_end=('end','last'),
        first_open=('open','first'),last_close=('close','last'),maximum=('high','max'),minimum=('low','min'),
        volume=('volume','sum'),squared_return=('squared_return','sum'),weighted_price=('weighted_price','sum'),
        dollar_notional=('dollar_notional','sum'))
    duration=(end-start).dt.total_seconds();d['coverage']=d.observed/(duration.reindex(d.index)/900)
    d['first_trade_delay']=(d.first_start-start.reindex(d.index)).dt.total_seconds()/duration.reindex(d.index)
    d['last_trade_age']=(end.reindex(d.index)-d.last_end).dt.total_seconds()/duration.reindex(d.index)
    late=active[active.timestamp>=active.cutoff-pd.Timedelta(hours=1)].groupby('date')
    last_open=late.open.first();last_volume=late.volume.sum()
    if segment=='after_hours':
        closing=f[(f.end==f.date.map(calendar.close_utc))&(f.timestamp>=f.date.map(calendar.open_utc))&on_grid&(f.volume>0)&(f.trade_count>0)]
        if closing.date.duplicated().any():raise ValueError('Duplicate regular close anchor')
        anchor=closing.set_index('date').close.reindex(d.index)
    else:anchor=d.first_open
    d['return']=d.last_close/anchor-1
    d['range']=d.maximum/d.minimum-1
    d['within_bar_realized_vol']=np.sqrt(d.squared_return)
    d['close_to_vwap']=d.last_close/(d.weighted_price/d.volume)-1
    d['log_dollar_volume']=np.log1p(d.dollar_notional)
    d['last_hour_return']=d.last_close/last_open-1
    d['late_volume_share']=(last_volume/d.volume).fillna(0.)
    d.loc[d.observed<4,FIELDS[:7]]=np.nan
    if ((d.coverage<0)|(d.coverage>1)|((d.first_trade_delay<0)|(d.first_trade_delay>1))|
        ((d.last_trade_age<0)|(d.last_trade_age>1))).any():raise ValueError('Invalid coverage or activity age')
    d['latest_bar_end']=d.last_end.astype(str);d=d.reset_index();d['date']=d.date.dt.strftime('%Y-%m-%d')
    return d[columns]


def build():
    if (OUT/'REVIEW.json').exists():raise ValueError('Completed extended-hours snapshot is immutable')
    manifest=json.loads((RAW/'manifest.json').read_text());data=load_data();dates=data['panel'].dates;tickers=data['panel'].tickers
    if manifest['completed_chunks']!=1288 or manifest['completed_chunks']!=manifest['chunk_count'] or manifest['feed']!='sip':raise ValueError('Incomplete SIP parent')
    for name,value in data['manifest']['hashes'].items():
        if sha(ROOT/data['manifest']['root']/name)!=value:raise ValueError('Reviewed economic inputs changed')
    calendar=calendar_frame(json.loads(CALENDAR.read_text()));np.testing.assert_array_equal(calendar.index,dates)
    aliases=data['manifest']['alias_map'];rules=json.loads(IDENTITY.read_text())['symbols']
    cubes={segment:np.full((len(dates),len(tickers),len(FIELDS)),np.nan,dtype='float32') for segment in ['after_hours','premarket']}
    counts={s:np.zeros((len(dates),len(tickers)),dtype='int16') for s in cubes}
    for cube in cubes.values():cube[...,7]=0.;cube[...,8:]=1.
    seen={s:set() for s in cubes};summaries=[];rejected=[];hashes={}
    for number,record in enumerate(manifest['chunks'],1):
        path=RAW/record['file'];symbol=record['query']['symbol'];security=aliases.get(symbol,symbol)
        if sha(path)!=record['sha256']:raise ValueError('Raw chunk hash changed')
        hashes[record['file']]=record['sha256'];frame=pd.read_csv(path)
        if len(frame)!=record['rows'] or (len(frame) and not frame.symbol.eq(symbol).all()):raise ValueError('Chunk identity/count mismatch')
        if symbol in rules and rules[symbol]['security_id']!=security:raise ValueError('Canonical identity mismatch')
        for segment in cubes:
            summary=summarize(frame,calendar,segment);summary,outside=filter_identity(summary,symbol,rules)
            if len(outside):rejected.append(outside.assign(source_symbol=symbol,security_id=security,segment=segment))
            summaries.append(summary.assign(source_symbol=symbol,security_id=security,segment=segment))
            if security not in tickers:continue
            node=tickers.index(security)
            for date,observed,*values in summary[['date','observed']+FIELDS].itertuples(index=False,name=None):
                key=security,date
                if key in seen[segment]:raise ValueError('Overlapping canonical extended sessions')
                seen[segment].add(key);day=dates.get_loc(pd.Timestamp(date))
                cubes[segment][day,node]=values;counts[segment][day,node]=observed
        if number%100==0:print(f'Reviewed extended-hours chunks {number}/1288',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'features.npz',**cubes,**{s+'_counts':v for s,v in counts.items()},
        dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str),tickers=np.asarray(tickers,dtype=str),feature_names=np.asarray(FIELDS,dtype=str))
    pd.concat(summaries,ignore_index=True).to_csv(OUT/'summaries.csv.gz',index=False,compression='gzip')
    reject=pd.concat(rejected,ignore_index=True) if rejected else pd.DataFrame(columns=['date','source_symbol','security_id','segment'])
    reject.to_csv(OUT/'rejected_sessions.csv',index=False)
    active=data['membership']&data['tradable']&(dates>=pd.Timestamp('2019-07-01'))[:,None]
    rows=[]
    for segment in cubes:
        for year in range(2019,2026):
            annual=active&(dates.year==year)[:,None]
            for node,ticker in enumerate(tickers):
                valid=annual[:,node]
                if valid.any():rows.append(dict(segment=segment,year=year,security_id=ticker,member_days=int(valid.sum()),
                    usable_days=int((counts[segment][valid,node]>=4).sum()),mean_coverage=float(cubes[segment][valid,node,7].mean())))
    pd.DataFrame(rows).to_csv(OUT/'coverage.csv',index=False)
    review=dict(status='passed_extended_hours_input_gate',source_chunks=len(hashes),raw_bars=manifest['rows'],
        sparse_rule='Four active on-grid bars; missing summaries remain missing without eligibility or label deletion',
        usable_member_fraction={s:float((counts[s][active]>=4).mean()) for s in cubes},
        mean_member_coverage={s:float(cubes[s][...,7][active].mean()) for s in cubes},
        source_hashes=hashes,raw_manifest_sha256=sha(RAW/'manifest.json'),calendar_sha256=sha(CALENDAR),
        identity_sha256=sha(IDENTITY),code_sha256=sha(__file__),feature_sha256=sha(OUT/'features.npz'),
        artifact_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},rejected_session_rows=len(reject),
        limitations=['Observed coverage measures active bars, not certified vendor completeness.',
                    'Retrospectively supplied trade bars are not an as-published quotes/latency database.',
                    'PM return intentionally excludes the prior-close gap to avoid cross-date raw split discontinuities.',
                    'No price or volume feature uses future regular-session data; AH anchor is the same-date regular close.'])
    for s in cubes:
        if review['usable_member_fraction'][s]<.25:raise ValueError('Too little extended-hours activity for the declared broad-universe study')
    (OUT/'REVIEW.json').write_text(json.dumps(review,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in review.items() if k!='source_hashes'},indent=2),flush=True)


if __name__=='__main__':build()
