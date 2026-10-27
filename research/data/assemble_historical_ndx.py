"""Assemble the declared free-data historical Nasdaq-100 research input.

The manifest stays pending until the separate review command validates inputs.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('data/nasdaq_monthly_pit')


def main():
    out=ROOT/'assembled';out.mkdir(parents=True,exist_ok=True)
    members=pd.read_csv(ROOT/'ndx_preflight/monthly_members_PROVISIONAL.csv')
    members=members[members.month>='2019-07-01'].copy()
    source=json.loads((ROOT/'yahoo_raw/manifest.json').read_text())
    aliases=source['aliases']
    native=set(source['native_tiingo_symbols'])|{'EA','WBD'}
    members['security_id']=members.ticker.map(lambda t:aliases.get(t,t))
    if members.duplicated(['month','security_id']).any():raise ValueError('Duplicate renamed security')
    ids=sorted(members.security_id.unique())
    read=lambda field:pd.read_csv(ROOT/'yahoo_raw'/(field+'.csv'),index_col=0,parse_dates=True)
    yc,yadj,yv=read('close'),read('adj_close'),read('volume')
    dates=yadj.index[yadj.SPY.notna()&yadj.QQQ.notna()]
    raw=pd.DataFrame(index=dates,columns=ids,dtype=float);economic=raw.copy();dv=raw.copy()
    source_records=[]
    for ticker in ids:
        if ticker in native:
            p=ROOT/'tiingo_raw'/(ticker+'_prices.json')
            f=pd.DataFrame(json.loads(p.read_text()))
            f.index=pd.to_datetime(f.date,utc=True).dt.tz_localize(None)
            if f.index.has_duplicates:raise ValueError('Duplicate price date')
            f=f.reindex(dates)
            raw[ticker]=f.close;economic[ticker]=f.adjClose;dv[ticker]=f.close*f.volume
            source_records.append(dict(security_id=ticker,provider='Tiingo',path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        else:
            raw[ticker]=yc[ticker].reindex(dates);economic[ticker]=yadj[ticker].reindex(dates)
            dv[ticker]=(yc[ticker]*yv[ticker]).reindex(dates)
            source_records.append(dict(security_id=ticker,provider='Yahoo Finance',path=str(ROOT/'yahoo_raw/adj_close.csv')))
    membership=pd.DataFrame(False,index=dates,columns=ids)
    for month,part in members.groupby('month'):
        mask=dates.to_period('M')==pd.Timestamp(month).to_period('M')
        membership.loc[mask,part.security_id]=True
    # Four acquisitions can occur while a monthly test snapshot still contains
    # the acquired stock. Economic marks include consideration, not stale quotes.
    events=[
      dict(security_id='XLNX',date='2022-02-14',cash_per_raw_share=0.,stock_ratio=1.7234,acquirer='AMD',
           source='https://ir.amd.com/financial-information/sec-filings/content/0000002488-22-000031/amd-20220214.htm'),
      dict(security_id='SGEN',date='2023-12-14',cash_per_raw_share=229.,stock_ratio=0.,acquirer=None,
           source='https://www.pfizer.com/news/press-release/press-release-detail/pfizer-completes-acquisition-seagen'),
      dict(security_id='SPLK',date='2024-03-18',cash_per_raw_share=157.,stock_ratio=0.,acquirer=None,
           source='https://newsroom.cisco.com/c/r/newsroom/en/us/a/y2024/m03/cisco-completes-acquisition-of-splunk.html'),
      dict(security_id='ANSS',date='2025-07-17',cash_per_raw_share=199.91,stock_ratio=.3399,acquirer='SNPS',
           source='https://www.sec.gov/Archives/edgar/data/883241/000114036125026139/ef20051970_8k.htm')]
    tradable=raw.notna() & raw.gt(0)
    for event in events:
        ticker=event['security_id'];date=pd.Timestamp(event['date'])
        before=raw.loc[raw.index<date,ticker].dropna().index[-1]
        factor=economic.at[before,ticker]/raw.at[before,ticker]
        stock_value=0. if event['acquirer'] is None else event['stock_ratio']*yc.at[date,event['acquirer']]
        payout=event['cash_per_raw_share']+stock_value
        if not np.isfinite(payout) or payout<=0:raise ValueError('Invalid acquisition consideration')
        event.update(last_pre_completion_price_date=str(before.date()),
                     payout_raw_dollars_at_completion=float(payout),adjusted_unit_factor=float(factor),
                     stock_fraction_at_completion=float(stock_value/payout))
        economic.loc[date:,ticker]=payout*factor
        raw.loc[date:,ticker]=np.nan;dv.loc[date:,ticker]=np.nan;tradable.loc[date:,ticker]=False
    # These earlier disappearances never create test holdings. Unpriced merger
    # outcomes remain missing labels rather than invented cash/share settlements.
    earlier_halts={'ALXN':'2021-07-21','CELG':'2019-11-20','CERN':'2022-06-08',
                   'CTXS':'2022-09-30','MXIM':'2021-08-26','MYL':'2020-11-16'}
    for ticker,date in earlier_halts.items():
        for f in [raw,economic,dv]:f.loc[pd.Timestamp(date):,ticker]=np.nan
        tradable.loc[pd.Timestamp(date):,ticker]=False
    unknown=[]
    for ticker in ids:
        active=membership[ticker] & ~tradable[ticker]
        # Missing cells are reviewed as IPO/when-issued warmup or completed
        # acquisitions. Never silently remove a stock from the membership list.
        if active.any():unknown.append(dict(security_id=ticker,first_missing=str(dates[active][0].date()),
                                           last_missing=str(dates[active][-1].date()),cells=int(active.sum())))
    frames=dict(raw_close=raw,economic_close=economic,dollar_volume=dv,tradable=tradable,
                membership=membership,etf_adj_close=yadj.loc[dates,['SPY','QQQ']],
                etf_dollar_volume=(yc*yv).loc[dates,['SPY','QQQ']])
    for field,frame in frames.items():frame.to_csv(out/(field+'.csv'))
    members.to_csv(out/'monthly_members.csv',index=False)
    manifest=dict(status='pending_review_no_training',root=str(out),training_start='2019-07-01',
                  monthly_snapshots=members.month.nunique(),securities=len(ids),
                  source_membership=json.loads((ROOT/'ndx_preflight/manifest.json').read_text()),
                  sources=source_records,test_settlements=events,earlier_unpriced_mergers=earlier_halts,
                  missing_active_prices=unknown,alias_map=aliases,
                  limitations=['Research-grade reconstructed historical index snapshots, not licensed as-published index data.',
                               'Training begins July 2019; first 2022 fold has 1.5 training years and one prior validation year.',
                               'Monthly snapshots retain official share classes, so security counts can exceed 100.',
                               'Adjusted closes approximate reinvested total returns; raw close times volume supplies liquidity.',
                               'Cash acquisition proceeds are available at completion close; stock consideration is liquidated at that close with fees. Actual broker settlement can be delayed.',
                               'Earlier unpriced merger outcomes are omitted only from supervised targets, with counts reported; signal-time members are retained.',
                               '2025 SOLS sources disagree on off-cycle dates but agree on the monthly November snapshot; it lacks the neural/momentum warmup. No off-cycle additions enter the monthly universe.',
                               '2022-2025 was examined previously; results remain exploratory. No outperforming model is promised.'],
                  hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(securities=len(ids),monthly_snapshots=manifest['monthly_snapshots'],
                         missing_active_prices=unknown,status=manifest['status']),indent=2))


if __name__=='__main__':main()
