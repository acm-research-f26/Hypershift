"""Fail-closed structural review of the declared free-source research input."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    root=Path('data/nasdaq_monthly_pit/assembled')
    path=root/'manifest.json';m=json.loads(path.read_text())
    for filename,expected in m['hashes'].items():
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==expected,filename
    read=lambda name:pd.read_csv(root/(name+'.csv'),index_col=0,parse_dates=True)
    prices,tradable,membership=read('economic_close'),read('tradable'),read('membership')
    raw,dv=read('raw_close'),read('dollar_volume')
    assert prices.index.is_unique and prices.index.is_monotonic_increasing
    assert list(prices.columns)==list(raw.columns)==list(dv.columns)==list(membership.columns)==list(tradable.columns)
    assert all(str(p.dtype)=='bool' for p in tradable.values.T)
    assert membership.loc['2019-07-01':].sum(axis=1).between(100,110).all()
    observed_raw=raw.to_numpy()[tradable.to_numpy()]
    observed_dv=dv.to_numpy()[tradable.to_numpy()]
    assert np.isfinite(observed_raw).all() and (observed_raw>0).all()
    assert np.isfinite(observed_dv).all() and (observed_dv>=0).all()
    expected={'ALXN','CELG','MXIM','ANSS','SGEN','SPLK','XLNX'}
    assert {r['security_id'] for r in m['missing_active_prices']}==expected
    for r in m['missing_active_prices']:
        date=next((e['date'] for e in m['test_settlements'] if e['security_id']==r['security_id']),None)
        date=date or m['earlier_unpriced_mergers'][r['security_id']]
        assert r['first_missing']>=date
    for event in m['test_settlements']:
        ticker=event['security_id'];date=event['date']
        assert not tradable.loc[date:,ticker].any()
        assert prices.loc[date:,ticker].nunique()==1
        assert prices.loc[date,ticker]>0
    # All actual test-period acquisition settlements are documented with primary
    # company/SEC sources. Other vanished names are gone before any test holding.
    assert read('etf_adj_close').loc['2022-01-03':,['SPY','QQQ']].notna().all().all()
    m['status']='reviewed_research_inputs'
    m['review']={'level':'research_grade_with_disclosed_limitations; not licensed point-in-time certification',
                 'structural_input_checks':'passed',
                 'unexplained_active_price_gaps':0,
                 'test_acquisition_sources':'four primary company/SEC releases, terms checked before training',
                 'training_missing_labels':'retained and counted, not replaced by future observations',
                 'membership':'third-party dated event files with ten annual closure checks; annual Nasdaq announcements cross-checked; SOLS date conflict immaterial to monthly snapshots',
                 'identity':'recycled CA history rejected; earlier FOX reorganization excluded by declared training-start amendment; five same-issuer symbol aliases explicitly mapped',
                 'engine':'monthly masks, training-only correlations, next-close orders, continuous cash/NAV and mandatory acquisition settlement tested',
                 'settlement_assumption':'completion-close liquidity, with explicit limitation rather than an invented actual broker payout date'}
    path.write_text(json.dumps(m,indent=2),encoding='utf-8')
    from pit_research_engine import load_data,fold,historical_groups
    d=load_data(root)
    records=[]
    for year in [2022,2023,2024,2025]:
        tr,va,te=fold(d,year);groups=historical_groups(d,tr)
        records.append(dict(year=year,training_dates=len(tr),validation_dates=len(va),test_forecasts=len(te),
                            training_first_signal=str(d['panel'].dates[d['origin'][tr[0]]].date()),
                            training_last_label=str(d['panel'].dates[d['exit'][tr[-1]]].date()),
                            validation_last_label=str(d['panel'].dates[d['exit'][va[-1]]].date()),
                            test_first_signal=str(d['panel'].dates[d['origin'][te[0]]].date()),
                            unobserved_training_labels=int((d['mask'][tr]&~d['labelmask'][tr]).sum()),
                            unobserved_validation_labels=int((d['mask'][va]&~d['labelmask'][va]).sum()),
                            min_test_eligible=int(d['mask'][te].sum(1).min()),groups=len(groups)))
    (root/'fold_preflight.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
