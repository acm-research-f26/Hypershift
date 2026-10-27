"""Frozen own-model hedge experiment, original-five/new-five/ten-seed groups."""
import argparse
from datetime import datetime,timezone
import json
import shutil
import numpy as np
import pandas as pd
import run_ndx_drawdown_research as parent
import run_ndx_weak_market_short as previous
from run_ndx_matched_controls import verify_sealed_inputs,RULES
from ndx_drawdown_engine import allocations,daily_allocation,risk_inputs,exposure_series,market_inputs
from ndx_matched_controls import etf_data,paired_inference
from ndx_weak_market_short import targets,append_qqq,simulate_signed,weak_signals
from ndx_model_hedge import SIGNALS,validation_bias,forecast_signals
from research_engine import performance

ROOT=parent.ROOT
RUN=ROOT/'runs/ndx_model_hedge_20261027'
DOC=ROOT/'docs/ndx_model_hedge_20261027'
MODES=['cash','hedge25','hedge50','switch50']
FAMILIES=[('think',ROOT/'runs/ndx_extended_hours_20261024/after_hours_premarket'),
          ('dynamic',ROOT/'runs/ndx_reference_networks_20261024/dynamic_restricted')]


def recipes():
    output=[dict(id=f'{signal}/{mode}',signal=signal,mode=mode,kind='model') for signal in SIGNALS for mode in MODES]
    output += [dict(id='baseline',signal='none',mode='baseline',kind='control')]
    output += [dict(id='qqq_ma200/'+mode,signal='qqq_ma200',mode=mode,kind='control') for mode in ['cash','hedge50','switch50']]
    output += [dict(id='always/hedge50',signal='always',mode='hedge50',kind='control')]
    return output


def freeze():
    previous.verify()
    locks=verify_sealed_inputs()
    for relative in ['config/ndx_prospective_candidate_20261027.json',
                     'runs/ndx_weak_market_short_20261027/batch2/PROTOCOL.json',
                     'runs/ndx_weak_market_short_20261027/batch2/COMPLETE.json',
                     'runs/ndx_weak_market_short_20261027/batch2/paths.npz',
                     'runs/ndx_weak_market_short_20261027/batch2/metrics.csv']:
        locks[relative]=parent.sha(ROOT/relative)
    names=['research/core/ndx_model_hedge.py','research/analysis/run_ndx_model_hedge.py',
           'research/evaluation/audit_ndx_model_hedge.py','tests/test_ndx_model_hedge.py',
           'research/core/ndx_weak_market_short.py','research/core/ndx_matched_controls.py',
           'research/evaluation/audit_ndx_weak_market_short.py']
    plan=dict(study='Own-model hedge signals from frozen raw five-session stock return forecasts',
        date='2026-10-27',history='2022-2025 repeatedly inspected development; no unseen confirmation',
        recipes=recipes(),groups=parent.GROUPS,accounts=495,
        forecasts='Per family/year average raw percentage-point predictions over fixed seeds. Equal50:50 raw family average for mean, selected-basket and negative-breadth signals. Stock selection still uses the unchanged50:50 percentile-rank blend and five issuers.',
        signals={'raw_mean':'Eligible-stock average predicted return <0',
                 'raw_basket':'Proposed equal-weight five-issuer basket predicted return <0',
                 'raw_breadth60':'At least60% of eligible stocks have negative blended predictions',
                 'raw_agreement':'Both family eligible-stock mean forecasts <0',
                 'cal_*':'Same four rules after subtracting prior-year validation mean forecast error independently per family/group/year. Bias averages eligible labeled errors within each date, then equally across dates. Fixed intercept only; no return-fitted thresholds/slopes.'},
        calibration='Only each annual fit own saved validation predictions, actual y*100 and labelmask; assert all validation outcomes precede first test signal. No test outcome, realized QQQ direction, test-return threshold optimization or new fitting. Calibration is a postprocessing estimate; checkpoints were selected on the same prior-year validation fold.',
        baseline='Five issuers, weekly20-session basket volatility target20%, exposure<=1; unused zero-interest cash. Same baseline for every signal and seed group. No QQQ trend cap in model strategy.',
        hedges='Keep baseline longs for hedge25/50, short QQQ up to min(cap,unused capital,20%/QQQ20-session vol). cash/switch50 exit basket when signal weak; switch50 shorts QQQ with cap50% and20% vol target. QQQ volatility sizes the hedge, but model predictions alone trigger model rules.',
        timing='All model predictions available09:20ET using AH/PM bars ending<=09:15, entry regular-close fill. Quantities use prior-close NAV/marks/ADV. Fixed inherited five-session dates. Validation data strictly pre-test. No same-close forecast execution.',
        costs=[1,2,4],borrow_base=.03,borrow_stress=[.01,.10],
        account_count='37recipes x3groups x3costs=333;27short recipes x3groups x2additional borrow assumptions=162;total495. All rules and failures disclosed.',
        controls='Unhedged baseline, MA200cash/hedge50/switch50, always-on hedge50 to distinguish signal timing from generic exposure reduction. Prior study paths must reproduce exactly for shared controls.',
        funding='Use sealed corrected signed engine: gross target<=1,1%ADV caps, calendar-day borrow,28-day short-fee reserve before buys, signed total-return/dividend approximation, no cash interest/rebate. Realized drift possible. Actual locates/recalls/margin/tax/auction conditions absent.',
        inference='All32model variants vs baseline and8switches vs matching modelcash, across all3groups:120 simultaneous active-mean contrasts.10000joint circular-block samples at5/20/60sessions;20primary. SR/CAGR difference intervals and selected absoluteSR intervals pointwise, conditional on fits; no adjustment for earlier research or retrospective leader selection.',
        publication='Report highest observed model-signalSR separately for original-five/ten-seed, compare QQQ/always/baseline, retain new-five diagnostic and all failures. Update charts.md top and existing xlsx using reviewable audited results, preserving prior findings. Prospective policy unchanged.',
        stop='One finite eight-signal/four-mode grid. No expansion from outcomes,2026 returns,new training,trades,automation or prospective activation.',
        source_sha256={n:parent.sha(ROOT/n) for n in names},input_sha256=locks)
    path=RUN/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Frozen study changed')
    else:
        parent.write(path,plan)
        parent.write(RUN/'FREEZE.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                     protocol_sha256=parent.sha(path),financial_results_absent=not (RUN/'metrics.csv').exists()))
    print('Frozen eight model signals,37total recipes,495accounts',flush=True)


def verify():
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    assert parent.sha(RUN/'PROTOCOL.json')==json.loads((RUN/'FREEZE.json').read_text())['protocol_sha256']
    for section in ['source_sha256','input_sha256']:
        for relative,expected in plan[section].items():
            if parent.sha(ROOT/relative)!=expected:raise ValueError('Frozen file changed: '+relative)
    return plan


def prepare():
    data=parent.load_data()
    assert data['panel'].dates[-1].year==2025
    ids=np.concatenate([parent.fold(data,y)[2] for y in [2022,2023,2024,2025]])
    scores=parent.saved_scores(data,ids);market=market_inputs(data)
    combined=append_qqq(data,etf_data(data,'QQQ'));gates=weak_signals(market['qqq'])
    prepared={};diagnostics=[];calibration=[]
    for group,seeds in parent.GROUPS.items():
        w=allocations(data,ids,scores[group],'issuer5')
        risk=risk_inputs(data,daily_allocation(data,ids,w))
        exposure=exposure_series(RULES['vol20'],market,risk)
        flags={name:[] for name in SIGNALS};cursor=0
        for year in [2022,2023,2024,2025]:
            _,va,te=parent.fold(data,year)
            assert data['exit'][va].max()<data['origin'][te].min()
            families=[];biases=[]
            for name,folder in FAMILIES:
                vp=[];tp=[]
                for seed in seeds:
                    with np.load(folder/str(year)/f'{seed}_predictions.npz') as p:
                        vp.append(p['validation'].astype(float));tp.append(p['test'].astype(float))
                val=np.mean(vp,axis=0);test=np.mean(tp,axis=0)
                assert val.shape==data['y'][va].shape and test.shape==data['y'][te].shape
                bias=validation_bias(val,data['y'][va].astype(float)*100,data['labelmask'][va])
                families.append(test);biases.append(bias)
                calibration.append(dict(group=group,year=year,family=name,bias_pp=bias,
                    validation_dates=len(va),last_validation_outcome=str(data['panel'].dates[data['exit'][va].max()].date()),
                    first_test_signal=str(data['panel'].dates[data['origin'][te].min()].date())))
            current,detail=forecast_signals(families,data['mask'][te],w[cursor:cursor+len(te)],biases)
            for name in SIGNALS:flags[name].extend(current[name])
            for j,i in enumerate(te):
                diagnostics.append(dict(group=group,year=year,signal_date=str(data['panel'].dates[data['origin'][i]].date()),
                    fill_date=str(data['panel'].dates[data['entry'][i]].date()),
                    **{k:float(v[j]) for k,v in detail.items()},**{k:bool(v[j]) for k,v in current.items()}))
            cursor+=len(te)
        for spec in recipes():
            weak=np.zeros(len(exposure),bool)
            if spec['kind']=='model':weak[data['origin'][ids]]=flags[spec['signal']]
            elif spec['signal']=='qqq_ma200':weak=gates['ma200']
            elif spec['signal']=='always':weak[:]=True
            prepared[spec['id'],group]=targets(w,exposure,data['origin'][ids],weak,market['qqqvol20'],spec['mode'])
    return data,ids,combined,prepared,pd.DataFrame(diagnostics),pd.DataFrame(calibration)


def run():
    plan=verify()
    if (RUN/'COMPLETE.json').exists():raise ValueError('Completed study; refusing overwrite')
    data,ids,combined,prepared,signals,cal=prepare()
    signals.to_csv(RUN/'signals.csv',index=False);cal.to_csv(RUN/'calibration.csv',index=False)
    dates=data['panel'].dates[int(data['entry'][ids[0]]):]
    old={}
    with np.load(previous.RUN/'paths.npz') as p:old=dict(zip(p['ids'].astype(str),p['returns'].T))
    metrics=[];paths={};annual=[];parity=[]
    target_info=[]
    for k,s in enumerate(plan['recipes']):
        for group in parent.GROUPS:
            w=prepared[s['id'],group]
            for j,i in enumerate(ids):
                target_info.append(dict(recipe=s['id'],group=group,fill_date=str(data['panel'].dates[data['entry'][i]].date()),
                                   long_target=float(w[j,:-1].sum()),short_target=float(-w[j,-1])))
            settings=[(c,.03) for c in [1,2,4]]
            if s['mode'] not in ['cash','baseline']:settings += [(1,b) for b in [.01,.10]]
            for cost,borrow in settings:
                key=f"{s['id']}/{group}/c{cost}/b{borrow}"
                m,path,_=simulate_signed(combined,ids,w,borrow,cost)
                paths[key]=path
                metrics.append(dict(id=key,recipe=s['id'],signal=s['signal'],mode=s['mode'],kind=s['kind'],
                                    group=group,cost_multiplier=cost,borrow_rate=borrow,**m))
                for year in [2022,2023,2024,2025]:
                    annual.append(dict(id=key,year=year,**performance(path[dates.year==year])))
                if s['signal'] in ['none','qqq_ma200']:
                    oldrecipe='vol20/baseline' if s['signal']=='none' else 'vol20/ma200/'+s['mode']
                    previouspath=old[f'{oldrecipe}/{group}/c{cost}/b{borrow}']
                    error=float(np.max(np.abs(path-previouspath)))
                    assert error<1e-12
                    parity.append(dict(id=key,max_return_error=error))
        if k%6==0:print(f'Completed{k+1}/37recipes,{len(metrics)}accounts',flush=True)
    frame=pd.DataFrame(metrics);assert len(frame)==495
    frame.to_csv(RUN/'metrics.csv',index=False);pd.DataFrame(annual).to_csv(RUN/'annual.csv',index=False)
    pd.DataFrame(target_info).to_csv(RUN/'targets.csv',index=False)
    np.savez_compressed(RUN/'paths.npz',ids=np.array(list(paths)),dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str),returns=np.column_stack(list(paths.values())))
    base=frame[(frame.cost_multiplier==1)&(frame.borrow_rate==.03)]
    base.sort_values(['group','sharpe'],ascending=[True,False]).to_csv(RUN/'base_rankings.csv',index=False)
    names=[f"{s['id']}/{g}" for g in parent.GROUPS for s in plan['recipes']]
    contrasts=[]
    for g in parent.GROUPS:
        contrasts += [(s['id']+'/'+g,'baseline/'+g) for s in plan['recipes'] if s['kind']=='model']
        contrasts += [(s['id']+'/'+g,s['signal']+'/cash/'+g) for s in plan['recipes'] if s['kind']=='model' and s['mode']=='switch50']
    assert len(contrasts)==120
    matrix=np.column_stack([paths[n+'/c1/b0.03'] for n in names]);inference=[]
    for block in [5,20,60]:
        print(f'Uncertainty:block{block},120contrasts,10000draws',flush=True)
        inference.append(paired_inference(matrix,names,contrasts,block))
    pd.concat(inference).to_csv(RUN/'uncertainty.csv',index=False)
    # Independently remeasure all daily paths and preserve unchanged study policy.
    for row in metrics:
        for field,value in performance(paths[row['id']]).items():assert abs(row[field]-value)<1e-10
        assert abs(row['terminal_nav']-1e6*np.prod(1+paths[row['id']]))<1e-5
    verify()
    parent.write(RUN/'ACCOUNT_CHECKS.json',dict(status='passed',accounts=495,prior_path_parity=parity,
                 max_self_financing_error=max(r['max_accounting_error'] for r in metrics)))
    DOC.mkdir(parents=True,exist_ok=True)
    for name in ['PROTOCOL.json','FREEZE.json','metrics.csv','annual.csv','targets.csv','base_rankings.csv',
                 'uncertainty.csv','signals.csv','calibration.csv','ACCOUNT_CHECKS.json']:
        shutil.copyfile(RUN/name,DOC/name)
    parent.write(RUN/'COMPLETE.json',dict(status='complete',timestamp_utc=datetime.now(timezone.utc).isoformat(),
                 output_sha256={p.name:parent.sha(p) for p in RUN.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
    shutil.copyfile(RUN/'COMPLETE.json',DOC/'COMPLETE.json')
    for group in parent.GROUPS:
        print(group,flush=True)
        print(base[(base.group==group)&(base.kind=='model')].sort_values('sharpe',ascending=False)[['recipe','sharpe','annualized_return','maximum_drawdown']].head(4).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['freeze','run'],required=True)
    freeze() if p.parse_args().stage=='freeze' else run()
