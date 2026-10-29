"""Reconstruct forecast calibration/signals and replay value-position accounts."""
import json
import numpy as np
import pandas as pd
import run_ndx_model_hedge as study
from ndx_drawdown_engine import allocations,daily_allocation,risk_inputs,exposure_series,market_inputs
from ndx_matched_controls import etf_data
from ndx_weak_market_short import append_qqq
from audit_ndx_weak_market_short import value_replay


def audit():
    study.verify()
    complete=json.loads((study.RUN/'COMPLETE.json').read_text())
    for name,expected in complete['output_sha256'].items():assert study.parent.sha(study.RUN/name)==expected
    data=study.parent.load_data()
    ids=np.concatenate([study.parent.fold(data,y)[2] for y in [2022,2023,2024,2025]])
    scores=study.parent.saved_scores(data,ids);market=market_inputs(data)
    p=pd.Series(market['qqq']);ma=(p<p.rolling(200).mean()).to_numpy()[data['origin'][ids]]
    combined=append_qqq(data,etf_data(data,'QQQ'))
    saved_signals=pd.read_csv(study.RUN/'signals.csv')
    saved_cal=pd.read_csv(study.RUN/'calibration.csv')
    saved_targets=pd.read_csv(study.RUN/'targets.csv')
    prepared={};max_bias_error=0.;checked_flags=0
    for group,seeds in study.parent.GROUPS.items():
        w=allocations(data,ids,scores[group],'issuer5')
        risk=risk_inputs(data,daily_allocation(data,ids,w))
        e=exposure_series(study.RULES['vol20'],market,risk)[data['origin'][ids]]
        flags={name:[] for name in study.SIGNALS};cursor=0
        for year in [2022,2023,2024,2025]:
            _,val,test=study.parent.fold(data,year)
            family=[];bias=[]
            for name,folder in study.FAMILIES:
                vp=[];tp=[]
                for seed in seeds:
                    with np.load(folder/str(year)/f'{seed}_predictions.npz') as s:
                        vp.append(s['validation'].astype(float));tp.append(s['test'].astype(float))
                vp=sum(vp)/len(seeds);tp=sum(tp)/len(seeds)
                errors=[]
                for row,i in enumerate(val):
                    active=np.flatnonzero(data['labelmask'][i])
                    errors.append(float(np.mean(vp[row,active]-100*data['y'][i,active].astype(float))))
                bias.append(float(np.mean(errors)));family.append(tp)
                expected=saved_cal[(saved_cal.group==group)&(saved_cal.year==year)&(saved_cal.family==name)].iloc[0]
                err=abs(expected.bias_pp-bias[-1]);max_bias_error=max(max_bias_error,err);assert err<1e-12
                assert data['exit'][val].max()<data['origin'][test].min()
            for row,i in enumerate(test):
                active=np.flatnonzero(data['mask'][i]);weights=w[cursor+row]
                expected=saved_signals[(saved_signals.group==group)&(saved_signals.fill_date==str(data['panel'].dates[data['entry'][i]].date()))].iloc[0]
                for prefix,forecasts in [('raw',family),('cal',[family[j]-bias[j] for j in range(2)])]:
                    average=(forecasts[0][row]+forecasts[1][row])/2
                    values={'mean':np.mean(average[active])<0,
                            'basket':np.dot(average[weights>0],weights[weights>0])<0,
                            'breadth60':np.count_nonzero(average[active]<0)/len(active)>=.6,
                            'agreement':all(np.mean(v[row,active])<0 for v in forecasts)}
                    for suffix,value in values.items():
                        name=prefix+'_'+suffix
                        assert bool(expected[name])==value
                        flags[name].append(value);checked_flags+=1
            cursor+=len(test)
        for spec in study.recipes():
            on=np.array(flags[spec['signal']],bool) if spec['kind']=='model' else ma if spec['signal']=='qqq_ma200' else np.ones(len(e),bool) if spec['signal']=='always' else np.zeros(len(e),bool)
            long=e.copy();short=np.zeros(len(e));mode=spec['mode']
            cap=.2/np.maximum(market['qqqvol20'][data['origin'][ids]],.05)
            if mode=='cash':long[on]=0.
            elif mode=='switch50':short[on]=np.minimum(.5,cap[on]);long[on]=0.
            elif mode in ['hedge25','hedge50']:
                short[on]=np.minimum(np.minimum(int(mode[5:])/100,1-long[on]),cap[on])
            target=np.column_stack([w*long[:,None],-short])
            prepared[spec['id'],group]=target
            expected=saved_targets[(saved_targets.recipe==spec['id'])&(saved_targets.group==group)]
            np.testing.assert_allclose(expected.long_target,target[:,:-1].sum(1),atol=1e-13)
            np.testing.assert_allclose(expected.short_target,-target[:,-1],atol=1e-13)
    m=pd.read_csv(study.RUN/'metrics.csv')
    base=m[(m.cost_multiplier==1)&(m.borrow_rate==.03)]
    selected=['baseline','qqq_ma200/hedge50','always/hedge50','raw_mean/hedge50','cal_mean/hedge50',
              'cal_basket/switch50','raw_agreement/cash']
    for group in study.parent.GROUPS:
        selected += base[(base.group==group)&(base.kind=='model')].nlargest(2,'sharpe').recipe.tolist()
    selected=list(dict.fromkeys(selected))
    sample=m[m.recipe.isin(selected)&(((m.cost_multiplier==1)&(m.borrow_rate==.03))|
             ((m.group=='ten_seed')&((m.cost_multiplier==4)|(m.borrow_rate==.1))))]
    replay=[]
    with np.load(study.RUN/'paths.npz') as s:paths=dict(zip(s['ids'].astype(str),s['returns'].T))
    for row in sample.itertuples():
        actual=value_replay(combined,ids,prepared[row.recipe,row.group],row.borrow_rate,row.cost_multiplier)
        error=float(np.max(np.abs(1e6*np.cumprod(1+actual)-1e6*np.cumprod(1+paths[row.id]))))
        assert error<1e-5,(row.id,error)
        replay.append(dict(id=row.id,max_nav_error=error))
    result=dict(status='passed',signal_flags_rebuilt=checked_flags,calibrations=24,
                max_calibration_error=max_bias_error,all_111_target_schedules_rebuilt=True,
                replayed_accounts=len(replay),max_nav_error=max(r['max_nav_error'] for r in replay),
                accounts=replay,source_sha256=study.parent.sha(__file__),
                limitation='Independent signal/calibration arithmetic and value-position account replay. Same vendor inputs and forecasts; no broker or prospective validation.')
    study.parent.write(study.RUN/'AUDIT.json',result);study.parent.write(study.DOC/'AUDIT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='accounts'},indent=2),flush=True)


if __name__=='__main__':audit()
