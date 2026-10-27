"""Single fixed first-wave recipes; each fit is restricted to its train IDs."""
import numpy as np
from experiments.strategy_search.features import window_sum,peers
from experiments.strategy_search.storage import atomic,arrays,sha
from .calibration import ridge,predict,calibrate,fold_sessions,labels,edge
from .execution import integer_targets,round_trip_cost
from experiments.strategy_validation.config import Execution


def features(m):
    raw,valid=m['raw'],m['valid'];sessions=m['sessions']
    own5,ok5=window_sum(raw,valid,sessions,5)
    own15,ok15=window_sum(raw,valid,sessions,15)
    peer15,peerok=peers(own15[:,:250],ok15[:,:250],m['industry'][:250])
    peer1,p1ok=peers(raw[:,:250],valid[:,:250],m['industry'][:250])
    stall,stallok=window_sum(raw[:,:250]-peer1,valid[:,:250]&p1ok,sessions,3)
    breadth,_=peers(np.sign(own15[:,:250]),ok15[:,:250],m['industry'][:250])
    average=np.divide(np.where(ok5[:,:250],own5[:,:250],0).sum(1),ok5[:,:250].sum(1),out=np.zeros(len(sessions)),where=ok5[:,:250].sum(1)>0)
    return dict(own5=own5,own15=own15,ok=ok5&ok15,relative=own15[:,:250]-peer15,
        peerok=peerok,stall=stall,stallok=stallok,breadth=breadth,residual5=own5[:,:250]-average[:,None])


def learn_structure(m,f,train_ids):
    mask=np.isin(m['sessions'],train_ids)&(m['clock']>=60)&(m['clock']<=210)&(m['clock']%5==0)
    ix=np.flatnonzero(mask);lag=ix-5
    good=f['ok'][ix,:250]&f['ok'][lag,:250]
    a=np.where(good,f['residual5'][lag],0);b=np.where(good,f['residual5'][ix],0)
    def corr(x,y):
        x=x-x.mean(0);y=y-y.mean(0)
        return (x.T@y)/np.maximum(np.sqrt((x*x).sum(0)[:,None]*(y*y).sum(0)[None,:]),1e-15)
    c=corr(a,b);split=max(1,len(a)//2)
    stable=(corr(a[:split],b[:split])>0)&(corr(a[split:],b[split:])>0)
    group=m['industry'][:250];allowed=(group[:,None]==group[None,:])&~np.eye(250,dtype=bool)&stable&(c>0)
    c=np.where(allowed,c,-np.inf)
    for j in range(250):
        keep=np.argsort(-c[:,j],kind='stable')[:3];discard=np.ones(250,bool);discard[keep]=False;c[discard,j]=-np.inf
    leaders=np.argmax(c,axis=0);leader_ok=np.isfinite(c[leaders,np.arange(250)])
    med_volume=np.nanmedian(m['volume'][mask,:250]*m['close'][mask,:250],axis=0)
    peers3=np.zeros((250,3),int);peer_ok=np.zeros(250,bool);partner=np.zeros(250,int)
    rng=np.random.default_rng(20261027)
    for j in range(250):
        js=np.flatnonzero((group==group[j])&(np.arange(250)!=j))
        if len(js):
            ordered=js[np.argsort(abs(np.log(np.maximum(med_volume[js],1)/max(med_volume[j],1))),kind='stable')]
            partner[j]=rng.choice(ordered[:min(5,len(ordered))])
            if len(js)>=3:peers3[j]=ordered[:3];peer_ok[j]=True
            else:peers3[j]=np.resize(ordered,3)
    relative=f['relative'][mask]/np.maximum(m['vol'][mask,:250]*np.sqrt(15),1e-8)
    threshold_ok=f['ok'][mask,:250]&f['peerok'][mask]&m['eligible_full'][mask,:250]
    threshold_ok &= np.isfinite(m['vol'][mask,:250])&(m['vol'][mask,:250]>1e-8)
    extreme=np.nanquantile(abs(relative[threshold_ok]),.9) if threshold_ok.any() else np.inf
    own_z=f['own15'][mask,:250]/np.maximum(m['vol'][mask,:250]*np.sqrt(15),1e-8)
    own_extreme=np.nanquantile(abs(own_z[threshold_ok]),.9) if threshold_ok.any() else np.inf
    stability=np.zeros(250,bool)
    for j in range(250):
        h=m['stock_etf'][j];x=f['own5'][mask,h];y=f['own5'][mask,j]
        ok=f['ok'][mask,h]&f['ok'][mask,j]
        if ok.sum()>=100:
            cc=np.corrcoef(x[ok],y[ok])[0,1]
            beta=np.dot(x[ok],y[ok])/max(np.dot(x[ok],x[ok]),1e-15)
            stability[j]=np.isfinite(cc) and cc>=.1 and .5<=beta<=2
    return dict(leaders=leaders,leader_ok=leader_ok,peers=peers3,peer_ok=peer_ok,partner=partner,
        extreme=float(extreme),own_extreme=float(own_extreme),etf_stable=stability,train_ids=list(train_ids),links=int(leader_ok.sum()),
        graph=c,liquidity=med_volume)


def frame(m,f,structure,config,ids,with_labels=False):
    selected=np.isin(m['sessions'],ids)&(m['clock']>=60)&(m['clock']<=210)&(m['clock']%5==0)
    clocks=np.flatnonzero(selected);origins=np.repeat(clocks,250);stocks=np.tile(np.arange(250),len(clocks))
    j=stocks;i=origins;arm=config['arm'];design=config['design']
    if design=='D01':
        h=structure['partner'][j] if arm=='partner' else structure['leaders'][j]
        hedges=h[:,None];leader=f['own5'][i,h]
        if arm in ('self','no_graph'):leader=np.zeros_like(leader)
        relative=f['own15'][i,j]-f['own15'][i,h]
        setup=structure['leader_ok'][j] if arm!='partner' else np.ones(len(i),bool)
    elif design=='D03':
        hedges=structure['peers'][j];leader=f['own5'][i[:,None],hedges].mean(1)
        relative=f['relative'][i,j];z=relative/np.maximum(m['vol'][i,j]*np.sqrt(15),1e-8)
        setup=structure['peer_ok'][j]&(abs(z)>=structure['extreme'])&f['peerok'][i,j]
        if arm=='stall':setup &= f['stallok'][i,j]&(np.sign(relative)*f['stall'][i,j]<=0)&(np.sign(relative)*f['breadth'][i,j]<.65)
        if arm=='own':
            relative=f['own15'][i,j];leader=np.zeros(len(i))
            own_z=relative/np.maximum(m['vol'][i,j]*np.sqrt(15),1e-8)
            setup=structure['peer_ok'][j]&(abs(own_z)>=structure['own_extreme'])
        if arm=='momentum':
            relative=f['own15'][i,j]-f['relative'][i,j]
            setup=structure['peer_ok'][j]&f['peerok'][i,j]&(abs(relative)>0)
    else:
        h=structure['partner'][j] if arm=='stock_hedge' else m['stock_etf'][j]
        hedges=h[:,None];leader=f['own5'][i,h]
        relative=f['own15'][i,j]-f['own15'][i,h]
        setup=structure['etf_stable'][j]
        if arm=='etf_only':relative=f['own15'][i,h];leader=f['own5'][i,h]
        if arm=='stock_only':relative=f['own15'][i,j];leader=np.zeros(len(i))
    x=np.c_[f['own5'][i,j],f['own15'][i,j],leader,relative,m['vol'][i,j],f['breadth'][i,j]]
    if arm=='self':x[:,[2,3,5]]=0
    if arm=='no_graph':x[:,2]=0
    if arm=='etf_only':x[:,[0,1,4,5]]=0
    if design=='D03' and arm=='own':x[:,[2,5]]=0
    if design=='D03' and arm=='momentum':
        x[:,0]=leader;x[:,1]=relative;x[:,3]=relative;x[:,4]=0
    valid=m['eligible_full'][i,j]&f['ok'][i,j]&np.isfinite(x).all(1)
    valid &= m['eligible_full'][i[:,None],hedges].all(1)&f['ok'][i[:,None],hedges].all(1)
    out=dict(x=x,origin=i,stock=j,hedges=hedges,setup=setup&valid,
        session=m['sessions'][i],relative=relative)
    if with_labels:out['y']=labels(m,i,j,hedges,config['horizon'])
    return out


def direction_mask(config,relative,score):
    if config['design']!='D03':return np.ones(len(score),bool)
    wanted=np.sign(relative)*(1 if config['arm']=='momentum' else -1)
    return np.sign(score)==wanted


def fit_recipe(m,f,config,output,spread_bps=None):
    train=m['meta']['session_ids'][:60];oof=[];fold_audits=[]
    for k,(fit_ids,test_ids) in enumerate(fold_sessions(train)):
        structure=learn_structure(m,f,fit_ids)
        a=frame(m,f,structure,config,fit_ids,True);mask=a['setup']&np.isfinite(a['y'])
        # Fit on all causal eligible setups. No future endpoint availability in
        # decisions; missing endpoints only determine which training labels exist.
        if mask.sum()<100:
            fold_audits.append(dict(fold=k,status='insufficient_training_setups',train_ids=fit_ids,test_ids=test_ids,observations=int(mask.sum())));continue
        model=ridge(a['x'][mask],a['y'][mask]);del a
        b=frame(m,f,structure,config,test_ids,True);mask=b['setup']&np.isfinite(b['y'])
        scores=predict(model,b['x'][mask]);indices=np.flatnonzero(mask)
        admitted=direction_mask(config,b['relative'][indices],scores);indices=indices[admitted];scores=scores[admitted];out=b['y'][indices]
        oof.append((scores,out,b['session'][indices],b['origin'][indices],b['stock'][indices],b['hedges'][indices]))
        fold_audits.append(dict(fold=k,train_ids=fit_ids,test_ids=test_ids,observations=len(out),
            training_observations=model['observations'],links=structure['links'],
            purge='No label crosses a session boundary; training sessions strictly precede validation sessions',
            signed_prediction_mean=float(np.mean(np.sign(scores)*out)) if len(scores) else None))
    structure=learn_structure(m,f,train)
    a=frame(m,f,structure,config,train,True);mask=a['setup']&np.isfinite(a['y'])
    if mask.sum()<100:
        return dict(valid=False,reason='fewer_than100_training_setups',structure=structure,folds=fold_audits)
    model=ridge(a['x'][mask],a['y'][mask])
    if oof:
        score,y,sessions=(np.concatenate([r[k] for r in oof]) for k in range(3))
        calibration=calibrate(score,y,sessions)
    else:score=y=sessions=np.array([]);calibration=dict(valid=False,observations=0,sessions=0,bins=[])
    fixed=float(np.quantile(abs(score),.9)) if len(score) else np.inf
    admission=[]
    if oof and spread_bps is not None:
        # Calibrate the random gate's frequency to the TRAIN-ONLY economic
        # gate, not an arbitrary percentile or development activity target.
        for sc,yy,ss,origins,stocks,hs in oof:
            for origin in np.unique(origins):
                kk=np.flatnonzero(origins==origin)
                kk=kk[np.argsort(-abs(sc[kk]),kind='stable')[:3]]
                for k in kk:
                    js=[int(stocks[k]),*map(int,hs[k])];prices=m['open'][int(origin)+2,js]
                    side=int(np.sign(sc[k]));weights=np.r_[1,np.full(len(js)-1,1/(len(js)-1))]
                    capacity=np.maximum(0,np.floor(np.nan_to_num(m['volume'][int(origin)+1,js],nan=0)*.01))
                    qty=integer_targets(prices,np.r_[side,np.full(len(js)-1,-side)],weights,3990,100000,m['sector'][js],capacity=capacity)
                    mean,se=edge(calibration,sc[k]);notional=abs(qty[0]*prices[0])
                    cost=round_trip_cost(qty,prices,prices*spread_bps[js]/10000,spread_bps[js],config['horizon'],Execution())
                    admission.append(bool(qty.any() and np.isfinite(se) and notional*(mean-se)>cost['total']))
    fit=dict(valid=True,structure=structure,model=model,calibration=calibration,fixed_score=fixed,folds=fold_audits,
        gate_random_probability=float(np.mean(admission)) if admission else 0.,training_random_gate_examples=len(admission))
    arrays(output/(config['id']+'.oof.npz'),score=score,outcome=y,session=sessions)
    atomic(output/(config['id']+'.json'),dict(config=config,train_ids=train,folds=fold_audits,calibration=calibration,
        model=model,structure=structure,fixed_score=fixed,oof_sha256=sha(output/(config['id']+'.oof.npz')),
        valid=True,model_units='arithmetic relative return over same15/30-minute horizon; dollars per leading notional dollar',neural_used=False))
    return fit


def candidates(m,f,fit,config,gate='net_edge'):
    if not fit['valid']:return []
    a=frame(m,f,fit['structure'],config,m['meta']['session_ids'])
    pred=predict(fit['model'],a['x']);good=a['setup']&(pred!=0)
    # Frozen-model forecasts at completed decision clocks are consumed by exits
    # only when their own completed bar plus the delay has become available.
    m['decision_forecast_indices']=np.unique(a['origin'])
    m['decision_forecasts']=pred.reshape(-1,250)
    m['industry_residual15']=f['relative']
    good &= direction_mask(config,a['relative'],pred)
    rows=[]
    for k in np.flatnonzero(good):
        mean,se=edge(fit['calibration'],pred[k])
        # A broad pre-cost filter can only save work; final D08 dollar gate is
        # evaluated on feasible actual integer quantities at arrival.
        rows.append(dict(origin=int(a['origin'][k]),stock=int(a['stock'][k]),hedges=a['hedges'][k].tolist(),
            side=int(np.sign(pred[k])),prediction=float(pred[k]),edge_mean=float(mean),edge_se=float(se),
            relative=float(a['relative'][k]),entry_industry_residual=float(f['relative'][a['origin'][k],a['stock'][k]]),
            horizon=config['horizon'],gate=gate,design=config['design']))
    # Freeze a bounded queue of3 candidates per clock. Engine enforces occupied
    # baskets and the per-stock/day cap without looking at future exit success.
    grouped={}
    for r in rows:grouped.setdefault(r['origin'],[]).append(r)
    bounded=[r for i in sorted(grouped) for r in sorted(grouped[i],key=lambda r:(-abs(r['prediction']),r['stock']))[:3]]
    if gate=='fixed_score':bounded=[r for r in bounded if abs(r['prediction'])>=fit['fixed_score']]
    if gate=='matched_random':
        rng=np.random.default_rng(20261027);bounded=[r for r in bounded if rng.random()<fit['gate_random_probability']]
    return bounded
