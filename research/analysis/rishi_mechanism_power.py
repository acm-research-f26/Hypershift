"""Conditional paired-date planning power; no prospective activation."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

RF=1.03**(1/252)-1


def stationary_draws(source_length,reps,length,block,rng):
    result=np.empty((reps,length),dtype=np.int32);result[:,0]=rng.integers(source_length,size=reps)
    for t in range(1,length):
        restart=rng.random(reps)<1/block
        result[:,t]=np.where(restart,rng.integers(source_length,size=reps),(result[:,t-1]+1)%source_length)
    return result


def holm_rejections(p):
    p=np.asarray(p,float)
    if p.ndim!=2 or p.shape[1]!=2 or np.any((p<0)|(p>1)):raise ValueError('Two finite p values required')
    order=np.argsort(p,axis=1);sorted_p=np.take_along_axis(p,order,axis=1)
    first=sorted_p[:,0]<=.025;second=first&(sorted_p[:,1]<=.05)
    result=np.zeros_like(p,dtype=bool)
    np.put_along_axis(result,order,np.column_stack([first,second]),axis=1)
    return result


def upper_p(reference,observed):
    reference=np.sort(reference)
    return (len(reference)-np.searchsorted(reference,observed,side='left')+1)/(len(reference)+1)


def planning_tape(raw):
    raw=np.asarray(raw,float);sd=raw.std(0,ddof=1)
    if raw.ndim!=2 or raw.shape[1]!=3 or not np.isfinite(raw).all() or np.any(sd<=0):raise ValueError('Three variable paired paths required')
    return raw-raw.mean(0)+RF+.5*sd/np.sqrt(252)


def sample_statistics(tape,horizon,reps,block,rng):
    means=[];sharpe=[];ratios=[];population_mean=tape.mean(0);population_sd=tape.std(0,ddof=1)
    for start in range(0,reps,500):
        ids=stationary_draws(len(tape),min(500,reps-start),horizon,block,rng)
        samples=tape[ids];mean=samples.mean(1);sd=samples.std(1,ddof=1)
        if np.any(sd<=0):raise ValueError('Degenerate simulated path')
        sr=(mean-RF)*np.sqrt(252)/sd
        means.append(252*(mean[:,[0]]-mean[:,1:]-(population_mean[0]-population_mean[1:])))
        sharpe.append(sr[:,[0]]-sr[:,1:]);ratios.append(population_sd[0]/sd[:,0])
    return dict(mean=np.concatenate(means),sr=np.concatenate(sharpe),left_sd_ratio=np.concatenate(ratios))


def nw_mean(path,lag):
    path=np.asarray(path,float);n=len(path);center=path-path.mean()
    long_variance=float(center@center/n)
    for j in range(1,lag+1):long_variance+=2*(1-j/(lag+1))*float(center[j:]@center[:-j]/n)
    se=252*np.sqrt(max(0,long_variance)/n);mean=float(path.mean()*252)
    return dict(annual_mean=mean,annual_se=float(se),low=mean-1.95996398454*se,high=mean+1.95996398454*se,
                normal_p_one_sided=.5*math.erfc(mean/se/np.sqrt(2)) if se>0 else None)


def main():
    out=Path('docs/rishi_mechanism_20261027/execution_batch7_power')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS/ACCOUNT_CURVES.csv')
    frame=pd.read_csv(source);pieces=[frame[frame.id==f'{m}/ten_seed/10'] for m in ['E','B','E_Hlabel']]
    if any(p.date.tolist()!=pieces[0].date.tolist() for p in pieces):raise ValueError('Paired dates')
    raw=np.column_stack([p.net.to_numpy() for p in pieces]);tape=planning_tape(raw)
    rng=np.random.default_rng(27102031);rows=[];nulls=[]
    for block in [21,5,63]:
        for horizon in [252,504,1008]:
            reference=sample_statistics(tape,horizon,10000,block,rng)
            simulated=sample_statistics(tape,horizon,10000,block,rng)
            for endpoint,effects in [('mean',[0.,.01,.02]),('sr',[0.,.1,.2])]:
                for j,right in enumerate(['B','E_Hlabel']):
                    low,high=np.quantile(reference[endpoint][:,j],[.025,.975])
                    nulls.append(dict(block=block,sessions=horizon,endpoint=endpoint,right=right,low=float(low),high=float(high)))
                for effect in effects:
                    alternative=simulated[endpoint]+(effect if endpoint=='mean' else effect*simulated['left_sd_ratio'][:,None])
                    p=np.column_stack([upper_p(reference[endpoint][:,j],alternative[:,j]) for j in range(2)])
                    reject=holm_rejections(p)
                    for name,indicator in [('E_minus_B',reject[:,0]),('E_minus_E_Hlabel',reject[:,1]),('either',reject.any(1)),('both',reject.all(1))]:
                        power=float(indicator.mean())
                        rows.append(dict(block=block,sessions=horizon,endpoint=endpoint,true_increment=effect,
                            contrast=name,rejection_rate=power,mc_se=float(np.sqrt(power*(1-power)/len(indicator)))))
            print(f'Finished power block{block}/sessions{horizon}',flush=True)
    pd.DataFrame(rows).to_csv(out/'POWER.csv',index=False)
    pd.DataFrame(nulls).to_csv(out/'REFERENCE_NULL_INTERVALS.csv',index=False)
    actual=[]
    for j,right in enumerate(['B','E_Hlabel']):
        for lag in [5,20]:actual.append(dict(right=right,lag=lag,**nw_mean(raw[:,0]-raw[:,j+1],lag)))
    pd.DataFrame(actual).to_csv(out/'NEWEY_WEST.csv',index=False)
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [source,Path(__file__),Path('docs/rishi_mechanism_20261027/POWER_PROTOCOL.md')]}
    (out/'RESULTS.json').write_text(json.dumps(dict(rows=len(rows),null_intervals=len(nulls),seed=27102031,
        reference_reps=10000,simulated_reps=10000,source_days=len(raw),hashes=hashes,
        limits='Conditional empirical planning; no prospective activation; same market history, not independent years/seeds'),indent=2)+'\n')


if __name__=='__main__':main()
