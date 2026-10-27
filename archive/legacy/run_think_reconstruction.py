"""NYSE daily ranking reconstruction; exact reproduction is explicitly gated.

Run with --allow-reconstruction to accept the documented unresolved settings.
Default 25 seeds / 50 epochs are experiment settings, not recovered THINK values.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from think_model import ThinkReconstruction

UNRESOLVED = [
    'Author THINK implementation and checkpoints unavailable; public repository empty.',
    'Exact processed incidence matrix and Wikidata snapshot unavailable.',
    'THINK lookback, kernels, widths, initialization, optimizer and stopping policy unavailable.',
    'Printed Eq.7 is ambiguous; standard Mobius scalar distance modulation is assumed.',
    'Eq.10 directional normalization follows cited HNN++ code; differs from literal printed inner product.',
    'No attention softmax is added: raw signed coefficients follow displayed Eq.15.',
    'NYSE split indices, lookback and ranking loss borrowed from cited STHAN-SR, not verified THINK settings.',
    'Correct NDCG uses positive realized return, k=5 and tie averaging; THINK relevance convention unknown.',
    'Twenty-five runs cannot verify published significance without matching per-run comparison protocols.',
]


def load_nyse(root, relation_path, lookback=4):
    tickers=(root/'NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv').read_text().splitlines()
    raw=np.stack([np.loadtxt(root/'NYSE'/f'NYSE_{t}_1.csv',delimiter=',') for t in tickers])
    features=raw[:,:,1:].transpose(1,0,2).astype('float32')
    missing=np.isclose(features,-1234)
    prices=features[:,:,-1].copy()
    valid=~missing[:,:,-1]
    features=np.where(missing,1.1,features)
    targets=np.arange(lookback,len(features))
    x=np.stack([features[t-lookback:t] for t in targets])
    mask=np.stack([valid[t-lookback:t+1].all(0) for t in targets])
    base=prices[targets-1]
    y=np.where(mask,prices[targets]/np.where(mask,base,1)-1,0).astype('float32')
    base=np.where(mask,base,1).astype('float32')
    ids={t:i for i,t in enumerate(tickers)}
    groups=[{ids[t] for t in group} for group in json.loads(relation_path.read_text())]
    # Partition on target row, so training never reaches into validation.
    split={'train':np.flatnonzero(targets<756),
           'validation':np.flatnonzero((targets>=756)&(targets<1008)),
           'test':np.flatnonzero(targets>=1008)}
    return {'x':torch.from_numpy(x),'y':torch.from_numpy(y),'base':torch.from_numpy(base),
            'mask':torch.from_numpy(mask.astype('float32')), 'targets':targets,
            'splits':split,'groups':groups,'tickers':tickers,'prices':prices,'valid':valid}


def rank_loss(prediction, truth, mask, alpha=1., chunk=128):
    # Exact all-pairs source-style hinge loss, chunked to control memory.
    n=prediction.shape[1]
    mse=((prediction-truth).square()*mask).mean()
    rank=prediction.new_zeros(())
    for start in range(0,n,chunk):
        dp=prediction[:,start:start+chunk,None]-prediction[:,None,:]
        dy=truth[:,None,:]-truth[:,start:start+chunk,None]
        pair_mask=mask[:,start:start+chunk,None]*mask[:,None,:]
        rank=rank+(torch.relu(dp*dy)*pair_mask).sum()
    return mse+alpha*rank/(prediction.shape[0]*n*n)


def dcg(gain,score,k=5):
    value=0.
    # Average discounts over ties; ranking never depends on ticker identity.
    order=np.argsort(-score,kind='stable'); g,s=gain[order],score[order]
    i=0
    while i<min(k,len(s)):
        j=i+1
        while j<len(s) and s[j]==s[i]: j+=1
        value+=g[i:j].mean()*sum(1/np.log2(rank+2) for rank in range(i,min(j,k)))
        i=j
    return float(value)


def metrics(pred,y,mask,k=5):
    rets,ndcg=[],[]
    for score,actual,valid in zip(pred,y,mask):
        valid=valid.astype(bool); score,actual=score[valid],actual[valid]
        kk=min(k,len(score))
        if kk==0: continue
        boundary=np.sort(score)[-kk]
        above,tied=score>boundary,score==boundary
        w=(above.astype(float)+tied*((kk-above.sum())/tied.sum()))/kk
        rets.append(float(w@actual))
        gains=np.maximum(actual,0); ideal=dcg(gains,gains,kk)
        if ideal>0: ndcg.append(dcg(gains,score,kk)/ideal)
    a=np.asarray(rets)
    return {'mse':float((((pred-y)*mask)**2).sum()/mask.sum()),
            'gross_sharpe_annualized_rf0':float(a.mean()/a.std(ddof=0)*np.sqrt(252)) if a.std()>0 else None,
            'ndcg_at_5_positive_return':float(np.mean(ndcg)), 'ndcg_days':len(ndcg),
            'portfolio_days':len(a),'gross_compounded_return':float(np.prod(1+a)-1)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-reconstruction',action='store_true')
    parser.add_argument('--out',type=Path,default=Path('runs/think_nyse_reconstruction'))
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(25)))
    parser.add_argument('--variants',nargs='+',default=['think','euclidean_temporal','no_distance','euclidean'])
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--hidden',type=int,default=16)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--pilot',action='store_true',help='Mark outputs as pipeline checks, not benchmark conclusions.')
    args=parser.parse_args()
    if not args.allow_reconstruction:
        parser.exit(2,'One-to-one reproduction unavailable. Unresolved prerequisites:\n- '+'\n- '.join(UNRESOLVED)+'\nUse --allow-reconstruction only for a declared reconstruction.\n')
    if min(args.epochs,args.patience,args.hidden,args.batch_size)<1 or args.lr<=0:
        parser.error('Training parameters must be positive.')
    if args.out.exists() and any(args.out.iterdir()): parser.error('Choose a fresh output directory.')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source=Path('data/think_audit'); relation=Path('runs/think_audit/NYSE_reconstructed_hyperedges.json')
    data=load_nyse(source,relation)
    meta={'status':'pilot only' if args.pilot else 'reconstruction, not exact reproduction',
          'unresolved':UNRESOLVED,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          'nodes':len(data['tickers']),'timesteps':len(data['prices']),'hyperedges':len(data['groups']),
          'memberships':sum(map(len,data['groups'])),'splits':{k:{'count':len(ids),'first_target_row':int(data['targets'][ids[0]]),'last_target_row':int(data['targets'][ids[-1]])} for k,ids in data['splits'].items()},
          'relation_sha256':hashlib.sha256(relation.read_bytes()).hexdigest(),
          'source_manifest_sha256':hashlib.sha256((source/'stock_sources.json').read_bytes()).hexdigest(),
          'torch':str(torch.__version__),'kernels':[2,2],'lookback':4,
          'selection':'Lowest validation return MSE; test evaluated once after selection.',
          'target':'Next-session normalized close; optimized as implied return with MSE plus all-pairs ranking hinge.',
          'portfolio':'Top-5 equally weighted, ties averaged. Gross close-to-close paper-style diagnostic. No costs/risk-free return. Assumes filling at signal close; not an executable backtest.',
          'code_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['think_model.py','run_think_reconstruction.py']}}
    (args.out/'experiment.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    train,val,test=[data['splits'][key] for key in ['train','validation','test']]

    def predict(model,indices):
        output=[]
        model.eval()
        with torch.no_grad():
            for ids in np.array_split(indices,max(1,int(np.ceil(len(indices)/args.batch_size)))):
                output.append(((model(data['x'][ids])-data['base'][ids])/data['base'][ids]).numpy())
        return np.concatenate(output)

    rows=[]
    for seed in args.seeds:
        for variant in args.variants:
            torch.manual_seed(seed)
            model=ThinkReconstruction(5,len(data['tickers']),data['groups'],args.hidden,variant=variant)
            optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=5e-4)
            rng=np.random.default_rng(seed)
            best,stale,state,history=float('inf'),0,None,[]
            started=time.perf_counter()
            for epoch in range(1,args.epochs+1):
                model.train(); total=0.
                for ids in np.array_split(rng.permutation(train),int(np.ceil(len(train)/args.batch_size))):
                    optimizer.zero_grad(set_to_none=True)
                    predicted=(model(data['x'][ids])-data['base'][ids])/data['base'][ids]
                    loss=rank_loss(predicted,data['y'][ids],data['mask'][ids])
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step(); total+=loss.item()*len(ids)
                pv=predict(model,val); yv=data['y'][val].numpy(); mv=data['mask'][val].numpy()
                score=float((((pv-yv)*mv)**2).sum()/mv.sum())
                history.append({'epoch':epoch,'train_loss':total/len(train),'validation_mse':score})
                if score<best:
                    best,best_epoch,stale=score,epoch,0; state=copy.deepcopy(model.state_dict())
                else: stale+=1
                print(f'{variant} seed={seed} epoch={epoch} validation_mse={score:.7g} elapsed={time.perf_counter()-started:.1f}s',flush=True)
                if stale>=args.patience: break
            model.load_state_dict(state)
            prediction=predict(model,test)
            result=metrics(prediction,data['y'][test].numpy(),data['mask'][test].numpy())
            result.update({'model':variant,'seed':seed,'best_epoch':best_epoch,'seconds':time.perf_counter()-started})
            rows.append(result)
            key=f'{variant}_{seed}'
            np.savez_compressed(args.out/f'{key}_predictions.npz',predicted_return=prediction,
                                actual_return=data['y'][test].numpy(),mask=data['mask'][test].numpy(),
                                target_rows=data['targets'][test],tickers=np.asarray(data['tickers']))
            torch.save({'state_dict':state,'variant':variant,'hidden':args.hidden,'groups':[sorted(g) for g in data['groups']],
                        'tickers':data['tickers'],'features':5,'kernels':(2,2)},args.out/f'{key}.pt')
            pd.DataFrame(history).to_csv(args.out/f'{key}_history.csv',index=False)
            pd.DataFrame(rows).to_csv(args.out/'metrics.csv',index=False)
            print(result,flush=True)
    table=pd.DataFrame(rows)
    table.groupby('model')[['mse','gross_sharpe_annualized_rf0','ndcg_at_5_positive_return']].agg(['mean','std']).to_csv(args.out/'summary.csv')


if __name__=='__main__': main()
