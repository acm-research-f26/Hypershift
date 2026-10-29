"""Synthetic preflight objectives; not a live portfolio simulator."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
from scipy.stats import spearmanr
import pandas as pd
import torch.nn.functional as F

def pairwise(score,label,mask):
 delta=label[:,:,None]-label[:,None,:];valid=mask[:,:,None]&mask[:,None,:]&(delta>0)
 diff=score[:,:,None]-score[:,None,:];loss=F.softplus(-diff)
 per=(loss*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
 return per.mean()

def portfolio(score,label,mask,cost=.02):
 if not mask.any(1).all():raise ValueError('empty universe')
 w=torch.softmax(score.masked_fill(~mask,-torch.inf),dim=1);returns=(w*label.masked_fill(~mask,0)).sum(1);previous=torch.cat((torch.zeros_like(w[:1]),w[:-1]),dim=0);turnover=.5*(torch.sqrt((w-previous).square()+1e-8)-1e-4).sum(1).mean()
 return -returns.mean()+.5*returns.var(unbiased=False)+cost*turnover

def main():
 out=Path('docs/historical_sharpe_20261029/objective_preflight');dest=Path('results/HISTORICAL_OBJECTIVE_PREFLIGHT_20261029')
 for p in (out,dest):
  if p.exists():raise FileExistsError(p)
 out.mkdir(parents=True);dest.mkdir(parents=True);rng=np.random.default_rng(29102029);x=rng.normal(size=(120,8,3)).astype('float32');y=(.01*(1.5*x[:,:,0]-.7*x[:,:,1])+.002*rng.normal(size=(120,8))).astype('float32');X=torch.tensor(x);Y=torch.tensor(y);mask=torch.ones_like(Y,dtype=torch.bool);rows=[];history=[];arrays={'x':x,'y':y}
 for mode in ('mse','pairwise','portfolio'):
  w=torch.nn.Parameter(torch.zeros(3));opt=torch.optim.Adam([w],lr=.03)
  for epoch in range(151):
   score=X[:80]@w
   loss=((score-Y[:80])**2).mean() if mode=='mse' else pairwise(score,Y[:80],mask[:80]) if mode=='pairwise' else portfolio(score,Y[:80],mask[:80])
   if epoch in (0,150):history.append(dict(mode=mode,epoch=epoch,loss=float(loss.detach())))
   if epoch==150:break
   opt.zero_grad();loss.backward();assert torch.isfinite(w.grad).all();opt.step()
  test=(X[80:]@w).detach();rho=float(np.mean([spearmanr(test[i].numpy(),y[80+i]).statistic for i in range(40)]));value=float(portfolio(test,Y[80:],mask[80:]).detach());initial=float(portfolio(torch.zeros_like(test),Y[80:],mask[80:]).detach());weights_test=torch.softmax(test,dim=1);prev=torch.cat((torch.zeros_like(weights_test[:1]),weights_test[:-1]),dim=0);turn=float((.5*(torch.sqrt((weights_test-prev).square()+1e-8)-1e-4).sum(1).mean()).detach());rows.append(dict(mode=mode,mean_rank_correlation=rho,test_portfolio_loss=value,initial_portfolio_loss=initial,test_turnover=turn,initial_turnover=.5,weights=w.detach().tolist()));arrays[mode+'_scores']=test.numpy()
 pd.DataFrame(rows).to_csv(out/'RESULTS.csv',index=False);pd.DataFrame(history).to_csv(out/'HISTORY.csv',index=False);np.savez_compressed(dest/'SYNTHETIC.npz',**arrays);files=[Path(__file__),Path('docs/historical_sharpe_20261029/OBJECTIVE_PREFLIGHT_PROTOCOL.md')];(out/'HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
