"""Independently recompute saved pretest characteristics from daily inputs."""
import numpy as np
import pandas as pd
from pit_research_engine import load_data,fold


def main():
    data=load_data();saved=pd.read_csv('docs/rishi_mechanism_20261027/execution_batch57_validation_style/STOCKS.csv')
    q=pd.read_csv('data/nasdaq_monthly_pit/assembled/etf_adj_close.csv',index_col=0,parse_dates=True).QQQ.pct_change(fill_method=None).to_numpy()
    indices={v:i for i,v in enumerate(data['panel'].tickers)};maximum=0.;checked=0
    for row in saved.itertuples():
        _,_,test=fold(data,row.year);i=test[0];day=int(data['origin'][i]);j=indices[row.security_id];assert data['mask'][i,j]
        if row.feature=='beta252':
            start=max(0,day-251);r=data['returns'][start:day+1,j];m=data['tradable'][start:day+1,j]&np.isfinite(r)&np.isfinite(q[start:day+1]);a=r[m];b=q[start:day+1][m];assert len(a)>=126
            value=np.dot(a-a.mean(),b-b.mean())/np.dot(b-b.mean(),b-b.mean())
        elif row.feature=='momentum126':
            assert day>=126 and data['tradable'][day,j] and data['tradable'][day-126,j];value=data['marks'][day,j]/data['marks'][day-126,j]-1
        elif row.feature=='volatility20':
            start=max(0,day-19);r=data['returns'][start:day+1,j];m=data['tradable'][start:day+1,j]&np.isfinite(r);a=r[m];assert len(a)>=15;value=np.sqrt(np.sum((a-a.mean())**2)/(len(a)-1))*np.sqrt(252)
        elif row.feature=='log10adv20':
            value=np.log10(data['adv'][day,j])
        else:raise ValueError(row.feature)
        maximum=max(maximum,abs(value-row.value));checked+=1
    assert checked==4560 and maximum<1e-10
    print(f'{checked} saved characteristic values independently reconstructed; max error {maximum:.3g}')


if __name__=='__main__':main()
