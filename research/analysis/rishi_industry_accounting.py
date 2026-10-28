"""Partition previously reconciled stock P&L by causally available SIC snapshots."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pit_research_engine import load_data


def active_snapshot(cutoffs,day):return int(np.searchsorted(cutoffs,day,side='left')-1)


def partition(values,categories):
    values=np.asarray(values,float);categories=np.asarray(categories)
    if values.shape!=categories.shape or not np.isfinite(values).all() or not np.isin(categories,[0,1,2]).all():raise ValueError('Aligned finite accounting/category arrays required')
    return np.stack([np.where(categories==i,values,0).sum(1) for i in range(3)],axis=1)


def main():
    study=Path('docs/rishi_mechanism_20261027');out=study/'execution_batch23_industry_accounting';large=Path('results/RISHI_INDUSTRY_ACCOUNTING_20261028')
    for p in (out,large):
        if p.exists():raise FileExistsError(p)
        p.mkdir(parents=True)
    trace=Path('results/RISHI_MECHANISM_TRACE_20261027');source=Path('results/RISHI_MECHANISM_REPLAY_20261027_EXPORTS')
    detail=study/'execution_batch21_saved_industry/DETAIL.csv';summary=study/'execution_batch21_saved_industry/SUMMARY.csv'
    df=pd.read_csv(detail,float_precision='round_trip');df=df[df.model=='E'];cut=pd.read_csv(summary);cut=cut[cut.model=='E'].sort_values('year')
    cutoffs=np.array(cut.signal_date,dtype='datetime64[D]');axis=json.loads((trace/'AXES.json').read_text());dates=np.array(axis['dates'],dtype='datetime64[D]');tickers=axis['security_ids']
    data=load_data()
    if tickers!=list(data['panel'].tickers):raise ValueError('Stock-axis mismatch')
    index=data['panel'].dates.get_indexer(pd.to_datetime(dates))
    if (index<0).any():raise ValueError('Price calendar mismatch')
    marks=np.nan_to_num(data['marks'][index],nan=0.)
    categories=np.full((len(dates),len(tickers)),2,dtype='uint8')
    for i,day in enumerate(dates):
        idx=active_snapshot(cutoffs,day)
        if idx<0:continue
        labels=df[df.year==int(cut.iloc[idx].year)].set_index('security_id').industry.reindex(tickers).to_numpy(float)
        categories[i]=np.where(np.isnan(labels),2,np.where(labels==6,0,1))
    np.savez_compressed(large/'CATEGORY_AXES.npz',categories=categories,dates=dates.astype('U'),security_ids=np.asarray(tickers,dtype='U'))
    curves=pd.read_csv(source/'ACCOUNT_CURVES.csv');groups=['ff12_6','other_classified','unclassified'];annual=[];calendar=[];reconciliations=[];paths=[detail,summary,trace/'AXES.json',source/'ACCOUNT_CURVES.csv',Path(__file__),study/'INDUSTRY_ACCOUNTING_PROTOCOL.md']
    for model in ('B','D','E','Hlabel','E_Hlabel'):
        for k in (5,10):
            key=f'{model}_ten_seed_k{k}';path=trace/(key+'.npz');ledger_path=source/f'ledger_{model}_ten_seed_{k}.csv';paths.extend([path,ledger_path])
            z=np.load(path,allow_pickle=False);ledger=pd.read_csv(ledger_path);curve=curves[curves.id==f'{model}/ten_seed/{k}']
            if ledger.date.tolist()!=axis['dates'] or curve.date.tolist()!=axis['dates']:raise ValueError('Account calendar mismatch')
            nav=ledger.nav.to_numpy();previous=np.r_[1_000_000.,nav[:-1]]
            net=partition(z['net_contribution'],categories);gross=partition(z['gross_dollars']/previous[:,None],categories)
            fee=partition(z['fees']/previous[:,None],categories);dollars=partition(z['fees'],categories)
            exposure=partition(z['units']*marks/nav[:,None],categories)
            errors=dict(net=float(np.max(abs(net.sum(1)-curve.net.to_numpy()))),decomposition=float(np.max(abs((gross-fee-net)))),
                fee_dollars=float(np.max(abs(dollars.sum(1)-ledger.fee.to_numpy()))),exposure=float(np.max(abs(exposure.sum(1)-ledger.holdings.to_numpy()/nav))))
            if errors['net']>1e-10 or errors['decomposition']>1e-10 or errors['fee_dollars']>1e-5 or errors['exposure']>1e-10:raise ValueError('Accounting partition does not reconcile '+str(errors))
            reconciliations.append(dict(account=key,**errors))
            np.savez_compressed(large/(key+'.npz'),net=net,gross=gross,fee=fee,exposure=exposure)
            for j,label in enumerate(groups):
                annual.append(dict(model=model,k=k,group=label,annual_net=float(net[:,j].mean()*252),annual_gross=float(gross[:,j].mean()*252),
                    annual_fee=float(fee[:,j].mean()*252),mean_exposure=float(exposure[:,j].mean()),fees_dollars=float(dollars[:,j].sum())))
                for year in (2022,2023,2024,2025):
                    use=pd.DatetimeIndex(dates).year==year
                    calendar.append(dict(model=model,k=k,year=year,group=label,sessions=int(use.sum()),annual_net=float(net[use,j].mean()*252),
                        mean_exposure=float(exposure[use,j].mean()),fees_dollars=float(dollars[use,j].sum())))
            z.close()
    pd.DataFrame(annual).to_csv(out/'FULL_SAMPLE.csv',index=False);pd.DataFrame(calendar).to_csv(out/'CALENDAR.csv',index=False)
    pd.DataFrame(reconciliations).to_csv(out/'RECONCILIATION.csv',index=False)
    (out/'RESULTS.json').write_text(json.dumps(dict(accounts=10,days=len(dates),groups=groups,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limits='Reconstructed economic account partition, not a counterfactual or certified broker records'),indent=2)+'\n')
    print(pd.DataFrame(annual).query("model == 'E'").to_string(index=False))


if __name__=='__main__':main()
