"""Frozen first mechanism batch: literal saved-path analysis, no financial fitting."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path.cwd()
DOC = ROOT/'docs/rishi_mechanism_20261027'
OUT = DOC/'execution_batch1'
NS = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_leader_paths(path):
    with ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        book = ET.fromstring(z.read('xl/workbook.xml'))
        target = next(s for s in book.find('s:sheets', NS) if s.attrib['name']=='Leader paths')
        rid = target.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        rels = {r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        name = rels[rid]
        name = name.lstrip('/') if name.startswith('/') else 'xl/'+name
        sheet = ET.fromstring(z.read(name))
        cells = {c.attrib['r']:c for c in sheet.findall('s:sheetData/s:row/s:c', NS)}
        def value(ref, literal=False):
            c = cells.get(ref)
            if c is None:
                raise ValueError('Missing cell '+ref)
            if literal and c.find('s:f', NS) is not None:
                raise ValueError('Source return must be literal: '+ref)
            v = c.find('s:v', NS)
            if c.attrib.get('t')=='inlineStr':
                return ''.join(c.find('s:is', NS).itertext())
            text = None if v is None else v.text
            return shared[int(text)] if c.attrib.get('t')=='s' else text
        headers = {chr(c):value(chr(c)+'5') for c in range(ord('C'), ord('N')+1) if chr(c)+'5' in cells}
        if not headers or len(set(headers.values()))!=len(headers):
            raise ValueError('Missing or duplicate source IDs')
        rows = sorted({int(ref[1:]) for ref in cells if ref.startswith('A') and ref[1:].isdigit() and int(ref[1:])>=6})
        initial = rows[0]
        if pd.Timestamp(value('A'+str(initial))).date()!=pd.Timestamp('2021-12-31').date():
            raise ValueError('Unexpected initial date')
        if any(float(value(c+str(initial), True))!=0. for c in headers):
            raise ValueError('Initial row must be zero')
        output = []
        for row in rows[1:]:
            date = pd.Timestamp(value('A'+str(row))).tz_localize(None)
            for col, key in headers.items():
                output.append(dict(date=date, id=key, net=float(value(col+str(row), True))))
        return pd.DataFrame(output)


def validate_path(dates, returns, expected):
    dates, expected = pd.DatetimeIndex(dates), pd.DatetimeIndex(expected)
    r = np.asarray(returns, float)
    if len(dates)!=len(r) or not dates.equals(expected) or dates.has_duplicates:
        raise ValueError('Missing, duplicate, unordered or different calendar')
    if not np.isfinite(r).all() or (r<=-1).any():
        raise ValueError('Nonfinite or bankrupt account')
    return r


def account_metrics(r, annualizer=252):
    r = np.asarray(r, float)
    if len(r)<2 or not np.isfinite(r).all() or (r<=-1).any():
        raise ValueError('Invalid account')
    w = np.r_[1., np.cumprod(1+r)]
    vol = r.std(ddof=1)*np.sqrt(annualizer)
    rf = np.expm1(np.log1p(.03)/annualizer)
    return dict(sharpe=(r.mean()-rf)*annualizer/vol if vol>1e-15 else None,
                annualized_return=np.expm1(np.log1p(r).sum()*annualizer/len(r)),
                annualized_volatility=vol, maximum_drawdown=np.min(w/np.maximum.accumulate(w)-1),
                annual_arithmetic_mean=r.mean()*annualizer, terminal_wealth=w[-1])


def stationary_indices(n, draws, block, seed):
    rng = np.random.default_rng(seed)
    ids = np.empty((draws,n), dtype=np.int32)
    ids[:,0] = rng.integers(n, size=draws)
    for j in range(1,n):
        restart = rng.random(draws)<1/block
        ids[:,j] = np.where(restart, rng.integers(n,size=draws), (ids[:,j-1]+1)%n)
    return ids


def paired_contrast(left, right, ids, annualizer, growth_periods=None):
    left, right = np.asarray(left,float), np.asarray(right,float)
    assert left.shape==right.shape and ids.shape[1]==len(left)
    growth = annualizer if growth_periods is None else growth_periods
    a, b = left[ids], right[ids]
    effect = (left-right).mean()*annualizer
    boot = (a-b).mean(1)*annualizer
    lo, hi = np.quantile(boot, [.025,.975])
    null = boot-effect
    p = (1+np.sum(null>=effect))/(len(boot)+1)
    rf = np.expm1(np.log1p(.03)/annualizer)
    def sr(x):
        return (x.mean(-1)-rf)/x.std(-1,ddof=1)*np.sqrt(annualizer)
    def cagr(x):
        return np.expm1(np.log1p(x).mean(-1)*growth)
    sr_boot, cagr_boot = sr(a)-sr(b), cagr(a)-cagr(b)
    return dict(mean_difference=float(effect),mean_low=float(lo),mean_high=float(hi),mean_p_one_sided=float(p),
                sr_difference=float(sr(left)-sr(right)),sr_low=float(np.quantile(sr_boot,.025)),sr_high=float(np.quantile(sr_boot,.975)),
                cagr_difference=float(cagr(left)-cagr(right)),cagr_low=float(np.quantile(cagr_boot,.025)),cagr_high=float(np.quantile(cagr_boot,.975)))


def initial_capital_pool(left, right):
    a, b = np.asarray(left,float), np.asarray(right,float)
    if a.shape!=b.shape:
        raise ValueError('Unaligned components')
    w = .5*np.r_[1.,np.cumprod(1+a)]+.5*np.r_[1.,np.cumprod(1+b)]
    return w[1:]/w[:-1]-1


def holm(p):
    p = np.asarray(p,float)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result


def inventory():
    relative = ['data/nasdaq_monthly_pit/assembled/manifest.json','runs/ndx_reference_networks_20261024',
                'runs/ndx_extended_hours_20261024','runs/ndx_drawdown_20261026/batch1']
    roots = [ROOT]+[ROOT.parent/n for n in ['Hypershift-rishi','Hypershift-rishi-latest','Hypershift-unify']]
    result = [dict(root=str(root), artifacts={p:(root/p).exists() for p in relative}) for root in roots]
    (OUT/'ARTIFACT_INVENTORY.json').write_text(json.dumps(dict(checkouts=result,
        raw_replay_available=any(all((root/p).exists() for p in relative) for root in roots),
        downloads_search='Bounded NDX/Rishi/prediction/reference/drawdown filename search; no relevant raw payload found'),indent=2))


def daily_batch():
    book = ROOT/'docs/NASDAQ100_strategy_comparison.xlsx'
    frame = read_leader_paths(book)
    old = pd.read_csv(ROOT/'docs/ndx_drawdown_20261026/comparison_curves.csv',parse_dates=['date'])
    expected = pd.DatetimeIndex(old[old.group=='ten_seed'].query("recipe=='equal5/full'").date)
    metrics = pd.read_csv(ROOT/'docs/ndx_drawdown_20261026/metrics.csv')
    paths, summary, checks, annual = {}, [], [], []
    for key, sub in frame.groupby('id',sort=False):
        r = validate_path(sub.date, sub.net, expected)
        m = account_metrics(r)
        paths[key] = r
        if key not in ['SPY','QQQ']:
            recipe, group = key.rsplit('/',1)
            source = metrics[(metrics.recipe==recipe)&(metrics.group==group)&(metrics.cost_multiplier==1)].iloc[0]
            error = max(abs(m[k]-float(source[k])) for k in ['sharpe','annualized_return','annualized_volatility','maximum_drawdown'])
            assert error<1e-10, (key,error)
            overlap = old[(old.recipe==recipe)&(old.group==group)]
            if len(overlap):
                assert np.max(abs(overlap.net.to_numpy()-r))<1e-14
            checks.append(dict(id=key,max_metric_error=error,csv_overlap=bool(len(overlap))))
        summary.append(dict(id=key,**m))
        for year in sorted(set(expected.year)):
            rr = r[expected.year==year]
            annual.append(dict(id=key,year=year,calendar_return=float(np.prod(1+rr)-1),log_growth=float(np.log1p(rr).sum())))
    frame.to_csv(OUT/'WORKBOOK_PATHS.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'WORKBOOK_METRICS.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'WORKBOOK_ANNUAL.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'WORKBOOK_VALIDATION.csv',index=False)
    pairs = [('volatility','equal5/basketvol20_target20_weekly','equal5/full'),
             ('issuer','issuer5/basketvol20_target20_weekly','equal5/basketvol20_target20_weekly'),
             ('trend','equal5/ma200off50_basketvol20_target20_weekly','equal5/basketvol20_target20_weekly')]
    rows = []
    for block in [21,5,63]:
        ids = stationary_indices(len(expected),10000,block,27102026)
        for group in ['original_five','ten_seed']:
            for name,left,right in pairs:
                result = paired_contrast(paths[left+'/'+group],paths[right+'/'+group],ids,252)
                rows.append(dict(contrast=name,group=group,left=left,right=right,block_sessions=block,**result))
    table = pd.DataFrame(rows)
    for block in [21,5,63]:
        select = table.block_sessions==block
        table.loc[select,'mean_p_holm'] = holm(table.loc[select,'mean_p_one_sided'])
    table.to_csv(OUT/'DAILY_CONTRASTS.csv',index=False)
    return dict(workbook_series=len(paths),sessions=len(expected),max_metric_error=max(c['max_metric_error'] for c in checks),
                csv_overlapping_series=sum(c['csv_overlap'] for c in checks))


def monthly_batch():
    folder = ROOT/'latest/results/reference_networks'
    monthly = pd.read_csv(folder/'monthly.csv')
    metrics = pd.read_csv(folder/'metrics.csv',dtype={'seed':str})
    groups = ['original_five','new_five','ten_seed']
    kinds = ['baseline','dynamic_restricted','dynamic_restricted_blend']
    paths, checks, pooled = {}, [], []
    calendar = pd.period_range('2022-01','2025-12',freq='M').astype(str).tolist()
    for group in groups:
        for k in [5,10]:
            for kind in kinds:
                key = f'{kind}/{group}/ensemble/{k}/1'
                sub = monthly[monthly.id==key].sort_values('month')
                if list(sub.month)!=calendar or sub.month.duplicated().any():
                    raise ValueError('Monthly calendar mismatch '+key)
                r = sub.net_return.to_numpy()
                assert np.isfinite(r).all() and (r>-1).all()
                paths[kind,group,k] = r
                old = metrics[metrics.id==key].iloc[0]
                cagr = np.expm1(np.log1p(r).sum()*252/1003)
                error = abs(cagr-float(old.annualized_return))
                assert error<1e-10,(key,error)
                checks.append(dict(id=key,cagr=cagr,cagr_error=error))
            p = initial_capital_pool(paths['baseline',group,k],paths['dynamic_restricted',group,k])
            e = paths['dynamic_restricted_blend',group,k]
            pool_cagr = float(np.expm1(np.log1p(p).sum()*252/1003))
            blend_cagr = float(np.expm1(np.log1p(e).sum()*252/1003))
            pooled.append(dict(group=group,k=k,pool_cagr=pool_cagr,blend_cagr=blend_cagr,cagr_difference=blend_cagr-pool_cagr,
                               pool_terminal_wealth=float(np.prod(1+p)),blend_terminal_wealth=float(np.prod(1+e)),
                               scope='Algebraic initial-capital combination of full-size return paths; half-AUM costs/caps not replayed'))
    rows = []
    for block in [3,1,6]:
        ids = stationary_indices(48,10000,block,27102027)
        for k in [5,10]:
            for right in ['baseline','dynamic_restricted']:
                result = paired_contrast(paths['dynamic_restricted_blend','ten_seed',k],paths[right,'ten_seed',k],ids,12,48*252/1003)
                result = {key:value for key,value in result.items() if not key.startswith('sr_')}
                rows.append(dict(k=k,right=right,block_months=block,**result))
    table = pd.DataFrame(rows)
    for block in [3,1,6]:
        select = table.block_months==block
        table.loc[select,'mean_p_holm'] = holm(table.loc[select,'mean_p_one_sided'])
    table.to_csv(OUT/'MONTHLY_MODEL_CONTRASTS.csv',index=False)
    pd.DataFrame(pooled).to_csv(OUT/'CAPITAL_POOL_DIAGNOSTIC.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'MONTHLY_VALIDATION.csv',index=False)
    return dict(monthly_accounts=len(checks),months=48,max_cagr_error=max(c['cagr_error'] for c in checks))


def reserve_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Preserve existing batch; choose a fresh destination: '+str(path))
    path.mkdir(parents=True,exist_ok=True)


def main():
    reserve_output(OUT)
    inventory()
    result = dict(status='completed_available_aggregate_tests_raw_mechanism_deferred',daily=daily_batch(),monthly=monthly_batch())
    files = ['docs/NASDAQ100_strategy_comparison.xlsx','docs/ndx_drawdown_20261026/comparison_curves.csv',
             'docs/ndx_drawdown_20261026/metrics.csv','latest/results/reference_networks/monthly.csv',
             'latest/results/reference_networks/metrics.csv','docs/rishi_mechanism_20261027/EXECUTION_PROTOCOL.md']
    result['input_sha256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    result['scope'] = 'Selected historical account-path diagnostics; no trained-signal or half-capital execution replay'
    (OUT/'BATCH_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
