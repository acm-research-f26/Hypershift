"""Post-freeze reporting only; never changes trading rules or selects recipes."""
import argparse
import base64
import csv
import gzip
import html
import hashlib
import inspect
import shutil
import json
from pathlib import Path
import numpy as np
from experiments.strategy_search.storage import atomic,read,sha,utc,clean
from experiments.decision_strategy.config import STUDY,SCOPES,CATALOG


def paired_uncertainty(a,b):
    """Paired five-session block resampling with cash days retained."""
    diff=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(20261027);means=[]
    if not len(diff):return dict(days=0,mean_difference=None,CI95=None)
    for _ in range(2000):
        starts=rng.integers(0,max(1,len(diff)-4),size=int(np.ceil(len(diff)/5)))
        ix=np.concatenate([np.arange(k,min(k+5,len(diff))) for k in starts])[:len(diff)]
        means.append(diff[ix].mean())
    return dict(days=len(diff),mean_daily_return_difference=float(diff.mean()),CI95=np.quantile(means,[.025,.975]).tolist(),
        method='paired moving5-session blocks,2000 repetitions; exposed historical data')


def read_rows(p):
    with gzip.open(p,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def csv_atomic(path,rows):
    from experiments.decision_strategy.runner import csv_gz
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    csv_gz(tmp,rows);tmp.replace(path)


def cash_tape(path,holdings,capital):
    return bool((path[:,:3]==capital).all() and (path[:,3:]==0).all() and not holdings.any())


def copy_export(source,dest):
    dest=Path(dest);tmp=dest.with_name(dest.name+'.tmp');shutil.copyfile(source,tmp)
    if sha(tmp)!=sha(source):raise ValueError('CSV reuse changed bytes')
    tmp.replace(dest)


def validate_files(output):
    output=Path(output);protocol=read(output/'protocol.json');errors=[];checked=0
    source=Path(__file__).parent/'decision_strategy'
    for name,h in protocol['binding']['source'].items():
        if sha(source/name)!=h:errors.append('changed source:'+name)
    for name,h in protocol['binding']['dependencies'].items():
        if sha(Path(__file__).parents[1]/name)!=h:errors.append('changed dependency:'+name)
    if sha(Path(__file__).parents[1]/'uv.lock')!=protocol['binding']['lock_sha256']:errors.append('changed dependency lock')
    if sha(output/'causal_classifications.json')!=protocol['binding']['classifications_sha256']:errors.append('changed historical classifications')
    if sha(output/'fits/models.pkl')!=read(output/'fits/manifest.json')['sha256']:errors.append('changed fitted parameters')
    if sha(output/'protected_sessions.json')!=protocol['binding']['protected_sha256']:errors.append('changed protected manifest')
    quote_files=0
    from experiments.decision_strategy.data import guard_dates
    for scope in SCOPES:
        root=output/'quotes'/scope;manifest=root/'manifest.json'
        if not manifest.exists():continue
        for name,h in read(manifest)['files'].items():
            stamp=int(name.split('_')[0].split('.')[0]);date=str(np.datetime64(stamp,'ns').astype('datetime64[D]'))
            guard_dates({date},scope,protocol)
            if sha(root/name)!=h:errors.append('changed quote:'+name)
            quote_files+=1
    for p in (output/'trials').glob('*/receipt.json'):
        r=read(p)
        if r['protocol_sha256']!=sha(output/'protocol.json'):errors.append('wrong trial protocol:'+r['key'])
        for name,h in r['files'].items():
            checked+=1
            if sha(p.parent/name)!=h:errors.append('changed trial file:'+str(p.parent/name))
    result=dict(verified_UTC=utc(),checked_trial_files=checked,checked_quote_files=quote_files,errors=errors,passed=not errors,
        protected_future_outcomes_accessed=False)
    atomic(output/'local_integrity_verification.json',result)
    if errors:raise ValueError(errors)
    return result


def audit_market(scope,output):
    """Minimal independent adapter: guard dates before touching outcome arrays."""
    from experiments.decision_strategy.data import guard_dates
    output=Path(output);protocol=read(output/'protocol.json');root=STUDY/'cache'/scope
    if scope==SCOPES[2] and not (output/'selection.json').exists():raise ValueError('Later cohort is locked until selection')
    meta=read(root/'market.json');guard_dates(meta['dates'],scope,protocol)
    names=('open','close','sessions','completed')
    for name in names:
        if sha(root/(name+'.npy'))!=meta['files'][name+'.npy']:raise ValueError('Changed audit input')
    m={name:np.load(root/(name+'.npy'),mmap_mode='r') for name in names}
    m.update(starts=m['completed']-60_000_000_000,scope=scope,meta=meta)
    guard_dates(set(np.datetime_as_string(m['starts'].astype('datetime64[ns]').astype('datetime64[D]'))),scope,protocol)
    # ETFs are reused only for these verified historical dates. No signal
    # masks, future endpoints or later classifications enter this audit.
    from experiments.closing_followup.config import ETFS
    from experiments.decision_strategy.data import ETF_MAP
    classes=read(output/'causal_classifications.json');sectors=sorted(ETF_MAP)
    symbols=read(Path(protocol['binding']['market_root'])/'metadata.json')['symbols']+[ETF_MAP[s] for s in sectors]
    extended={k:np.full((len(m['starts']),11),np.nan) for k in ('open','close')}
    lookup={int(v):i for i,v in enumerate(m['starts'])}
    etfroot=STUDY/'closing_momentum_followup_2026-10-26'/'etf_cache'
    hashes=read(etfroot/(scope+'.bars_manifest.json'))
    for date in meta['dates']:
        p=etfroot/'bars'/(date+'.json')
        if sha(p)!=hashes[p.name]:raise ValueError('Changed ETF audit cache')
        d=read(p)
        for k,s in enumerate(symbols[250:]):
            for bar in d['bars'].get(s,[]):
                stamp=int(np.datetime64(bar['t'].rstrip('Z'),'ns').astype(np.int64))
                if stamp in lookup:
                    i=lookup[stamp];extended['open'][i,k]=bar['o'];extended['close'][i,k]=bar['c']
    for key in ('open','close'):m[key]=np.c_[m[key],extended[key]]
    m['symbols']=symbols;m['evaluate_ids']=meta['session_ids'][60:] if scope==SCOPES[0] else meta['session_ids']
    return m


def exact_attribution(receipt,m,shadow=False):
    """Reconcile per-leg cash flows, residual marks and allocated actual borrow."""
    root=Path(receipt['root']);prefix='shadow_' if shadow else ''
    decisions=read_rows(root/(prefix+'decisions.csv.gz'));fills=read_rows(root/(prefix+'fills.csv.gz'))
    roles=('primary_long','primary_short','hedge_long','hedge_short');definitions={}
    for d in decisions:
        if not d.get('basket'):continue
        syms=json.loads(d['symbols']);qty=json.loads(d['quantities'])
        definitions[d['basket']]={m['symbols'].index(s):((0 if qty[k]>0 else 1) if k==0 else (2 if qty[k]>0 else 3),qty[k]<0) for k,s in enumerate(syms)}
    with np.load(root/'minute_tapes.npz') as z:
        p=z['shadow_path' if shadow else 'equity'];holdings=z['shadow_holdings' if shadow else 'holdings']
    chosen=np.isin(m['sessions'],m['evaluate_ids']);ix=np.flatnonzero(chosen);a,b=ix[0],ix[-1]
    t=len(p);stats=np.zeros((4,5));by_stock={}
    for f in fills:
        i,j=int(f['minute']),int(f['stock']);basket=f['parent'].split(':')[0];role,negative=definitions[basket][j]
        q=int(f['shares']);px=float(f['price']);fees=sum(float(f[k]) for k in ('commission','exchange','clearing','regulatory'))
        if chosen[i]:stats[role,0]-=q*px+fees;stats[role,1]+=fees;stats[role,2]+=abs(q*px)
        by_stock.setdefault(j,[]).append((i,role,q,negative))
    allocated=np.zeros(t)
    for j,items in by_stock.items():
        raw=np.empty(2*t);raw[::2]=m['open'][:,j];raw[1::2]=m['close'][:,j]
        good=np.isfinite(raw)&(raw>0);idx=np.maximum.accumulate(np.where(good,np.arange(2*t)+1,0))
        close=np.r_[0.,raw][idx][1::2]
        short_changes=np.zeros((t,4));initial=np.zeros(4);ending=np.zeros(4)
        for i,role,q,negative in items:
            if i<a:initial[role]+=q
            if i<=b:ending[role]+=q
            if negative:short_changes[i,role]+=q
        stats[:,3]+=ending*close[b]-(initial*close[a-1] if a else 0.)
        short=np.maximum(-np.cumsum(short_changes,axis=0),0.);den=short.sum(1)
        dollars=np.maximum(-holdings[:,j]*close,0.)*receipt['settings']['borrow']/(365*1440)
        weights=np.divide(short,den[:,None],out=np.zeros_like(short),where=den[:,None]>0)
        allocation=weights*dollars[:,None];allocated+=allocation.sum(1);stats[:,4]+=allocation[chosen].sum(0)
    np.testing.assert_allclose(allocated,p[:,9],rtol=1e-10,atol=1e-8)
    pnl=stats[:,0]+stats[:,3]-stats[:,4]
    nav_pnl=p[b,1]-(receipt['settings']['capital'] if not a else p[a-1,1])
    np.testing.assert_allclose(pnl.sum(),nav_pnl,rtol=1e-9,atol=1e-5)
    return dict(key=receipt['key'],shadow=shadow,period='evaluation only;train60 excluded',
        legs={name:dict(trade_cash_pnl=float(stats[k,0]),fees=float(stats[k,1]),turnover=float(stats[k,2]),
            residual_mark_adjustment=float(stats[k,3]),borrow=float(stats[k,4]),net_pnl=float(pnl[k])) for k,name in enumerate(roles)},
        total_nav_pnl=float(nav_pnl),reconciled=True,maximum_borrow_allocation_error=float(abs(allocated-p[:,9]).max()))


def export_and_attribute(output,partial=False):
    """Complete CSV tapes plus independent exact leg attribution for all trials."""
    from experiments.decision_strategy.runner import csv_gz
    output=Path(output);proofs=[]
    rows=[read(p) for p in sorted((output/'trials').glob('*/receipt.json'))] if partial else read(output/'proxy_results.json')['rows']+read(output/'quote_results.json')['rows']
    algorithm_sha256=hashlib.sha256((''.join(inspect.getsource(f) for f in (read_rows,audit_market,exact_attribution))).encode()).hexdigest()
    for scope in SCOPES:
        if not any(r['scope']==scope for r in rows):continue
        m=audit_market(scope,output);cache={}
        for r in [x for x in rows if x['scope']==scope]:
            if shutil.disk_usage(output).free<10_200_000_000:raise RuntimeError('10GB disk reserve plus export headroom')
            root=Path(r['root']);dest=root/'independent_attribution.json'
            receipt_sha256=sha(root/'receipt.json')
            if dest.exists():
                old=read(dest)
                if old['algorithm_sha256']!=algorithm_sha256 or old['receipt_sha256']!=receipt_sha256:raise ValueError('Changed attribution algorithm or receipt')
            else:atomic(dest,dict(scaled=exact_attribution(r,m),shadow=exact_attribution(r,m,True),algorithm_sha256=algorithm_sha256,receipt_sha256=receipt_sha256))
            with np.load(root/'minute_tapes.npz') as z:
                starts,sessions=z['starts'],z['sessions']
                np.testing.assert_array_equal(starts,m['starts']);np.testing.assert_array_equal(sessions,m['sessions'])
                reused={}
                daily_file=root/'daily_evaluation.csv.gz'
                if not daily_file.exists():
                    csv_atomic(daily_file,[dict(session=int(v[0]),evaluated=int(v[0]) in m['evaluate_ids'],
                        fitting_only=scope==SCOPES[0] and int(v[0]) not in m['evaluate_ids'],
                        scaled_return=float(v[1]),scaled_nav=float(v[2]),shadow_return=float(w[1]),shadow_nav=float(w[2]),
                        causal_volatility_scale=float(scale)) for v,w,scale in zip(z['daily'],z['shadow_daily'],z['scale'])])
                for shadow in (False,True):
                    prefix='shadow_' if shadow else '';navfile=root/(prefix+'minute_nav.csv.gz')
                    p=z['shadow_path' if shadow else 'equity'];held=z['shadow_holdings' if shadow else 'holdings']
                    is_cash=cash_tape(p,held,r['settings']['capital'])
                    navkey=('nav',r['settings']['capital'],tuple(r['path_columns']))
                    if not navfile.exists():
                        if is_cash and navkey in cache:
                            copy_export(cache[navkey],navfile);reused[navfile.name]=str(cache[navkey])
                        else:csv_atomic(navfile,[dict(minute=i,arrival_ns=int(starts[i]),session=int(sessions[i]),**dict(zip(r['path_columns'],p[i]))) for i in range(len(p))])
                    if is_cash:cache.setdefault(navkey,navfile)
                    heldfile=root/(prefix+'minute_holdings_sparse.csv.gz')
                    if not heldfile.exists():
                        if not held.any() and 'holdings' in cache:
                            copy_export(cache['holdings'],heldfile);reused[heldfile.name]=str(cache['holdings'])
                        else:
                            ii,jj=np.nonzero(held)
                            csv_atomic(heldfile,[dict(minute=int(i),stock=int(j),symbol=r['symbols'][j],shares=int(held[i,j])) for i,j in zip(ii,jj)])
                    if not held.any():cache.setdefault('holdings',heldfile)
                if reused:atomic(root/'csv_reuse.json',dict(sources=reused,identical_cohort_timestamps_verified=True,
                    cash_NAV_capital_verified=True,zero_positions_verified=True,unchanged_trading_outcomes=True))
            exports=['minute_nav.csv.gz','shadow_minute_nav.csv.gz','minute_holdings_sparse.csv.gz','shadow_minute_holdings_sparse.csv.gz','daily_evaluation.csv.gz']
            proofs.append(dict(key=r['key'],sha256=sha(dest),reconciled=read(dest)['scaled']['reconciled'] and read(dest)['shadow']['reconciled'],
                csv_exports={name:dict(sha256=sha(root/name),size=(root/name).stat().st_size) for name in exports}))
            atomic(output/'export_progress.json',dict(updated_UTC=utc(),partial=partial,completed=len(proofs),target=len(rows),last_trial=r['key']))
        del m
    atomic(output/('attribution_partial.json' if partial else 'attribution_verification.json'),dict(completed_UTC=utc(),partial=partial,proofs=proofs,passed=all(p['reconciled'] for p in proofs),
        holdings_schema='minute_nav.csv.gz contains every minute; omitted holdings in minute_holdings_sparse.csv.gz are exactly zero. Complete dense integer holdings also in minute_tapes.npz.'))
    return proofs


def attempt_history(output):
    """Keep superseded/interrupted trials visible, separate from valid comparisons."""
    from experiments.decision_strategy.runner import csv_gz
    output=Path(output);rows=[];versions=[]
    for suffix in ('','_v2','_v3','_v4','_v5','_v6','_v7'):
        root=STUDY/('decision_strategy_execution_2026-10-27'+suffix)
        if not root.exists():continue
        protocol=root/'protocol.json';saved=0;interrupted=0
        for status in sorted((root/'trials').glob('*/status.json')):
            rp=status.parent/'receipt.json';s=read(status);r=read(rp) if rp.exists() else {}
            if r:saved+=1
            else:interrupted+=1
            rows.append(dict(version=suffix or 'v1',trial=status.parent.name,
                disposition='completed' if r else 'interrupted_preserved',recorded_status=s.get('status'),
                included_in_current_comparisons=root==output and bool(r),
                imported_from=r.get('imported_from'),return_value=r.get('metrics',{}).get('total_return'),
                roundtrips=r.get('metrics',{}).get('roundtrips'),valid=r.get('metrics',{}).get('accounting_valid'),
                evidence=str(rp if r else status),evidence_sha256=sha(rp if r else status)))
        versions.append(dict(path=str(root),protocol_sha256=sha(protocol) if protocol.exists() else None,
            completed_receipts=saved,interrupted_attempts=interrupted,
            use='current finite study' if root==output else 'preserved correctness/engineering history; never pooled for selection'))
        resume=root/'operational_queue_resume.json'
        if resume.exists():
            for interrupted in read(resume)['interrupted_attempts']:
                rows.append(dict(version=suffix,trial=interrupted['key'],disposition='interrupted_operational_resume',
                    recorded_status=interrupted['record']['status'],included_in_current_comparisons=False,
                    evidence=str(resume),evidence_sha256=sha(resume),note='Same frozen job subsequently restarted; original status and partial file hashes preserved.'))
        resume=root/'operational_parallel_resume.json'
        if resume.exists():
            for interrupted in read(resume).get('interrupted',[]):
                rows.append(dict(version=suffix,trial=Path(interrupted['root']).name,disposition='interrupted_operational_resume',
                    recorded_status=interrupted['status']['status'],included_in_current_comparisons=False,
                    evidence=str(resume),evidence_sha256=sha(resume),note='Serial-to-two-worker continuation; original attempt status preserved.'))
    result=dict(created_UTC=utc(),versions=versions,attempts=rows,
        current_trial_budget=315,correctness_repairs_are_not_extra_recipes=True,
        final_version_transitions='See protocol.json correctness_version, version_transition.json in v6 and version_transition_v7.json in v7. Guard-only v6 reuse and stock-only-control v7 branch reuse have explicit proofs; no affected-control outcome existed before its fix.')
    atomic(output/'attempt_history.json',result);csv_gz(output/'attempt_history.csv.gz',rows);return result


def fitted_data_audit(output):
    """Audit saved fit/OOF ownership without refitting or opening neural values."""
    output=Path(output);protocol=read(output/'protocol.json');training=protocol['binding']['partitions'][SCOPES[0]]['warmup']['session_ids']
    if len(training)!=60:raise ValueError('Expected exact first60 fitting sessions')
    proofs=[]
    for c in CATALOG:
        p=output/'fits'/(c['id']+'.json');d=read(p)
        if d['train_ids']!=training:raise ValueError('Changed fit ownership')
        for fold,n in zip(d['folds'],(20,30,40,50)):
            if fold['train_ids']!=training[:n] or fold['test_ids']!=training[n:n+10]:raise ValueError('Nonchronological fitting fold')
            if max(fold['train_ids'])>=min(fold['test_ids']):raise ValueError('Overlapping fit/OOF sessions')
        for k in ('mean','sd','beta'):
            if not np.isfinite(d['model'][k]).all():raise ValueError('Nonfinite fitted coefficients')
        oof=output/'fits'/(c['id']+'.oof.npz')
        if sha(oof)!=d['oof_sha256']:raise ValueError('Changed OOF calibration file')
        with np.load(oof) as z:
            if not np.isin(z['session'],training[20:]).all():raise ValueError('OOF data escaped fitting period')
            if not np.isfinite(z['score']).all() or not np.isfinite(z['outcome']).all():raise ValueError('Nonfinite calibration inputs')
            count=len(z['score']);sessions=len(np.unique(z['session']))
        cal=d['calibration']
        if cal['observations']!=count or cal['sessions']!=sessions:raise ValueError('Calibration counts disagree')
        if cal['valid']!=(count>=1000 and sessions>=20):raise ValueError('Calibration minimum weakened')
        proofs.append(dict(recipe=c['id'],fit_sha256=sha(p),oof_sha256=sha(oof),fit_sessions=60,
            OOF_examples=count,OOF_sessions=sessions,calibration_valid=cal['valid'],units=d['model_units'],neural_used=d['neural_used']))
    result=dict(verified_UTC=utc(),passed=True,proofs=proofs,fit_sessions=training,
        no_neural_retraining=True,protected_outcomes_accessed=False,
        label_purge='Every holding label stays within one session; fitting and OOF session sets strictly separated.',
        calibration_uncertainty='Standard error of session means; one-SE hurdle is a heuristic, not guaranteed statistical coverage.')
    atomic(output/'fitted_data_audit.json',result);return result


def comparisons(rows,output):
    out=[]
    controls={'D01':['D01_partner','D01_self','D01_no_graph'],'D03':['D03_no_stall','D03_own','D03_momentum'],
        'D04':['D04_stock_hedge','D04_etf_only','D04_stock_only']}
    for scope in SCOPES:
        for base,cs in controls.items():
            a=next(r for r in rows if r['scope']==scope and r['config']['id']==base and r['gate']=='net_edge' and r['scenario']=='cost1')
            with np.load(Path(a['root'])/'minute_tapes.npz') as z:ad=z['daily'].copy()
            ids=set(read(output/'protocol.json')['binding']['partitions'][scope].get('warmup',{}).get('session_ids',[]))
            ad=ad[~np.isin(ad[:,0],list(ids))]
            for control in cs:
                b=next(r for r in rows if r['scope']==scope and r['config']['id']==control and r['gate']=='net_edge' and r['scenario']=='cost1')
                with np.load(Path(b['root'])/'minute_tapes.npz') as z:bd=z['daily'].copy()
                bd=bd[np.isin(bd[:,0],ad[:,0])];np.testing.assert_array_equal(ad[:,0],bd[:,0])
                out.append(dict(scope=scope,base=base,control=control,**paired_uncertainty(ad[:,1],bd[:,1]),
                    base_roundtrips=a['metrics']['roundtrips'],control_roundtrips=b['metrics']['roundtrips'],
                    base_activity=a['metrics']['active_sessions'],control_activity=b['metrics']['active_sessions']))
    for scope in SCOPES:
        for base in ('D01','D03','D04'):
            a=next(r for r in rows if r['scope']==scope and r['config']['id']==base and r['gate']=='net_edge' and r['scenario']=='cost1')
            with np.load(Path(a['root'])/'minute_tapes.npz') as z:ad=z['daily'].copy()
            warm=set(read(output/'protocol.json')['binding']['partitions'][scope].get('warmup',{}).get('session_ids',[]));ad=ad[~np.isin(ad[:,0],list(warm))]
            for gate in ('no_gate','fixed_score','matched_random'):
                b=next(r for r in rows if r['scope']==scope and r['config']['id']==base and r['gate']==gate and r['scenario']=='cost1')
                with np.load(Path(b['root'])/'minute_tapes.npz') as z:bd=z['daily'].copy()
                bd=bd[np.isin(bd[:,0],ad[:,0])];np.testing.assert_array_equal(ad[:,0],bd[:,0])
                out.append(dict(scope=scope,base=base,control=base+'_'+gate,layer='D08',**paired_uncertainty(ad[:,1],bd[:,1]),
                    base_roundtrips=a['metrics']['roundtrips'],control_roundtrips=b['metrics']['roundtrips'],
                    base_activity=a['metrics']['active_sessions'],control_activity=b['metrics']['active_sessions'],
                    matching_status='insufficient_joint_activity' if min(a['metrics']['roundtrips'],b['metrics']['roundtrips'])==0 else 'requires_entry-risk_matched_analysis',
                    matching_note='Random-gate probability was frozen from train-only economic admissions; realized activity/risk must be reported and cannot be assumed matched.'))
    atomic(output/'paired_controls.json',out);return out


def dispositions(output):
    """Fail closed on an eligible but unimplemented conditional stage."""
    output=Path(output);selection=read(output/'selection.json');quotes=read(output/'quote_results.json')['rows']
    candidates={r['id']:r for r in selection['candidates']};designs=[];pending=[]
    engineering=read(output/'engineering/train_streams.json')
    failed_marks=sum(not p['verification']['open_close_risk_valid'] for p in engineering['proofs'])
    designs.append(dict(design='D09',status='implemented_accounting_verified_active_risk_failures_retained',evidence=f"Fixed old/new stream, targeted tests and {len(engineering['proofs'])} unchanged train-only stream/stress audits. All cash/NAV reconciled: {engineering['all_accounting_reconciled']}. Active train/stress audits with minute-mark risk failures: {failed_marks}. These remain failures; no atomic fills or discarded exposure."))
    calibrations={name:{k:read(output/'fits'/(name+'.json'))['calibration'].get(k) for k in ('valid','observations','sessions')} for name in ('D01','D03','D04')}
    designs.append(dict(design='D08',status='implemented_and_evaluated',evidence='Same-horizon arithmetic-dollar OOF calibration, feasible integer costs, all hedge fees, borrow and adverse buffer; net_edge, no_gate, fixed_score, train-frequency random controls in 315-trial ledger. Zero activity is insufficient evidence; uncertainty hurdle is heuristic. Frozen minimum: 1000 OOF examples over 20 sessions. Observed calibration: '+str(calibrations)))
    valid_bases=[]
    for name in ('D01','D03','D04'):
        c=candidates[name];q=[r for r in quotes if r['config']['id']==name and r['scenario']=='primary' and r['scope'] in SCOPES[:2]]
        sufficient=c['activity'] and c['valid']
        quote_valid=len(q)==2 and all(r['metrics']['accounting_valid'] and (r['metrics']['quote_coverage'] or 0)>=.99 and r['metrics']['queried_fraction']==1 for r in q)
        if sufficient and not q:
            pending.append(name+': eligible active base needs guarded quote replay; missing retrieval is not a deferral')
        if sufficient and quote_valid:valid_bases.append(name)
        designs.append(dict(design=name,status='historical_first_wave_complete',evidence=f"Development completed round trips {c['roundtrips']}, active sessions {c['active_sessions']}, five-block round trips {c['block_roundtrips']}, risk/accounting valid {c['valid']}, activity gate {c['activity']}; primary quote development rows {len(q)}, quote-valid {quote_valid}. Proxy results are distinct from executable bid/ask evidence."))
    for name,eligible,reason in [
        ('D06',any(s in valid_bases for s in ('D01','D03')),'Needs an accounting/risk-valid,99%-fresh quote-valid D01/D03 base with frozen development activity.'),
        ('D02',('D01' in valid_bases),'Needs D01 graph/pair activity,quote validity and positive replicated/control-incremental development/OOF evidence before relay paths.'),
        ('D05',('D01' in valid_bases),'Needs D01 graph/pair activity,quote validity and positive replicated/control-incremental development/OOF evidence before clock corrections.'),
        ('D07',bool(valid_bases),'Needs valid active quote base and adequate train-only reversal/momentum action evidence.'),
        ('D10',bool(valid_bases),'Needs valid active quote base and adequate OOF quote/decay evidence; mandatory exits never wait.'),
    ]:
        # When the prefix gate passes, investigate remaining exact conjuncts
        # rather than automatically calling the design deferred.
        if eligible:
            status='pending_exact_gate_audit_or_implementation';pending.append(name)
        else:status='deferred_preregistered_base_gate_unmet'
        designs.append(dict(design=name,status=status,evidence=reason+' Available valid active primary quote bases:'+str(valid_bases)+'. Specific first-wave counts and validity are in the rows above and selection.json.'))
    result=dict(created_UTC=utc(),designs=sorted(designs,key=lambda d:d['design']),pending_required_work=pending,
        preregistered_gate_source='protocol.json binding.stages',quote_absence_is_not_coverage=True)
    atomic(output/'design_dispositions.json',result);return result


def quote_comparisons(rows,output):
    controls={'D01':['D01_partner','D01_self','D01_no_graph'],'D03':['D03_no_stall','D03_own','D03_momentum'],
        'D04':['D04_stock_hedge','D04_etf_only','D04_stock_only']};out=[]
    for base,names in controls.items():
        for scope in SCOPES:
            a=next((r for r in rows if r['scope']==scope and r['config']['id']==base and r['gate']=='net_edge' and r['scenario']=='primary'),None)
            if a is None:continue
            for name in names:
                b=next((r for r in rows if r['scope']==scope and r['config']['id']==name and r['gate']=='net_edge' and r['scenario']=='primary'),None)
                if b is None:out.append(dict(base=base,scope=scope,control=name,status='missing_required_quote_control'));continue
                with np.load(Path(a['root'])/'minute_tapes.npz') as z:ad=z['daily'].copy()
                with np.load(Path(b['root'])/'minute_tapes.npz') as z:bd=z['daily'].copy()
                warm=read(output/'protocol.json')['binding']['partitions'][scope].get('warmup',{}).get('session_ids',[])
                ad=ad[~np.isin(ad[:,0],warm)];bd=bd[np.isin(bd[:,0],ad[:,0])];np.testing.assert_array_equal(ad[:,0],bd[:,0])
                out.append(dict(base=base,scope=scope,control=name,status='complete',**paired_uncertainty(ad[:,1],bd[:,1])))
    atomic(output/'paired_quote_controls.json',out);return out


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);proxy=read(output/'proxy_results.json');quote=read(output/'quote_results.json')
    rows=proxy['rows']+quote['rows'];selection=read(output/'selection.json')
    paired=comparisons(proxy['rows'],output)
    quoted_paired=quote_comparisons(quote['rows'],output)
    disposition=read(output/'design_dispositions.json')
    qualified=[];qualification=[]
    for c in selection['candidates']:
        qr=[r for r in quote['rows'] if r['config']['id']==c['id']]
        prim=[r for r in qr if r['scenario']=='primary'];stress=[r for r in qr if r['scenario']=='adverse_0p5']
        later=[r for r in prim if r['scope']==SCOPES[2]]
        paired_dev=[x for x in quoted_paired if x['base']==c['id'] and x['scope'] in SCOPES[:2]]
        improves=len(paired_dev)==6 and all(x['status']=='complete' and x['mean_daily_return_difference']>0 for x in paired_dev)
        causal=read(output/'causal_quote_validation.json')['passed'] if (output/'causal_quote_validation.json').exists() else False
        passed=c['id'] in ('D01','D03','D04') and c['development_qualified'] and improves and causal and len(prim)==3 and len(stress)==3
        passed=bool(passed and all(r['metrics']['accounting_valid'] and (r['metrics']['quote_coverage'] or 0)>=.99 and r['metrics']['queried_fraction']==1 and r['metrics']['total_return']>0 for r in prim)
            and later and (later[0]['metrics']['Sharpe'] or -999)>=1 and later[0]['metrics']['active_sessions']>=20
            and all(r['metrics']['accounting_valid'] and r['metrics']['total_return']>0 for r in stress))
        qualification.append(dict(id=c['id'],qualified=passed,positive_paired_development_improvement=improves))
        if passed:qualified.append(c['id'])
    summary=dict(created_UTC=utc(),proxy_trials=len(proxy['rows']),quote_trials=len(quote['rows']),qualified=qualified,
        selection=selection,qualification=qualification,conditional_dispositions=disposition,
        model_contribution='No HYB_LN incremental-value claim unless eligible D06 is actually completed and beats LN4/matched controls.',
        neural_confirmation_evaluated=any(r['config']['design']=='D06' for r in rows),
        neural_forecasts_used=any(r['config']['design']=='D06' for r in rows),
        all_history_exposed=True,forward_test_completed=False)
    atomic(output/'summary.json',summary)
    fig,axs=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for col,base in enumerate(('D01','D03','D04')):
        for scope,color in zip(SCOPES,('#1f77b4','#ff7f0e','#2ca02c')):
            for gate,style in [('net_edge','o-'),('no_gate','x--')]:
                a=sorted([r for r in proxy['rows'] if r['config']['id']==base and r['gate']==gate and r['scope']==scope],key=lambda r:r['settings']['adverse_bps'])
                cost=[r['settings']['adverse_bps'] for r in a]
                invalid=any(not r['metrics']['accounting_valid'] for r in a)
                for row,metric,mult in [(0,'total_return',100),(1,'Sharpe',1),(2,'maximum_drawdown',100)]:
                    values=[r['metrics'][metric]*mult if r['metrics'][metric] is not None else np.nan for r in a]
                    axs[row,col].plot(cost,values,style,label=scope+' / '+gate+(' [invalid risk]' if invalid else ''),color=color,alpha=1 if gate=='net_edge' else .65)
                    axs[row,col].set_xlabel('Per-side flat cost (bp)');axs[row,col].grid(alpha=.2)
        axs[0,col].set_title(base+': gate and no-gate control')
    for row,title in enumerate(('Net return (%)','Daily Sharpe (cash days included)','Minute drawdown (%)')):axs[row,0].set_ylabel(title)
    axs[0,0].legend(fontsize=6);fig.suptitle('Minute-open proxies | dashed invalid paths retained; cash Sharpe undefined')
    fig.savefig(output/'cost_curves.png',dpi=150);plt.close(fig)
    img=base64.b64encode((output/'cost_curves.png').read_bytes()).decode()
    def val(v):return 'not measured / undefined' if v is None else format(v,'.5g') if isinstance(v,(int,float)) else html.escape(str(v))
    trs=''.join('<tr>'+''.join('<td>'+val(v)+'</td>' for v in [r['scope'],r['config']['id'],r['gate'],r['scenario'],r['mode'],r['metrics']['total_return'],r['metrics']['Sharpe'],r['metrics']['roundtrips'],r['metrics']['active_sessions'],r['verification']['accounting_reconciled'] and r['shadow_verification']['accounting_reconciled'],r['metrics']['accounting_valid'],r['metrics']['risk_failure_minutes'],r['metrics']['quote_coverage']])+'</tr>' for r in rows)
    ds=''.join('<tr><td>'+html.escape(d['design'])+'</td><td>'+html.escape(d['status'])+'</td><td>'+html.escape(d['evidence'])+'</td></tr>' for d in disposition['designs'])
    document='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Decision strategy study</title><style>body{font:16px/1.5 system-ui;max-width:1500px;margin:35px auto;padding:0 25px;color:#122138;background:#f6f8fc}h1,h2{color:#112d4e}table{border-collapse:collapse;width:100%;font-size:13px;background:white}td,th{padding:8px;border-bottom:1px solid #dce3ef;text-align:left}th{position:sticky;top:0;background:#112d4e;color:white}.notice{padding:20px;background:#fff1d8;border-left:5px solid #d48a11}.scroll{overflow:auto;max-height:680px}img{max-width:100%;background:white}code{word-break:break-all}</style></head><body>'''
    document+=f'<h1>Bounded decision-strategy research</h1><p>{len(proxy["rows"])} minute-open proxy trials; {len(quote["rows"])} quote replays. Historically qualified: <strong>{html.escape(str(qualified))}</strong>.</p>'
    primary=[r for r in proxy['rows'] if r['config']['id'] in ('D01','D03','D04') and r['gate']=='net_edge' and r['scenario']=='cost1']
    controls=[r for r in proxy['rows'] if r['config']['id'] in ('D01','D03','D04') and r['gate']=='no_gate' and r['scenario']=='cost1']
    cash=sum(r['metrics']['roundtrips']==0 and r['metrics']['turnover_dollars']==0 for r in primary)
    invalid=sum(not r['metrics']['accounting_valid'] for r in controls)
    document+=f'<p>At the 1 bp proxy, {cash} of {len(primary)} primary strategy/cohort paths stayed entirely in cash. Across the corresponding no-gate paths, {invalid} of {len(controls)} fail the combined accounting/risk contract. Zero activity cannot establish a trading edge; a reconciled but risk-invalid path cannot qualify.</p>'
    document+='<div class="notice">All three historical cohorts were previously exposed. Reserved outcomes remain sealed through January 13, 2027. No forward confirmation, deployment or guaranteed profitability is claimed. Cash-only Sharpe is undefined. An empty quote set does not establish quote coverage.</div>'
    document+='<h2>What changed</h2><p>D09 evaluates every stock fill separately and reconciles integer holdings, fees, cash and NAV. D08 uses expanding train-only same-horizon calibration and full round-trip dollars. The first wave keeps stated controls and fixed criteria. Pre-2025 coarse industry groups replace the hindsight 2026 taxonomy; nine assignments use documented business descriptions. The mandated 250-stock universe remains retrospectively selected. D04 adds eleven explicitly identified sector ETFs.</p>'
    document+='<p>The prior round completed210 minute-open proxy trials and30 closing benchmarks, qualified no candidate and had an empty development shortlist. It executed ZERO new bid/ask strategy replays. Isolated later self-reversal did not replicate in development; some books failed risk despite exact NAV reconciliation, and graph activity was sparse. S10 and the prior price-only graph strategies did not establish HYB_LN value.</p>'
    document+='<h2>Design dispositions</h2><table><tr><th>Design</th><th>Status</th><th>Evidence</th></tr>'+ds+'</table>'
    document+='<h2>Primary rules and their no-gate controls at 1 bp</h2><table><tr><th>Cohort</th><th>Rule</th><th>Gate</th><th>Return (%)</th><th>Daily Sharpe</th><th>Completed baskets</th><th>Risk-failure minutes</th><th>Combined valid</th></tr>'
    for r in primary+controls:
        values=[r['scope'],r['config']['id'],r['gate'],r['metrics']['total_return']*100,r['metrics']['Sharpe'],r['metrics']['roundtrips'],r['metrics']['risk_failure_minutes'],r['metrics']['accounting_valid']]
        document+='<tr>'+''.join('<td>'+val(v)+'</td>' for v in values)+'</tr>'
    document+='</table><p>Paired five-session-block bootstrap intervals and every family control appear in paired_controls.json. Same-session differences retain cash days. Exact long, short, hedge-long and hedge-short P&amp;L, fees, turnover and allocated actual borrow reconcile to evaluation NAV in each trial’s independent_attribution.json.</p>'
    document+='<p>Flat-cost proxies apply their per-side cost through execution prices and disable the separate broker fee schedule, so their fee columns are zero and price-cost effects remain in trade cash P&amp;L. Bid/ask replays, when eligible, charge the frozen order-level fees separately. These are distinct cost experiments.</p>'
    document+='<h2>Cost sensitivity</h2><p>These curves are per-side cost proxies, distinct from bid/ask execution. Zero execution cost retains3% annual short borrowing. Price-only performance cannot establish HYB_LN value.</p><img alt="Net return, daily Sharpe and drawdown curves" src="data:image/png;base64,'+img+'">'
    document+='<h2>Trial ledger</h2><p>The first 60 earlier sessions fit models only and are excluded from earlier evaluation. Every cohort retains its causal 20-session cash volatility warmup. Cash days enter daily Sharpe; zero variance gives undefined Sharpe. Return is a fraction. The combined validity column requires scaled and unscaled accounting, fill/mark risk, liquidation and action checks.</p><div class="scroll"><table><tr>'+''.join('<th>'+s+'</th>' for s in ['Cohort','Strategy','Gate','Scenario','Mode','Net return','Daily Sharpe','Round trips','Active days','Cash/NAV reconciled','Combined valid','Risk-failure minutes','Quote coverage'])+'</tr>'+trs+'</table></div>'
    document+='<h2>Execution and evidence limits</h2><p>Risk is assessed after each simulated child fill and at minute marks. Snapshot data cannot certify unseen intraminute excursions or quote replenishment. Displayed depth is consumed once per timestamp. Short locates,3% borrow and route charges remain assumptions. Same-dollar hedges need not be beta neutral. D08 uncertainty is a heuristic. Stops request delayed execution and cannot guarantee a stop price.</p>'
    overnight=sum(r['metrics']['overnight_failure_sessions']>0 for r in rows)
    residual=sum(bool(r['verification']['unliquidated'] or r['shadow_verification']['unliquidated']) for r in rows)
    document+=f'<p>{overnight} scaled paths retain at least one failed overnight liquidation; {residual} paths retain a final residual basket in the scaled or unscaled trace. These paths are invalid. Borrow attribution reconciles the simulator’s observed-minute charges; unobserved overnight calendar carry is not included in those invalid-path returns. Residual marked positions and action failures remain in the underlying tapes.</p>'
    document+='<p>The conservative sequential integer projection can reject high-priced or poorly matched legs even when a continuously divisible portfolio looks neutral. The reported result concerns these frozen implementations and constraints. It does not prove that every possible strategy in these families lacks predictability. Controls with different realized activity are disclosed as unmatched; zero joint activity provides no basis for a matched causal or model-value claim.</p>'
    document+='<p>At each completed decision clock the source keeps up to three candidates by absolute fitted relative-return score, then evaluates feasible integer quantities and full dollar costs. Calibration uses OOF score bins and session-mean standard errors; unreliable calibration abstains. Superseded and interrupted correctness runs remain separately indexed in attempt_history.json. V6 adds a finite-cost guard; 19 earlier completed trials have exact single-line guard-equivalence receipts.</p>'
    document+='<p>V7 removes industry breadth from the D04 stock-only signal before that control’s first development trial. Only this control is refit on the first 60 sessions. Eleven models and their unchanged trial data carry forward with branch-equivalence and byte-hash receipts; all original versions remain preserved. This is an information-restriction correction, with unchanged primary rules and qualification criteria.</p>'
    document+='<h2>Publication, crowding and scientific value</h2><p>The actual qualified set is '+html.escape(str(qualified))+'. Publication therefore documents a bounded result and an execution audit; it does not reveal a demonstrated deployable edge. Any apparent relation may decay or be crowded. Scientific value rests on causal partitions, explicit controls, complete failure records and reproducible accounting. SIP prices and quotes do not identify informed traders or their private information. Directed associations do not establish causal transmission.</p>'
    document+='<p>Fees follow the frozen scenario using the <a href="https://www.interactivebrokers.com/en/pricing/commissions-stocks.php">official IBKR schedule</a>; the quote source is <a href="https://docs.alpaca.markets/us/docs/historical-stock-data-1">Alpaca SIP</a>. See FILE_PATHS.txt, protocol.json, paired_controls.json and design_dispositions.json for exact inputs and machine-readable evidence.</p></body></html>'
    (output/'report.html').write_text(document,encoding='utf-8')
    (output/'judge_questions.html').write_text(document,encoding='utf-8')
    details=[]
    for r in rows:
        details.append(dict(scope=r['scope'],strategy=r['config']['id'],gate=r['gate'],scenario=r['scenario'],mode=r['mode'],
            stream_candidates=r['stream_candidates'],**r['metrics'],root=r['root'],query_closure_passes=r['query_closure_passes'],
            **{'scaled_'+k:v for k,v in r['verification'].items()},**{'unscaled_'+k:v for k,v in r['shadow_verification'].items()}))
    fields=list(dict.fromkeys(k for r in details for k in r))
    with (output/'trial_ledger.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in details:w.writerow({k:json.dumps(clean(v),separators=(',',':')) if isinstance(v,(dict,list,tuple,np.ndarray)) else v for k,v in r.items()})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['verify','report','attribute','dispositions','fits','history']);p.add_argument('--output',default=str(STUDY/'decision_strategy_execution_2026-10-27_v7'));p.add_argument('--partial',action='store_true');a=p.parse_args()
    if a.command=='verify':print(validate_files(a.output))
    elif a.command=='attribute':export_and_attribute(Path(a.output),a.partial)
    elif a.command=='dispositions':print(dispositions(Path(a.output)))
    elif a.command=='fits':print(fitted_data_audit(Path(a.output)))
    elif a.command=='history':print(attempt_history(Path(a.output)))
    else:render(Path(a.output))
