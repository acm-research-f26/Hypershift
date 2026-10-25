"""Verify saved ranking intervals and publish consolidated docs; no training."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_engine import performance
from report_pooled_uncertainty import indices, bootstrap_sharpe, interval

ROOT=Path('runs/robust_comparison')
RUN=Path('runs/huber_ranking_2022_2025')
BASE='think_mix_huber'
NEW=['think_mix_huber_rank_01','think_mix_huber_rank_03','think_mix_huber_rank_10']
KINDS=[BASE]+NEW+['ranking_choice']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(path,body):
    start,end='<!-- huber-ranking:start -->','<!-- huber-ranking:end -->'
    text=path.read_text(encoding='utf-8')
    if start in text:
        before,tail=text.split(start,1)
        _,after=tail.split(end,1)
        text=before+after.lstrip('\n')
    first,tail=text.split('\n',1)
    path.write_text(first+'\n\n'+start+'\n'+body+'\n'+end+'\n\n'+tail.lstrip('\n'),encoding='utf-8')


def name(kind):
    return {BASE:'THINK+mixer+Huber',NEW[0]:'Huber+ranking, weight 0.1',NEW[1]:'Huber+ranking, weight 0.3',
            NEW[2]:'Huber+ranking, weight 1.0','ranking_choice':'Validation choice of ranking weight',
            'market_proxy':'SPY','qqq':'QQQ','think':'THINK','think_mix':'THINK+mixer / MSE'}[kind]


def ci(row,key):
    return f'[{row[key+"_ci_low"]:.3f}, {row[key+"_ci_high"]:.3f}]'


def verify(pooled,curves):
    checked=0
    for (dataset,policy),frame in curves.groupby(['dataset','policy'],sort=False):
        base=frame[frame.model==BASE]
        ids=indices(base.groupby('fold_year',sort=True).size().tolist())
        samples={}
        for kind,path in frame.groupby('model',sort=False):
            np.testing.assert_array_equal(path.date,base.date)
            np.testing.assert_array_equal(path.fold_year,base.fold_year)
            assert path.date.is_unique and path.date.is_monotonic_increasing
            assert np.isfinite(path[['gross','net']]).all().all()
            row=pooled[(pooled.dataset==dataset)&(pooled.policy==policy)&(pooled.model==kind)].iloc[0]
            for side in ['net','gross']:
                for key,value in performance(path[side]).items():
                    np.testing.assert_allclose(value,row[f'{side}_{key}'],rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose((1+path.net).cumprod(),path.net_wealth,rtol=1e-10,atol=1e-12)
            samples[kind]=bootstrap_sharpe(path.net,ids)
            np.testing.assert_allclose(interval(samples[kind]),[row.net_sharpe_ci_low,row.net_sharpe_ci_high],atol=1e-10)
            checked+=1
        for kind,values in samples.items():
            row=pooled[(pooled.dataset==dataset)&(pooled.policy==policy)&(pooled.model==kind)].iloc[0]
            for control,key in [(BASE,'huber'),('market_proxy','spy'),('qqq','qqq')]:
                np.testing.assert_allclose(interval(values-samples[control]),
                    [row[f'delta_{key}_ci_low'],row[f'delta_{key}_ci_high']],atol=1e-10)
                other=pooled[(pooled.dataset==dataset)&(pooled.policy==policy)&(pooled.model==control)].iloc[0]
                np.testing.assert_allclose(row.net_sharpe-other.net_sharpe,row[f'delta_{key}'],atol=1e-12)
    assert checked==36
    # Frozen training source is authoritative after the subsequent CUDA setup.
    protocol=json.loads((RUN/'PROTOCOL.json').read_text())
    for source,expected in protocol['code_hashes'].items():
        assert digest(RUN/'source_snapshot'/Path(source).name)==expected
    for source,expected in json.loads((RUN/'DATA_INPUT_LOCK.json').read_text()).items():
        assert digest(Path(source))==expected
    chronology=json.loads((RUN/'CHRONOLOGY_AUDIT.json').read_text())
    assert chronology['passed'] and chronology['folds']==8
    # Reconstruct validation-only choices from the recorded policies and verify
    # zero is a candidate and each test year cannot use later validation scores.
    policies=json.loads((RUN/'validation_search.json').read_text())
    lock=json.loads((RUN/'RANKING_SELECTION_LOCK.json').read_text())
    family=[p for p in policies if p['model'] in [BASE]+NEW]
    actual_hash=hashlib.sha256(json.dumps(sorted(family,key=lambda p:(p['dataset'],p['year'],p['model'])),sort_keys=True).encode()).hexdigest()
    assert actual_hash==lock['validation_family_sha256']
    for choice in lock['choices']:
        expected={}
        for kind in [BASE]+NEW:
            history=[p for p in family if p['dataset']==choice['dataset'] and p['model']==kind and p['year']<=choice['year']]
            values=[p['selected']['score'] if choice['policy']=='validation-selected' else
                    next(c['score'] for c in p['candidates'] if c['method']=='equal' and c['k']==10) for p in history]
            expected[kind]=float(np.median(values)-.25*np.std(values))
        assert expected==choice['validation_scores']
        assert choice['model']==max(expected,key=expected.get)
    return checked


def main():
    pooled=pd.read_csv(ROOT/'huber_ranking_pooled.csv')
    curves=pd.read_csv(ROOT/'huber_ranking_curves.csv')
    checked=verify(pooled,curves)
    original=pd.read_csv(ROOT/'combined_2022_2025_metrics.csv')
    original_curves=pd.read_csv(ROOT/'combined_2022_2025_curves.csv')
    original['policy']='validation-selected'
    original['experiment']='original batch'
    additions=pooled[(pooled.policy=='equal K10')|pooled.model.isin(NEW+['ranking_choice'])].copy()
    additions['horizon']=5
    additions['experiment']='Huber+ranking follow-up'
    additions['accounting']='annual-reset fold compounding; boundary gaps omitted'
    current=pd.concat([original,additions],ignore_index=True)
    assert not current.duplicated(['horizon','dataset','policy','model']).any()
    assert len(current)==102
    for key in ['net_sharpe_ci_low','net_sharpe_ci_high','delta_spy_ci_low','delta_spy_ci_high','delta_qqq_ci_low','delta_qqq_ci_high']:
        assert current[key].notna().all()
    original_curves['policy']='validation-selected'
    added_curves=curves[(curves.policy=='equal K10')|curves.model.isin(NEW+['ranking_choice'])].copy()
    added_curves['horizon']=5
    current_curves=pd.concat([original_curves,added_curves],ignore_index=True)
    for key,group in current_curves.groupby(['horizon','dataset','policy','model']):
        assert group.date.is_unique
        row=current[(current.horizon==key[0])&(current.dataset==key[1])&(current.policy==key[2])&(current.model==key[3])].iloc[0]
        np.testing.assert_allclose(performance(group.net)['sharpe'],row.net_sharpe,atol=1e-10)
    current.to_csv(ROOT/'combined_2022_2025_metrics_current.csv',index=False)
    current_curves.to_csv(ROOT/'combined_2022_2025_curves_current.csv',index=False)
    intervals=current[[c for c in current.columns if '_ci_' in c]+['horizon','dataset','model','label','policy','experiment','net_sharpe','delta_spy','delta_qqq']].copy()
    intervals.to_csv(ROOT/'pooled_confidence_intervals_current.csv',index=False)
    k=pd.read_csv(ROOT/'huber_ranking_k_pooled.csv')
    k_checked=0
    for row in k.itertuples():
        paths={}
        for kind in [BASE,row.model]:
            parts=[pd.read_csv(RUN/f'{row.dataset}_{year}'/f'{kind}_equal_k{row.k}_daily.csv') for year in range(2022,2026)]
            paths[kind]=pd.concat(parts,ignore_index=True)
        np.testing.assert_array_equal(paths[BASE].date,paths[row.model].date)
        ids=indices([len(p) for p in parts])
        low,high=interval(bootstrap_sharpe(paths[row.model].net,ids)-bootstrap_sharpe(paths[BASE].net,ids))
        np.testing.assert_allclose([low,high],[row.delta_huber_ci_low,row.delta_huber_ci_high],atol=1e-10)
        k_checked+=1
    header='## Huber plus ranking: updated results and confidence intervals\n\n'
    method=('All ranking results below are **five-session forecasts for 2022–2025**, trained on CPU before CUDA was configured. '
            'The models were not retrained for this documentation update. **95% intervals use 1,000 paired circular '
            '20-session block resamples separately within each annual fold** (seed 20261005). '
            'Sharpe uses the same 3% annual hurdle. Absolute Sharpe uncertainty and paired differences are distinct. '
            'Intervals condition on saved fits, selected policies and observed years; they do not retrain or '
            'adjust for repeated testing, multiple comparisons or survivorship bias.\n\n')
    links=('[Full ranking report](HUBER_RANKING.md) | [Ranking metrics and intervals CSV](huber_ranking_pooled.csv) | '
           '[Consolidated 102-series metrics](combined_2022_2025_metrics_current.csv) | '
           '[Consolidated interval table](pooled_confidence_intervals_current.csv)\n\n')
    tables=''
    for dataset in ['NASDAQ_recent','NYSE_recent']:
        for policy in ['equal K10','validation-selected']:
            tables+=f'### {dataset}: {policy}\n\n'
            tables+='| Model | Net SR | 95% SR CI | ΔSR vs Huber [95% CI] | ΔSR vs SPY [95% CI] | ΔSR vs QQQ [95% CI] |\n'
            tables+='|---|---:|---|---|---|---|\n'
            for kind in KINDS+['market_proxy','qqq']:
                row=pooled[(pooled.dataset==dataset)&(pooled.policy==policy)&(pooled.model==kind)].iloc[0]
                differences=[f'{row[f"delta_{c}"]:+.3f} {ci(row,f"delta_{c}")}' for c in ['huber','spy','qqq']]
                tables+=f'| {name(kind)} | {row.net_sharpe:.3f} | {ci(row,"net_sharpe")} | '+' | '.join(differences)+' |\n'
            tables+='\n'
    conclusion=('**Interpretation:** ranking weight 0.3 reaches NASDAQ Sharpe 0.711 with its selected portfolio, '
                'but its advantage over Huber is +0.037, CI [-0.178, 0.252]. NYSE weight 1.0 reaches 0.239 '
                'versus Huber 0.194, with gain CI [-0.433, 0.473]. Neither establishes an improvement. '
                'The primary fixed-K10 validation-selected ranking procedure changes NASDAQ 0.532→0.488 '
                '(gain CI [-0.129, 0.053]) and NYSE 0.089→0.148 (gain CI [-0.137, 0.260]). '
                'Secondary validation-chosen ranking weights yield NASDAQ 0.638 and NYSE 0.189. '
                'No best test-performing weight or K is promoted to the validation-selected procedure.\n\n')
    section(ROOT/'CONFIDENCE_INTERVALS.md',header+method+links+tables+conclusion+
            '[Verified top-5/top-10/top-20 paired intervals](huber_ranking_k_pooled.csv). '
            'The original batch’s interval tables follow below.\n')
    summary='| Model / procedure | NASDAQ net SR | NYSE net SR |\n|---|---:|---:|\n'
    for kind in KINDS+['market_proxy','qqq']:
        values=[pooled[(pooled.dataset==d)&(pooled.policy=='validation-selected')&(pooled.model==kind)].iloc[0].net_sharpe
                for d in ['NASDAQ_recent','NYSE_recent']]
        summary+=f'| {name(kind)} | {values[0]:.3f} | {values[1]:.3f} |\n'
    summary+='\nEach model uses its own prior-validation-selected portfolio. '
    summary+='The separate ranking-weight choice uses only current-and-prior validation and includes zero weight.\n\n'
    for doc in ['REPORT.md','FINDINGS.md','ETF_CONTROLS.md']:
        section(ROOT/doc,header+summary+conclusion+links+'[Full 95% uncertainty tables](CONFIDENCE_INTERVALS.md). '
                'The original study below is retained as a reference.\n')
    charts=''
    for dataset in ['NASDAQ_recent','NYSE_recent']:
        charts+=f'![{dataset}: all five-session models including ranking variants](figures/huber_ranking_all_models_{dataset}.png)\n\n'
    section(ROOT/'COMBINED_2022_2025.md',header+summary+conclusion+links+
            '[Consolidated dated returns and curves](combined_2022_2025_curves_current.csv). '
            'The current table includes every original series, the three new strengths, '
            'validation-selected ranking procedures and fixed-K10 follow-up comparisons. '
            'Daily series are unchanged.\n\n'+charts+'The original comparison tables below are preserved.\n')
    (ROOT/'ranking_documentation_audit.json').write_text(json.dumps(dict(
        checked_on='2026-10-08',status='passed',ranking_pooled_paths_verified=checked,
        paired_differences_verified=checked*3,k_comparisons_verified=k_checked,
        consolidated_model_control_series=len(current),all_series_returns_and_sharpe_verified=True,
        validation_selections_reconstructed=True,frozen_training_snapshot_verified=True,
        no_retraining=True,no_new_model_or_policy_selection=True,
        input_hashes={str(p):digest(p) for p in [ROOT/'huber_ranking_pooled.csv',ROOT/'huber_ranking_curves.csv',
                                               ROOT/'huber_ranking_k_pooled.csv',RUN/'RANKING_SELECTION_LOCK.json']},
        report_source_sha256=digest(Path(__file__))),indent=2),encoding='utf-8')
    print(f'Verified {checked} ranking paths, {checked*3} paired intervals and {k_checked} K comparisons; published {len(current)} consolidated series.')


if __name__=='__main__':
    main()
