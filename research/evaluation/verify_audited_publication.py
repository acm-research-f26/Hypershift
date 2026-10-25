"""Verify current charts, corrected artifacts, frozen inputs and draw averages."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re
import numpy as np
import pandas as pd
from research_engine import performance

def main():
    completion=json.loads(Path('runs/random_trials_20261010/completion.json').read_text())
    assert completion['neural_fits']==288
    trial=Path('docs/random_trials_20261010')
    assert json.loads((trial/'audit.json').read_text())['independent_share_ledger_paths']==336
    verified=[]
    for root in [Path('docs'),Path('docs/largecap_retraining_20261010'),trial]:
        md=root/('ARCHIVED_NYSE_CHARTS.md' if root==Path('docs/largecap_retraining_20261010') else 'CHARTS.md');text=md.read_text(encoding='utf-8')
        assert 'NYSX' in text and '2026' in text
        for character in [0x2013,0x2014,0x2212,0x00d7]:
            assert chr(character).encode('utf-8').decode('cp1252',errors='replace') not in text,md
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if ':' in link or link.startswith('#'):continue
            target=(md.parent/link.split('#')[0]).resolve();assert target.exists(),target
            verified.append(target)
        curves=pd.read_csv(root/('curves.csv' if root==trial else 'audited_curves.csv'),low_memory=False)
        assert not ((curves.dataset.str.startswith('NYSE'))&(curves.model=='qqq')).any()
        grouping=['dataset','model','policy'] if root==trial else ['horizon','dataset','model']
        rows=pd.read_csv(root/('pooled.csv' if root==trial else 'audited_metrics.csv'))
        for key,piece in curves.groupby(grouping,sort=False):
            assert piece.date.is_unique and piece.date.is_monotonic_increasing
            selected=rows
            for column,value in zip(grouping,key):selected=selected[selected[column]==value]
            assert len(selected)==1,(key,len(selected))
            np.testing.assert_allclose(performance(piece.net)['sharpe'],selected.sharpe.iloc[0],atol=1e-10)
        verified+=[md,root/('curves.csv' if root==trial else 'audited_curves.csv'),root/('pooled.csv' if root==trial else 'audited_metrics.csv')]
    pooled=pd.read_csv(trial/'pooled.csv');means=pd.read_csv(trial/'mean_across_draws.csv')
    for r in means.itertuples():
        sub=pooled[(pooled.market==r.market)&(pooled.model==r.model)&(pooled.policy==r.policy)]
        assert len(sub)==3
        np.testing.assert_allclose([r.mean_sharpe,r.sd_sharpe],[sub.sharpe.mean(),sub.sharpe.std()],atol=1e-12)
    for root in ['data/recent_cohorts','data/largecap_20261010','data/random_trials_20261010']:
        path=Path(root);manifest=json.loads((path/'manifest.json').read_text())
        for name,expected in manifest['hashes'].items():assert hashlib.sha256((path/name).read_bytes()).hexdigest()==expected
    decks=list(Path('docs/slides').glob('*.pptx'))+list(Path('presentations/output').glob('*.pptx'))
    assert len(decks)==4
    for record in json.loads(Path('docs/audit/slide_export_update.json').read_text()):
        assert hashlib.sha256(Path(record['deck']).read_bytes()).hexdigest()==record['deck_sha256']
        archive=Path(record['archive'])
        with ZipFile(archive) as z:
            assert z.testzip() is None
            for name,expected in record['members'].items():
                assert hashlib.sha256(z.read(name)).hexdigest()==expected
                for folder in record['folders']:
                    path=Path(folder)/name
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
                    verified.append(path)
        verified.append(archive)
    doc=Path('docs/largecap_retraining_20261010/Hypershift_Large_Company_Retraining_Results_Benchmark_Corrected.docx')
    for p in decks+[doc,Path('docs/largecap_retraining_20261010/Hypershift_Large_Company_Results_Data.zip')]:
        with ZipFile(p) as z:
            assert z.testzip() is None
            if p.suffix in ['.docx','.pptx']:
                text=''.join(z.read(n).decode('utf-8',errors='ignore') for n in z.namelist() if n.endswith('.xml'))
                assert 'NYSX' in text,p
        verified.append(p)
    verified += [Path('docs/audit/backtests_20261010/EXPLANATION.md'),
                 Path('docs/largecap_retraining_20261010/Hypershift_Large_Company_Retraining_Results.pdf')]
    retirement=json.loads(Path('docs/audit/nasdaq_2026_retirement.json').read_text(encoding='utf-8-sig'))
    assert retirement['status']=='deleted'
    for folder in retirement['folders']:assert not Path(folder['path']).exists()
    assert not list(Path('runs/largecap_20261010').rglob('NASDAQ100_20261010_20*'))
    current=Path('docs/nasdaq_monthly_pit/CHARTS.md')
    current_text=current.read_text(encoding='utf-8')
    assert 'Completed: 120 neural fits, 120 continuous portfolio paths' in current_text
    assert 'official NASDAQ-100 fallback' in current_text
    for link in re.findall(r'\]\(([^)]+)\)',current_text):
        if ':' not in link and not link.startswith('#'):
            linked=(current.parent/link).resolve();assert linked.exists();verified.append(linked)
    active=Path('runs/nasdaq100_historical_monthly_20261011')
    active_completion=json.loads((active/'completion.json').read_text())
    active_audit=json.loads((active/'AUDIT.json').read_text())
    assert active_completion['neural_fits']==120 and active_completion['continuous_portfolio_paths']==120
    assert active_audit['status']=='passed' and active_audit['neural_checkpoints_reproduced']==120
    assert active_audit['independent_share_cash_paths']==120
    assert active_completion['annual_account_resets']==0
    ranking_audit=json.loads((current.parent/'RANKING_AUDIT.json').read_text())
    assert ranking_audit['status']=='passed' and ranking_audit['models']==9
    assert ranking_audit['full_forecast_periods']==200
    for source,expected in ranking_audit['prediction_and_ridge_hashes'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected
    assert hashlib.sha256((active/'ranking.csv').read_bytes()).hexdigest()==ranking_audit['source_ranking_sha256']
    ranking=pd.read_csv(active/'ranking.csv').set_index('model')
    periods=pd.read_csv(current.parent/'ranking_periods.csv')
    fields=['IC','RankIC','NDCG@5','NDCG@10','NDCG@20']
    for model,sub in periods.groupby('model'):
        assert sub.entry_date.is_unique and sub.entry_date.is_monotonic_increasing
        np.testing.assert_allclose(sub[fields].mean(),ranking.loc[model,fields],atol=1e-10)
        assert sub.IC.count()==200 and sub.unobserved.sum()==ranking.loc[model,'unobserved_labels']
        for field in fields:
            assert f'{ranking.loc[model,field]:.4f}' in current_text
    by_year=pd.read_csv(current.parent/'ranking_by_year.csv').set_index(['model','year'])
    np.testing.assert_allclose(by_year[fields+['Random NDCG@10']],
        periods.groupby(['model','year'])[fields+['Random NDCG@10']].mean(),atol=1e-10)
    for filename in ['metrics.csv','annual.csv','monthly.csv','training_records.csv','paired_bootstrap.csv','ranking.csv',
                     'AUDIT.json','PROTOCOL.json','INPUT_CODE_LOCK.json','DEPENDENCY_CODE_LOCK.json','FITS_COMPLETE_LOCK.json','completion.json']:
        assert (active/filename).read_bytes()==(current.parent/filename).read_bytes()
        verified.append(current.parent/filename)
    for source,expected in json.loads((active/'INPUT_CODE_LOCK.json').read_text()).items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected
    for source,expected in json.loads((active/'DEPENDENCY_CODE_LOCK.json').read_text()).items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected
    records=pd.read_csv(active/'training_records.csv')
    assert len(records)==120 and not records.duplicated(['year','model','seed']).any()
    assert set(records.year)=={2022,2023,2024,2025} and set(records.seed)=={7,19,42,61,97}
    curves=pd.read_csv(active/'curves.csv',dtype={'seed':str})
    rows=pd.read_csv(active/'metrics.csv',dtype={'seed':str})
    for key,piece in curves.groupby(['model','seed','k','cost_multiplier'],sort=False):
        selected=rows
        for column,value in zip(['model','seed','k','cost_multiplier'],key):selected=selected[selected[column]==value]
        assert len(selected)==1 and piece.date.is_unique and piece.date.is_monotonic_increasing
        assert piece.date.iloc[0]=='2022-01-03' and piece.date.iloc[-1]=='2025-12-31'
        metrics=performance(piece.net)
        for field in ['sharpe','annualized_return','cumulative_return','maximum_drawdown']:
            np.testing.assert_allclose(metrics[field],selected[field].iloc[0],atol=1e-10)
    assert len(rows)==120
    extension=Path('runs/nasdaq_topk_pseudohuber_20261011')
    extension_audit_path=extension/'AUDIT.json'
    if extension_audit_path.exists():
        extension_out=current.parent/'topk_pseudohuber'
        extension_audit=json.loads(extension_audit_path.read_text())
        extension_complete=json.loads((extension/'completion.json').read_text())
        assert extension_audit['status']=='passed' and extension_audit['new_checkpoints_reproduced']==20
        assert extension_audit['independent_share_cash_paths']==204 and extension_audit['parent_paths_unchanged']==120
        assert extension_complete['portfolio_sizes']==[5,10,20] and extension_complete['new_portfolio_paths']==84
        assert '20 additional pseudo-Huber fits' in current_text
        for filename in ['metrics.csv','curves.csv','annual.csv','monthly.csv','ranking.csv','ranking_periods.csv',
            'ranking_by_year.csv','paired_bootstrap.csv','training_records.csv','PROTOCOL.json','completion.json',
            'AUDIT.json','INPUT_CODE_LOCK.json','SOURCE_ARTIFACT_LOCK.json','FITS_COMPLETE_LOCK.json','VALIDATION_EXPORT_CORRECTION.json']:
            assert (extension/filename).read_bytes()==(extension_out/filename).read_bytes(),filename
            verified.append(extension_out/filename)
        for lockfile in ['INPUT_CODE_LOCK.json','SOURCE_ARTIFACT_LOCK.json']:
            for source,expected in json.loads((extension/lockfile).read_text()).items():
                assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected,source
        for source,expected in json.loads((extension/'FITS_COMPLETE_LOCK.json').read_text())['hashes'].items():
            assert hashlib.sha256((extension/source).read_bytes()).hexdigest()==expected,source
        correction=json.loads((extension/'VALIDATION_EXPORT_CORRECTION.json').read_text())
        assert correction['training_and_test_predictions_changed'] is False and len(correction['exports'])==20
        for record in correction['exports']:
            assert record['test_forecasts_unchanged']
            assert hashlib.sha256(Path(record['corrected_file']).read_bytes()).hexdigest()==record['sha256']
        ecurves=pd.read_csv(extension/'curves.csv',dtype={'seed':str})
        erows=pd.read_csv(extension/'metrics.csv',dtype={'seed':str})
        assert len(erows)==204 and not erows.duplicated(['model','seed','k','cost_multiplier']).any()
        for key,piece in ecurves.groupby(['model','seed','k','cost_multiplier'],sort=False):
            selected=erows
            for column,value in zip(['model','seed','k','cost_multiplier'],key):
                selected=selected[selected[column]==value]
            assert len(selected)==1 and piece.date.is_unique and len(piece)==1003
            calculated=performance(piece.net)
            for field in ['sharpe','annualized_return','cumulative_return','maximum_drawdown']:
                np.testing.assert_allclose(calculated[field],selected[field].iloc[0],atol=1e-10)
        eannual=pd.read_csv(extension/'annual.csv',dtype={'seed':str})
        for key,sub in eannual.groupby(['model','seed','k','cost_multiplier']):
            selected=erows
            for column,value in zip(['model','seed','k','cost_multiplier'],key):
                selected=selected[selected[column]==value]
            np.testing.assert_allclose((1+sub.cumulative_return).prod(),1+selected.cumulative_return.iloc[0],atol=1e-10)
        for k in [5,10,20]:
            assert (extension_out/'figures'/f'wealth_top{k}.png').exists()
            assert (extension_out/'figures'/f'years_top{k}.png').exists()
    short_audit_path=Path('runs/nasdaq_long_short_20261011/AUDIT.json')
    if short_audit_path.exists():
        from verify_ndx_long_short_publication import verify as verify_short_publication
        verified.extend(verify_short_publication(current_text))
    depth_audit_path=Path('runs/nasdaq_rank_depth_20261011/AUDIT.json')
    if depth_audit_path.exists():
        from publish_ndx_rank_depth import verify as verify_rank_depth
        verified.extend(verify_rank_depth(current_text))
    loss_audit_path=Path('runs/nasdaq_loss_similarity_20261011/AUDIT.json')
    if loss_audit_path.exists():
        from publish_ndx_loss_similarity import verify as verify_loss_similarity
        verified.extend(verify_loss_similarity(current_text))
    verified += [current,Path('docs/nasdaq_monthly_pit/DATA_REQUIREMENTS.md'),
                 Path('config/nasdaq_monthly_pit.json'),Path('docs/audit/nasdaq_2026_retirement.json')]
    result=dict(status='passed_historical_monthly_fallback_existing_evidence_and_retirement',as_of='2026-10-11',neural_fits_new=408,
        random_sample_neural_fits=288,historical_monthly_neural_fits=120,
        original_checkpoints_reproduced=384,original_share_ledger_paths=384,new_share_ledger_paths=336,
        checkpoint_reproduction='historical checks completed before deletion; retired NASDAQ files no longer reproducible locally',
        nasdaq_2026_neural_checkpoints_deleted=retirement['neural_checkpoints'],
        nasdaq_2026_ridge_models_deleted=retirement['ridge_models'],
        active_study='NASDAQ100_historical_monthly',active_study_training='completed_exploratory_research',
        historical_monthly_checkpoints_reproduced=120,historical_monthly_independent_share_cash_paths=120,
        historical_monthly_annual_account_resets=0,
        draw_means_verified=True,NYSE_QQQ_overlays_removed=True,NYSX_historical_returns='unavailable, no fabricated curve',
        frozen_CSV_fingerprints_preserved=True,original_Word_file='preserved due to unsaved OneDrive Word session; corrected copy is current',
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(verified))})
    if extension_audit_path.exists():
        result.update(neural_fits_new=428,topk_pseudohuber_new_fits=20,
            topk_pseudohuber_independent_paths=204,topk_pseudohuber_parent_paths_unchanged=120,
            active_study_extension='NASDAQ100_top5_top10_top20_pseudoHuber')
    if short_audit_path.exists():
        result.update(long_short_new_model_fits=0,long_short_total_paths=600,
            long_short_independent_signed_paths=570,long_short_parent_ensemble_paths_unchanged=30,
            long_short_historical_borrow_availability='assumed, not verified',
            active_portfolio_extension='fixed_long_short_and_matched_cash_allocations')
    if depth_audit_path.exists():
        result.update(rank_depth_new_model_fits=0,rank_depth_diagnostic_paths=18,
            rank_depth_original_ensemble_paths_unchanged=6,
            rank_depth_stock_and_year_return_contributions_reconciled=True)
    if loss_audit_path.exists():
        result.update(loss_similarity_new_model_fits=0,loss_similarity_matched_checkpoint_pairs=20,
            loss_similarity_marginal_accounts=12,loss_similarity_chance_fraction='not identifiable')
    Path('docs/audit/FINAL_AUDIT_20261011.json').write_text(json.dumps(result,indent=2))
    # Current delivery inventory supplements the older publication manifest.
    Path('docs/publication_manifest.json').write_text(json.dumps(result,indent=2))
    print('Verified historical monthly charts, 120 audited paths, preserved older evidence and NASDAQ retirement')

if __name__=='__main__':main()
