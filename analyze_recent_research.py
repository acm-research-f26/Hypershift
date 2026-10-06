"""Fixed post-run diagnostics: SPY-relative returns and random top-K controls.

This never modifies a selection lock, model, portfolio policy or prediction.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_engine import load_panel, prepare_panel, fold_indices, backtest
from run_architecture_research import bootstrap_difference, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('runs/recent_2024_2025'))
    args = parser.parse_args()
    root = args.run
    f = pd.read_csv(root/'all_metrics.csv')
    primary = f[(f.seed=='ensemble') & f.selected_policy & (f.cost_multiplier==1)]
    locks = json.loads((root/'SELECTION_LOCK.json').read_text())
    relative, controls, intervals, coverage_rows = [], [], {}, []
    raw_coverage = pd.read_csv('data/recent_cohorts/coverage.csv')
    exclusions = json.loads(Path('config/recent_identity_exclusions.json').read_text())
    for name in primary.dataset.unique():
        data = prepare_panel(load_panel(name))
        p = data['panel']
        cov = raw_coverage.set_index('ticker').reindex(p.tickers).reset_index()
        cov['dataset'] = name
        cov['identity_rejected'] = cov.ticker.isin(exclusions)
        cov['usable_history'] = (cov.observed_2023 > 0) & ~cov.identity_rejected
        coverage_rows.extend(cov.to_dict('records'))
        for year in sorted(primary[primary.dataset==name].year.unique()):
            _, _, test = fold_indices(data, int(year))
            folder = root/f'{name}_{year}'
            spy = pd.read_csv(folder/'market_proxy_daily.csv')
            winner = locks[name]['by_year'][str(year)]['winner']
            for _, row in primary[(primary.dataset==name) & (primary.year==year)].iterrows():
                path = pd.read_csv(folder/f'{row.model}_daily.csv')
                assert path.date.equals(spy.date), 'SPY/model date mismatch'
                a, b = path.net.to_numpy(), spy.net.to_numpy()
                active = a-b
                excess_b = b-((1.03)**(1/252)-1)
                excess_a = a-((1.03)**(1/252)-1)
                beta, intercept = np.linalg.lstsq(np.c_[excess_b, np.ones(len(b))], excess_a, rcond=None)[0]
                tracking = active.std(ddof=1)*np.sqrt(252)
                relative.append(dict(dataset=name, year=year, model=row.model,
                                     spy_net_sharpe=float(primary[(primary.dataset==name)&(primary.year==year)&(primary.model=='market_proxy')].net_sharpe.iloc[0]),
                                     active_annualized_mean=float(active.mean()*252), tracking_error=float(tracking),
                                     information_ratio=float(active.mean()*252/tracking) if tracking>1e-12 else None,
                                     beta_to_spy=float(beta), alpha_annualized_ols=float(intercept*252),
                                     first_date=path.date.iloc[0], last_date=path.date.iloc[-1],
                                     prediction_periods=len(test),
                                     eligible_min=int(data['mask'][test].sum(1).min()),
                                     eligible_max=int(data['mask'][test].sum(1).max()),
                                     missing_test_labels=int((data['mask'][test]&~data['labelmask'][test]).sum())))
            selected_path = pd.read_csv(folder/f'{winner}_daily.csv').net.to_numpy()
            intervals[f'{name}_{year}'] = dict(winner=winner, comparisons={})
            for base in ['think', 'buy_hold', 'market_proxy']:
                intervals[f'{name}_{year}']['comparisons'][base] = bootstrap_difference(
                    selected_path, pd.read_csv(folder/f'{base}_daily.csv').net.to_numpy())
            # Fifty fixed random score paths, paired across K. These are diagnostic
            # controls, never candidates for validation or post-test model selection.
            for seed in range(50):
                rng = np.random.default_rng(89000 + int(year)*100 + seed)
                scores = rng.standard_normal(data['y'][test].shape)
                for k in [5, 10, 20]:
                    metric, _, _ = backtest(data, test, scores, k, 'equal')
                    controls.append(dict(dataset=name, year=year, seed=seed, k=k, **metric))
            print(f'Diagnostics {name}/{year}: 150 random top-K controls', flush=True)
    pd.DataFrame(relative).to_csv(root/'spy_relative_metrics.csv', index=False)
    pd.DataFrame(controls).to_csv(root/'random_topk_controls.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(root/'universe_coverage.csv', index=False)
    write_json(root/'paired_bootstrap_by_year.json', intervals)


if __name__ == '__main__':
    main()
