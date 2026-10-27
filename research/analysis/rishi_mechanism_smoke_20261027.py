"""Bounded exploratory mechanism diagnostics; no financial training or new scoring.

Run from repository root: python -m hypershift run rishi_mechanism_smoke_20261027
Only committed spent-period account curves and synthetic inputs are evaluated.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
OUT = ROOT / 'docs/rishi_mechanism_20261027'


def metrics(r):
    r = np.asarray(r, dtype=float)
    assert len(r) > 1 and np.isfinite(r).all() and (r > -1).all()
    wealth = np.r_[1., np.cumprod(1+r)]
    vol = r.std(ddof=1)*np.sqrt(252)
    return dict(sharpe=(r.mean()-np.expm1(np.log1p(.03)/252))*252/vol,
                sharpe_rf0=r.mean()*252/vol, annualized_return=wealth[-1]**(252/len(r))-1,
                annualized_volatility=vol, maximum_drawdown=(wealth/np.maximum.accumulate(wealth)-1).min(),
                annual_arithmetic_mean=r.mean()*252, days=len(r))


def regression(y, x, lag=5):
    rf = np.expm1(np.log1p(.03)/252)
    x = np.column_stack([np.ones(len(x)), np.asarray(x)-rf])
    y = np.asarray(y)-rf
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y-x@b
    z = x*e[:, None]
    meat = z.T@z
    for k in range(1, lag+1):
        a = z[k:].T@z[:-k]
        meat += (1-k/(lag+1))*(a+a.T)
    bread = np.linalg.inv(x.T@x)
    cov = bread@meat@bread*len(y)/(len(y)-x.shape[1])
    return dict(alpha_arithmetic_annual=b[0]*252, beta=b[1],
                alpha_hac_t=b[0]/np.sqrt(cov[0, 0]) if np.std(e) > 1e-12 else None,
                r_squared=1-np.sum(e**2)/np.sum((y-y.mean())**2), lag=lag,
                scope='Ex-post descriptive spanning regression; no causal trading control or search adjustment')


def source_hashes():
    lock = json.loads((ROOT/'latest/results/reference_networks/INPUT_CODE_LOCK.json').read_text())
    result = {}
    for name in ['layer.py', 'model.py']:
        path = OUT/'source_snapshot'/name
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        expected = [v for k, v in lock.items() if k.replace('\\', '/').endswith('/dynamic_hypergraph/src/framework/'+name)][0]
        # Preserve connector-delivered source and disclose newline-only differences.
        candidates = {hashlib.sha256(raw.replace(b'\r\n', b'\n').rstrip(b'\n')+end).hexdigest()
                      for end in [b'', b'\n', b'\n\n']}
        assert sha == expected or expected in candidates, (name, sha, expected)
        result[name] = dict(saved_sha256=sha, frozen_sha256=expected,
                            matches_ignoring_terminal_newlines=expected in candidates or sha == expected,
                            commit='9477001ad9b85beb3455234675535e26c3d33e44')
    return result


def synthetic_probes():
    from ndx_extended_hours import ExtendedModel
    import ndx_reference_models as reference
    torch.set_num_threads(2)
    torch.manual_seed(42)
    n = 12
    groups = [{i, (i+1) % n} for i in range(n)]
    parent = ExtendedModel(n, groups, 'after_hours_premarket').eval()
    # Identical histories and symmetric graph isolate node-specific mixer terms.
    x = (.2*torch.randn(1, 16, 1, 5)).expand(1, 16, n, 5).clone()
    mask = torch.ones(1, n)
    extra = torch.randn(1, n, 320)
    with torch.no_grad():
        core = parent.core.base(x, mask)
        total = parent(x, extra, mask)[0]
        extra_changed = parent(x, extra*100, mask)[0]
    assert float(torch.std(core)) < 1e-5
    assert float(torch.std(total)) > 1e-3
    assert torch.equal(total, extra_changed)  # Zero-initialized auxiliary head.

    path = OUT/'source_snapshot/layer.py'
    spec = importlib.util.spec_from_file_location('mechanism_pinned_dynamic_layers', path)
    layers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layers)
    cls = reference.definitions(OUT/'source_snapshot/model.py', ['Model'], vars(layers))['Model']
    reference.dynamic_class = lambda: cls  # Process-local injection; no frozen source/cache edits.
    snapshot = reference.wavelet_snapshot(groups, n)
    model = reference.make_reference('dynamic_restricted', n, snapshot).eval()
    features = .2*torch.randn(n, 16, 25)
    eligible = torch.ones(n, dtype=torch.bool)
    with torch.no_grad():
        author = model.network(features)
        adapted = model(features, None, eligible)[0]
        torch.testing.assert_close(author, adapted)
        eligible[-3:] = False
        bad = features.clone()
        bad[-3:] = float('nan')
        torch.testing.assert_close(model(features, None, eligible)[0], model(bad, None, eligible)[0])
        # Dynamic layer aggregation is a sum, not an edge-degree mean.
        embedding = torch.randn(n, 32)
        dynamic = model.network.dynamic_graph_layer
        _, incidence = dynamic(embedding)
        e1 = incidence.T@embedding
        _, incidence2 = dynamic(embedding.repeat(2, 1))
        e2 = incidence2.T@embedding.repeat(2, 1)
        torch.testing.assert_close(e2, 2*e1)
        torch.testing.assert_close(incidence.sum(1), torch.ones(n))
        feature_clone = features[:1].expand(n, -1, -1).clone()
        all_on = torch.ones(n, dtype=torch.bool)
        clone_score = model(feature_clone, None, all_on)[0]
        model.network.whconv1.diagonal_weight_filter.fill_(1.)
        model.network.whconv2.diagonal_weight_filter.fill_(1.)
        clone_score_uniform_filters = model(feature_clone, None, all_on)[0]
    # RMSE reweights each date's gradient even though its within-date direction matches MSE.
    s = torch.tensor([0., 0.], requires_grad=True)
    y = torch.tensor([1., 3.])
    mse = ((s-y)**2).mean()
    gm = torch.autograd.grad(mse, s, retain_graph=True)[0]
    gr = torch.autograd.grad(torch.sqrt(mse+1e-8), s)[0]
    torch.testing.assert_close(gr, gm/(2*torch.sqrt(mse.detach()+1e-8)))
    return dict(parent_identical_input_core_sd=float(core.std()),
                parent_identical_input_with_mixer_sd=float(total.std()),
                parent_auxiliary_zero_initialization_exact=True,
                dynamic_all_active_author_parity=True, dynamic_inactive_nan_invariance=True,
                dynamic_parameter_count=sum(p.numel() for p in model.parameters()),
                dynamic_identical_input_score_sd=float(clone_score.std()),
                dynamic_identical_input_uniform_filters_score_sd=float(clone_score_uniform_filters.std()),
                dynamic_edge_sum_doubles_when_population_duplicated=True,
                wavelet_times_inverse_max_identity_error=float((snapshot['wavelets']@snapshot['wavelets_inv']-torch.eye(n)).abs().max()),
                rmse_to_mse_gradient_ratio=float((gr/gm)[0]),
                scope='Random initialization and synthetic inputs only; representational capacity is not a learned financial mechanism')


def curve_diagnostics():
    frame = pd.read_csv(ROOT/'docs/ndx_drawdown_20261026/comparison_curves.csv', parse_dates=['date'])
    published = pd.read_csv(ROOT/'docs/ndx_drawdown_20261026/metrics.csv')
    benchmark = pd.read_csv(ROOT/'docs/ndx_extended_hours_20261024/curves.csv', parse_dates=['date'])
    paths = {}
    rows, regressions, annual = [], [], []
    for (recipe, group), p in frame.groupby(['recipe', 'group']):
        assert len(p) == 1003 and not p.date.duplicated().any()
        p = p.set_index('date').sort_index().net
        paths[recipe, group] = p
        m = metrics(p)
        old = published[(published.recipe == recipe)&(published.group == group)&(published.cost_multiplier == 1)].iloc[0]
        errors = {k: abs(m[k]-float(old[k])) for k in ['sharpe','annualized_return','annualized_volatility','maximum_drawdown']}
        assert max(errors.values()) < 1e-10
        rows.append(dict(recipe=recipe, group=group, **m, max_metric_error=max(errors.values())))
        for year, sub in p.groupby(p.index.year):
            annual.append(dict(recipe=recipe, group=group, year=int(year),
                               annual_log_growth=float(np.log1p(sub).sum()),
                               calendar_return=float(np.expm1(np.log1p(sub).sum())), **metrics(sub)))
    for ticker in ['SPY', 'QQQ']:
        b = benchmark[(benchmark.view == ticker)&(benchmark.cost_multiplier == 1)]
        # A benchmark appears once; never take arbitrary duplicate group rows.
        assert not b.date.duplicated().any() and len(b) == 1003
        paths[ticker, 'benchmark'] = b.set_index('date').sort_index().net
    for (recipe, group), p in paths.items():
        if group == 'benchmark':
            continue
        for name, base in [('QQQ', paths['QQQ','benchmark']),('original_full', paths['equal5/full',group])]:
            assert p.index.equals(base.index)
            regressions.append(dict(recipe=recipe, group=group, comparator=name, **regression(p, base)))
    pd.DataFrame(rows).to_csv(OUT/'RECOMPUTED_PATH_METRICS.csv', index=False)
    pd.DataFrame(regressions).to_csv(OUT/'DESCRIPTIVE_SPANNING.csv', index=False)
    pd.DataFrame(annual).to_csv(OUT/'ANNUAL_PATH_DIAGNOSTICS.csv', index=False)
    # Exact accounting metrics can be recomputed from paths; holdings/identity cannot.
    return dict(recomputed_paths=len(rows), sessions_per_path=1003,
                max_metric_error=max(r['max_metric_error'] for r in rows),
                benchmarks=['SPY','QQQ'], raw_inputs_available=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(scope='Exploratory context smoke tests requested by user; no financial training or protected-window scoring',
                  source_hashes=source_hashes(), synthetic=synthetic_probes(), curves=curve_diagnostics(),
                  availability={p: (ROOT/p).exists() for p in ['runs/ndx_reference_networks_20261024',
                    'runs/ndx_extended_hours_20261024','runs/ndx_drawdown_20261026/batch1',
                    'data/nasdaq_monthly_pit/assembled/manifest.json','.cache/architecture_references_20261024']})
    files = ['docs/ndx_drawdown_20261026/comparison_curves.csv','docs/ndx_drawdown_20261026/metrics.csv',
             'docs/ndx_extended_hours_20261024/curves.csv','latest/results/reference_networks/ranking.csv',
             'latest/results/reference_networks/complementarity.csv']
    result['input_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    (OUT/'SMOKE_RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
