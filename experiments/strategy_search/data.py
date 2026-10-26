"""Adapters for audited private caches; no downloads, neural fitting or sealed data."""
from pathlib import Path

import numpy as np

from .features import clocks, returns, peers, stock_volatility
from .storage import read, sha, atomic

SCOPES = ('earlier_OOF', 'initial_validation', 'later_development64')


class Study:
    def __init__(self, study_root, market_root, output):
        self.root = Path(study_root).resolve()
        self.market_root = Path(market_root).resolve()
        self.output = Path(output).resolve()
        self.loaded = {}

    def market(self, scope):
        if scope == SCOPES[2] and not (self.output / 'selection.json').exists():
            raise RuntimeError('Freeze development selection before later replay')
        if scope in self.loaded:
            return self.loaded[scope]
        root = self.root / 'cache' / scope
        meta = read(root / 'market.json')
        for name in ('open', 'close', 'sessions', 'actions', 'eligible', 'signal_index', 'sector', 'industry', 'completed'):
            if sha(root / (name + '.npy')) != meta['files'][name + '.npy']:
                raise ValueError('Changed market input: ' + name)
        m = {name: np.load(root / (name + '.npy'), mmap_mode='r') for name in
             ('open', 'close', 'sessions', 'actions', 'eligible', 'signal_index', 'sector', 'industry', 'completed')}
        raw, valid = returns(m['close'], m['sessions'])
        sector, sector_ok = peers(raw, valid, m['sector'])
        industry, industry_ok = peers(raw, valid, m['industry'])
        market, market_ok = peers(raw, valid, np.zeros(len(m['sector']), int))
        vol, vol_ok = stock_volatility(raw, valid, m['sessions'])
        eligible = np.zeros_like(valid)
        rows = np.flatnonzero(m['signal_index'] >= 0)
        assert np.array_equal(m['signal_index'][rows], np.arange(len(rows)))
        eligible[rows] = m['eligible']
        eligible &= valid & vol_ok
        m.update(meta=meta, raw=raw, valid=valid, residual=raw - .5 * sector - .5 * market,
                 residual_valid=valid & sector_ok & market_ok, spread=raw - industry,
                 spread_valid=valid & industry_ok, sector_peers=sector, industry_peers=industry,
                 sector_ok=sector_ok, industry_ok=industry_ok, vol=vol, eligible_full=eligible,
                 clock=clocks(m['sessions']), rows=rows)
        self.loaded[scope] = m
        return m

    def forecast(self, model, scope, seed='ensemble'):
        m = self.market(scope)
        suffix = 'ensemble' if seed == 'ensemble' else 'seed' + str(seed)
        path = self.root / 'cache' / scope / (model + '_' + suffix + '.npy')
        receipt = read(path.parent / 'forecasts.json')
        reference = next(x for x in receipt['files'] if Path(x['path']).name == path.name)
        assert sha(path) == reference['sha256']
        result = np.zeros_like(m['raw'])
        result[m['rows']] = np.load(path, mmap_mode='r')
        return result

    def volume(self, scope):
        m = self.market(scope)
        if 'volume' not in m:
            path = self.market_root / '1m' / 'bars.npy'
            bars = np.load(path, mmap_mode='r')[m['meta']['begin']:m['meta']['end']]
            assert bars.shape[:2] == m['close'].shape and bars.shape[2] == 7
            np.testing.assert_array_equal(bars[:, :, 3], m['close'])
            np.testing.assert_array_equal(bars[:, :, 0], m['open'])
            m['volume'] = np.asarray(bars[:, :, 4], float)
            m['typical'] = (np.asarray(bars[:, :, 1], float) + np.asarray(bars[:, :, 2], float) + m['close']) / 3
        return m['volume'], m['typical']

    def audit_volume(self):
        metadata = read(self.market_root / 'metadata.json')
        m = self.market(SCOPES[0])
        volume, typical = self.volume(SCOPES[0])
        train = np.isin(m['sessions'], m['meta']['session_ids'][:60])
        values = volume[train]
        passed = bool(metadata['adjustment'] == 'raw' and values.shape[1] == 250
                      and np.isfinite(values).all() and (values >= 0).all())
        receipt = dict(passed=passed, adjustment=metadata['adjustment'], column=4, name='raw share volume',
                       definition='data.archive.BAR_COLUMNS and experiments.market_ablation.dataset.aggregate',
                       available_after='completed bar close', price='(high + low + close)/3',
                       zero_volume_fraction=float((values == 0).mean()), fit_sessions=m['meta']['session_ids'][:60],
                       cached_open_close_match=True, source_metadata_sha256=sha(self.market_root / 'metadata.json'))
        atomic(self.output / 'volume_audit.json', receipt)
        return receipt
