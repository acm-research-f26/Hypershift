"""Read audited bars and fit artifacts without accessing sealed outcomes."""
from pathlib import Path
import json
import numpy as np

from experiments.strategy_search.data import Study, SCOPES
from experiments.strategy_search.signals import build
from experiments.strategy_search.storage import read, sha


def load(reference, scope):
    reference = Path(reference)
    protocol = read(reference / 'protocol.json')
    config = next(x['config'] for x in read(reference / 'selection.json')['leaders']
                  if x['config']['family'] == 'S10' and x['config']['style'] == 'long_short')
    study = Study(protocol['study_root'], protocol['market_root'], reference)
    m = study.market(scope)
    with np.load(reference / 'fit.npz') as z:
        fitted = {k: z[k] for k in z.files}
    base, diagnostics = build(study, config, scope, fitted)
    volume, _ = study.volume(scope)
    m['volume'] = volume
    m['starts'] = np.asarray(m['completed'], np.int64) - 60_000_000_000
    metadata = read(Path(protocol['market_root']) / 'metadata.json')
    m['symbols'] = metadata['symbols']
    m['scope'] = scope
    m['evaluate_ids'] = m['meta']['session_ids'][60:] if scope == SCOPES[0] else m['meta']['session_ids']
    return m, base, diagnostics


def order_windows(m, base):
    """All possible entry/retry/closing times, not just successfully filled orders."""
    windows = []
    for sid in m['meta']['session_ids']:
        ix = np.flatnonzero(m['sessions'] == sid)
        stocks = np.flatnonzero(np.any(base[ix] != 0, axis=0))
        if not len(stocks):
            continue
        for i in ix[max(0, len(ix) - 35):]:
            windows.append((int(i), stocks.tolist()))
    return windows
