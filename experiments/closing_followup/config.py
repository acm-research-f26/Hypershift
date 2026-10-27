"""Research choices frozen before fitting or comparing outcomes."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from experiments.strategy_search.storage import atomic, read, sha
from experiments.strategy_validation.config import Execution, SOURCES

SCOPES = ('earlier_OOF', 'initial_validation', 'later_development64')
ETFS = ('SPY', 'QQQ', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY')
CATALOG = [
    dict(id='control', family='control', description='Original S10 event targets, corrected balanced execution'),
    *[dict(id=f'concentrated_{k}_{gate}', family='concentrated', names_per_side=k,
           cost_gate=gate == 'cost', description=f'Top {k} each side at one closing-window decision; {gate}')
      for k in (3, 5) for gate in ('plain', 'cost')],
    dict(id='participation_volume', family='participation', rvol=1.25, breadth=0.,
         description='One closing-window decision; early relative-volume >=1.25 and |early score| >=1'),
    dict(id='participation_confirmed', family='participation', rvol=1.25, breadth=.65,
         description='Volume filter plus 65% of sector peers retain their early raw-return sign'),
    dict(id='etf_market_pair', family='etf', kind='market_pair', description='SPY/QQQ early relative-momentum pair'),
    dict(id='etf_sector_pairs', family='etf', kind='sector_pairs', description='Two strongest and weakest sector ETFs'),
]


def scenarios():
    s = Execution()
    return {
        'primary': ('quotes', s),
        'commission_free': ('quotes', replace(s, commission_free=True)),
        'adverse_0p5': ('quotes', replace(s, adverse_bps=.5)),
        'zero_explicit_fees': ('quotes', replace(s, commission_free=True, exchange_taker_share=0.,
            clearing_share=0., cat_share=0., sec_sell_fraction=0., finra_sell_share=0.)),
        'zero_execution_cost_proxy': ('bar_proxy', replace(s, commission_free=True,
            exchange_taker_share=0., clearing_share=0., cat_share=0., sec_sell_fraction=0., finra_sell_share=0.)),
    }


def freeze(reference, prior, output):
    reference, prior, output = map(lambda x: Path(x).resolve(), (reference, prior, output))
    output.mkdir(parents=True, exist_ok=True)
    protected = read(prior / 'protected_sessions.json')
    binding = dict(catalog=CATALOG, etfs=ETFS, reference=str(reference), prior=str(prior),
        inputs={n: sha(reference / n) for n in ('protocol.json', 'selection.json', 'fit.npz')},
        protected_sha256=sha(prior / 'protected_sessions.json'),
        reserved_session_dates=protected['reserved_session_dates'],
        scenarios={k: dict(mode=m, settings=asdict(s)) for k, (m, s) in scenarios().items()},
        partitions=read(reference / 'protocol.json')['partitions'],
        fit='First 60 earlier sessions only; all training performance excluded from ranking',
        selection='Five earlier/initial development blocks only; later cohort replay after frozen selection',
        later_status='Previously research-exposed; not an untouched test set',
        stock_liquidity='Training median closing-window dollar volume, top half of stocks',
        concentration='One decision at last-30-minute bar close; strongest original S10 names; no reranking',
        sizing='Inverse trailing-60-minute volatility, <=4% per name, gross<=90%, sector net<=4%; cash if infeasible',
        cost_gate='Pooled training-only signed sector-relative closing-return regression on absolute early score; predicted edge exceeds observed full spread + explicit round-trip fees + 1bp',
        participation='First-hour cumulative volume divided by training median first-hour volume; >=1.25; absolute early standardized sector-relative return>=1; optional sector peer sign persistence>=65%',
        etf_signals='First-hour log return; closing-window return must retain same relative sign; paired dollar-neutral 4% sleeves; no market-direction exposure',
        execution='Whole shares; original delay; paired depth/previous-volume-constrained fills; pending entry canceled after 3 attempts; five closing attempts; never synthesize missing fills',
        operating_net_band=.001, hard_net_band=.0025,
        coverage='Original >=99% attempted-notional fresh-quote coverage retained; first-parent coverage also reported',
        risk_scaling='Prior-20-session shadow volatility target 10% annualized capped at 1; first20 sessions cash',
        qualification='Valid accounting, >=99% quote coverage, positive both development cohorts, >=20 active development days, median of five development block Sharpes>=1; frozen later replay SR>=1 and positive; +0.5bp stress positive in all cohorts',
        research_trial_count=len(CATALOG)*len(scenarios())*len(SCOPES),
        sources=SOURCES, source={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        dependencies={str(p.relative_to(Path(__file__).parents[1])): sha(p)
          for folder in ('strategy_search', 'strategy_validation')
          for p in sorted((Path(__file__).parents[1]/folder).glob('*.py'))})
    path = output / 'protocol.json'
    if path.exists():
        saved = read(path)
        if saved['binding'] != __import__('json').loads(__import__('json').dumps(binding)):
            raise ValueError('Frozen follow-up protocol/source changed; choose a new output directory')
        return saved
    saved = dict(frozen_UTC=datetime.now(timezone.utc).isoformat(), binding=binding)
    atomic(path, saved)
    return saved
