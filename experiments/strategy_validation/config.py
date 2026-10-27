"""Preregistered execution assumptions, kept separate from the signal catalog."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

from experiments.strategy_search.storage import atomic, read, sha


@dataclass(frozen=True)
class Execution:
    capital: float = 100_000.
    delay: int = 2
    borrow: float = .03
    adverse_bps: float = 0.
    commission_free: bool = False
    participation: float = .01
    net_tolerance: float = .0025
    minimum_order_weight: float = .005
    quote_max_age_seconds: float = 5.
    # Current published cost schedule is a counterfactual applied consistently
    # across all historical periods, not a claim about historical invoices.
    commission_share: float = .0035
    commission_minimum: float = .35
    commission_maximum_fraction: float = .01
    fractional_minimum: float = .01
    clearing_share: float = .0002
    cat_share: float = .000003
    sec_sell_fraction: float = .0000206
    finra_sell_share: float = 0.
    finra_sell_cap: float = 9.79
    commission_pass_through: float = .000735
    exchange_taker_share: float = .003

    def __post_init__(self):
        if self.capital <= 0 or self.delay not in (2, 3, 5):
            raise ValueError('Positive capital and delay 2, 3 or 5 required')
        if not 0 < self.participation <= 1 or self.borrow < 0 or self.adverse_bps < 0:
            raise ValueError('Invalid execution assumptions')


SOURCES = {
    'commission': 'https://www.interactivebrokers.com/en/pricing/commissions-stocks.php',
    'SEC': 'https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2',
    'FINRA_TAF_pause_2026_Q4': 'https://www.sec.gov/rules-regulations/self-regulatory-organization-rulemaking/sr-finra-2026-021',
    'quote_sizes': 'https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change',
    'SIP_access': 'https://docs.alpaca.markets/us/docs/market-data-faq',
    'paper_limits': 'https://docs.alpaca.markets/us/docs/paper-trading',
}


def freeze(reference, output):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if reference == output or reference in output.parents:
        raise ValueError('Use a separate sibling validation directory')
    selection = read(reference / 'selection.json')
    config = next(x['config'] for x in selection['leaders']
                  if x['config']['family'] == 'S10' and x['config']['style'] == 'long_short')
    source = Path(__file__).parent
    binding = {p.name: sha(p) for p in sorted(source.glob('*.py'))}
    immutable = dict(config=config, reference=str(reference),
                     reference_files={n: sha(reference / n) for n in ('protocol.json', 'selection.json', 'fit.npz')},
                     validation_source=binding, assumptions=asdict(Execution()), fee_sources=SOURCES,
                     protected_sessions_sha256=sha(output / 'protected_sessions.json') if (output / 'protected_sessions.json').exists() else None,
                     strategy_dependencies={p.name: sha(p) for p in sorted((source.parent / 'strategy_search').glob('*.py'))},
                     accounts=[25_000., 100_000., 1_000_000.], delays=[2, 3, 5], stresses_bps=[0., .5, 1.],
                     warmup_sessions=20, forward_evaluation_sessions=60,
                     quote_coverage_required=.99, historical_data_status='research-exposed',
                     borrow_availability='assumed; historical locates unavailable',
                     exchange_fee='0.003 USD/share taker scenario; venue/rebate invoices unknown',
                     fee_schedule='2026-10-26 published snapshot applied counterfactually to every cohort',
                     policy='One fixed practical implementation; no parameter search',
                     share_size_units='Alpaca quote date before 2025-11-03: lots x100; thereafter shares')
    path = output / 'protocol.json'
    if path.exists():
        saved = read(path)
        if saved['binding'] != immutable:
            raise ValueError('Frozen protocol/source changed; use a new output directory')
        return saved
    output.mkdir(parents=True, exist_ok=True)
    saved = dict(frozen_UTC=datetime.now(timezone.utc).isoformat(), binding=immutable)
    atomic(path, saved)
    return saved
