"""Immutable protocol and explicit conditional-stage prerequisites."""
from dataclasses import asdict
from pathlib import Path
import platform
import sys

from experiments.strategy_search.storage import atomic, read, sha, utc
from experiments.strategy_validation.config import Execution, SOURCES

STUDY = Path(r'C:\Users\ethan\OneDrive\Pictures\Documents\Hypershift Papers\trading_strategy_optimization_2026-10-26')
REFERENCE = STUDY / 'next_strategy_round' / 'execution_v3'
PRIOR = STUDY / 'closing_momentum_validation_2026-10-26' / 'validated'
ETF_ROOT = STUDY / 'closing_momentum_followup_2026-10-26'
SCOPES = ('earlier_OOF', 'initial_validation', 'later_development64')
COSTS = (0., .5, 1., 2., 5.)
CATALOG = [
    dict(id='D01', design='D01', arm='graph', horizon=30),
    dict(id='D01_partner', design='D01', arm='partner', horizon=30),
    dict(id='D01_self', design='D01', arm='self', horizon=30),
    dict(id='D01_no_graph', design='D01', arm='no_graph', horizon=30),
    dict(id='D03', design='D03', arm='stall', horizon=30),
    dict(id='D03_no_stall', design='D03', arm='no_stall', horizon=30),
    dict(id='D03_own', design='D03', arm='own', horizon=30),
    dict(id='D03_momentum', design='D03', arm='momentum', horizon=30),
    dict(id='D04', design='D04', arm='etf', horizon=15),
    dict(id='D04_stock_hedge', design='D04', arm='stock_hedge', horizon=15),
    dict(id='D04_etf_only', design='D04', arm='etf_only', horizon=15),
    dict(id='D04_stock_only', design='D04', arm='stock_only', horizon=15),
]
GATES = ('net_edge', 'no_gate', 'fixed_score', 'matched_random')


def freeze(output):
    import numpy
    import numba
    output = Path(output)
    protected = read(PRIOR / 'protected_sessions.json')
    reference = read(REFERENCE / 'protocol.json')
    source = Path(__file__).parent
    binding = dict(version='decision_v5', source={p.name: sha(p) for p in sorted(source.glob('*.py'))},
        dependencies={str(p.relative_to(source.parents[1])): sha(p) for folder in
            ('strategy_search', 'strategy_validation', 'closing_followup', 'longer_horizon')
            for p in sorted((source.parent / folder).glob('*.py'))},
        environment=dict(python=sys.version, numpy=numpy.__version__, numba=numba.__version__, platform=platform.platform()),
        lock_sha256=sha(source.parents[1] / 'uv.lock'),
        design_sha256=sha(STUDY / 'decision_strategy_designs_2026-10-27' / 'decision_designs.json'),
        classifications_sha256=sha(output/'causal_classifications.json'),
        classifications='241 pre-2025 archival business industry groups plus9 pre-2025 primary-business reconstructions; coarser than prior42industries, no later classification values. Freeze before outcomes.',
        prompt_sha256=sha(STUDY / 'decision_strategy_designs_2026-10-27' / 'GOAL_PROMPT.txt'),
        protected_sha256=sha(PRIOR / 'protected_sessions.json'), protected=protected,
        partitions=reference['partitions'], market_root=reference['market_root'], catalog=CATALOG,
        costs_bps=COSTS, gates=GATES, fees=asdict(Execution()), fee_sources=SOURCES,
        fee_evidence='IBKR official commissions page checked 2026-10-27; inherited 2026-10-26 schedule. Taker route is a scenario, not an invoice.',
        universe='All 250 stocks with causal masks; D04 adds XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY only as explicit hedges.',
        fit='Expanding train sessions 0:20 ->20:30, 0:30 ->30:40, 0:40 ->40:50, 0:50 ->50:60. Intra-session samples require label end before session close; session folds purge entire boundary holding horizon plus two-minute delay. Refit on first60 only.',
        graph='Within-industry directed lag5 correlation, positive in both chronological training halves, top3 per follower. Best stable leader per follower; no enlarged path search.',
        model='Pooled ridge 1e-3 on standardized own5,own15,leader5,relative15,vol60,breadth. Features are log returns; same-horizon delayed ARITHMETIC relative returns label dollar edge. No neural training. D03 own control uses own90th-percentile event; momentum control uses industry-only returns/breadth in the common hedged book.',
        calibration='OOF signed score deciles: mean realized signed relative return and standard error of session means; at least5 distinct sessions per bin, pooled OOF fallback. Hurdle is heuristic, not coverage.',
        matched_random='Fixed seed20261027; train-only economic-gate admission frequency on top3 OOF candidates at each clock and same whole-share costs. Report realized count/risk differences; paired activity/risk resampling is analysis, not a new policy.',
        entry='Completed minute clocks60..210 every5; per-stock/day one entry; rank expected net dollars; at most3 concurrent baskets. One recipe each; no additional recipe.',
        sizing='min(4%,0.25%/trailing60-minute annualized-to-daily volatility); whole shares; hedge total equal dollars within $100 at $100k; operating gross30%,sector net3%,instrument4%; hard gross100%,sector5%,instrument5%,net0.25%. Dollar neutral is not beta neutral.',
        execution='Primary inherited delay2: origin bar completion precedes arrival by one full minute (origin index+2 open). Sequential individual fills checked against hard risk; parent fees aggregate partials, depth never replenishes at identical quote timestamp. Reject infeasible integer baskets. Mandatory exits retry every minute and last5 closing minutes. Risk is observed at each fill and minute open/close; intraminute unseen excursions cannot be certified.',
        no_trade='Fixed $50 discrepancy band for discretionary rebalancing; fixed entry quantities; required exits/risk reductions bypass it. Expire partial entry after3 arrival attempts and flatten residual basket.',
        costs='Round-trip dollars sum every entry/hedge/exit fee, causal prior exit spread, current spread,3% borrow and0.5bp per side buffer. Proxy gates substitute train-only spread estimates and are explicitly not quote execution.',
        activity=dict(per_development_block_roundtrips=10,development_roundtrips=100,development_active_sessions=40,later_active_sessions=20),
        stages={
            'D09': 'First; fixtures and unchanged streams, old engine comparison; then3 primary price-only candidate streams on train60 x normal,long-first,short-first,hedge-unavailable, before any development run. Retain adverse sequencing failures.',
            'D08': 'Second; >=20 OOF sessions and >=1000 labeled training examples per base recipe; otherwise cash with unreliable calibration disposition.',
            'D01_D03_D04': 'First wave unconditional implementation, primary plus3 stated controls; D08 extra arms only on primary recipes.',
            'quotes': 'One diagnostic per design: highest five-block median1bp among accounting-valid arms with >=10 roundtrips each block,>=100 total,>=40 active; no profit requirement for diagnostic. Freeze before later. Replay primary,commission_free,+0.5bp,long-first,short-first,hedge-unavailable and age60 sensitivity; all unscaled/scaled requests must close.',
            'D06': 'At least one primary D01/D03 with development accounting/risk validity, required development activity,>=99% primary fresh quote intended-notional coverage and100% intended-query completeness. Full audited ensembles only. Same base sizing/exits; train median signed confirmation, LN4, fixed-seed shuffled signs and activity/risk matching.',
            'D02_D05': 'D01 primary quote-valid with required activity,positive both development cohorts,positive median paired improvement over self/no_graph/partner controls,>=3/4 positive OOF fold improvements. D02 additionally >=30 stable pure relay paths in train60 and>=100 OOF relay candidates. D05 exactly clock60..105,110..155,160..210,shrink bucket correction n/(n+1000).',
            'D07': 'Valid quote base with required activity plus >=20 distinct OOF sessions and>=100 OOF trades in each fixed reversal/momentum action. Depth2 maximum, fixed volatility/breadth/correlation/spread features; no cohort labels.',
            'D10': 'Valid quote base with required activity plus>=100 OOF quote-observed trades over>=20 sessions and positive OOF net edge at arrival; chronological wait maximum2 minutes, required exits immediate.',
        },
        qualification='Valid accounting AND all fill/open/close risk,positive both development cohorts,median5block daily Sharpe>=1,required activity,>=99% primary quote coverage,complete queries;later positive and Sharpe>=1,>=20 active;positive+0.5bp stress. Historical only, never deployment.',
        budgets=dict(first_wave_configs=12,primary_extra_gates=9,proxy_trials=315,additional_recipes=0,
                     quote_diagnostics_max=3,bootstrap_repetitions=2000,cpu_workers=2,disk_reserve_GB=10),
        future='Never open reserved outcomes. Prospective test starts after source freeze and after2027-01-13; earliest currently2027-01-14. No collection scheduled.',
        prior='210 main proxies and30 closing benchmarks;0 qualified,empty shortlist,ZERO new quote replays. Empty quote receipts do not establish coverage. S10/prior graph rules price-only.')
    binding['correctness_version']='v1 NPZ writer and ineligible zero-vol quantile bugs;v2 keeps90th percentile on eligible windows,retains5 cash trials and failed active-control tape;v3 repairs1.57509613-dollar idle-mark mismatch and retains30 trials before discovering expired empty slots. V4 was frozen but never experimentally run; it releases slots,implements the50-dollar discrepancy band,proves a sequential integer net path,and sizes costs within known volume/depth. V5 pre-run conformance audit replaces log P&L labels with arithmetic dollar labels,uses actual residual half/1.5x exits for D03,corrects own/industry-control feature restrictions,and makes graph top3 positive as declared. Risk,activity,qualification and90th-percentile rules unchanged. Train-only fits refit; earlier versions retained,none selected a cross-recipe shortlist.'
    # JSON normalizes tuples before immutable equality comparison.
    import json
    binding = json.loads(json.dumps(binding))
    dest = output / 'protocol.json'
    if dest.exists():
        old = read(dest)
        if old['binding'] != binding:
            raise ValueError('Protocol/source changed; preserve originals and use a new version directory')
        return old
    result = dict(frozen_UTC=utc(), binding=binding)
    atomic(dest, result)
    import shutil
    snapshot=output/'source_v5';snapshot.mkdir(parents=True,exist_ok=True)
    for p in sorted(source.glob('*.py')):shutil.copyfile(p,snapshot/p.name)
    test=source.parents[1]/'test'/'test_decision_strategy.py'
    shutil.copyfile(test,snapshot/test.name)
    shutil.copyfile(PRIOR/'protected_sessions.json',output/'protected_sessions.json')
    atomic(output/'source_freeze_receipt_v5.json',dict(frozen_UTC=result['frozen_UTC'],protocol_sha256=sha(dest),
        files={p.name:sha(p) for p in sorted(snapshot.iterdir())},protected_sha256=sha(output/'protected_sessions.json')))
    atomic(output / 'progress.json', dict(stage='protocol_frozen', updated_UTC=utc(), completed=[], pending=['D09','D08','first_wave','conditional_gates','delivery']))
    return result
