from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import json
from experiments.strategy_search.storage import read,sha,atomic
from experiments.strategy_validation.config import Execution,SOURCES

SCOPES=('earlier_OOF','initial_validation','later_development64')
FAMILIES=('direct','two_hop','industry','sector','self_reversal','self_momentum','shuffled_two_hop')
CATALOG=[dict(id=f'{family}_{hold}',family=family,hold=hold) for family in FAMILIES for hold in (60,120)]
FEES=(0.,.5,1.,2.,5.)


def proxy_settings(bps):
    return Execution(adverse_bps=bps,commission_free=True,exchange_taker_share=0.,clearing_share=0.,
                     cat_share=0.,sec_sell_fraction=0.,finra_sell_share=0.)


def quote_scenarios():
    s=Execution()
    return [('primary',s),('commission_free',replace(s,commission_free=True)),
            ('adverse_0p5',replace(s,adverse_bps=.5)),('delay5',replace(s,delay=5)),
            ('age60_sensitivity',replace(s,quote_max_age_seconds=60.))]


def freeze(reference,prior,graph,output):
    reference,prior,graph,output=map(lambda p:Path(p).resolve(),(reference,prior,graph,output));output.mkdir(parents=True,exist_ok=True)
    old=read(reference/'protocol.json');protected=read(prior/'protected_sessions.json')
    binding=dict(reference=str(reference),prior=str(prior),graph=str(graph),catalog=CATALOG,proxy_costs_bps=FEES,
        reference_files={n:sha(reference/n) for n in ['protocol.json','selection.json','fit.npz']},
        graph_files={n:sha(graph/n) for n in ['networks.npz','network_fit.json']},
        protected_sha256=sha(prior/'protected_sessions.json'),reserved_session_dates=protected['reserved_session_dates'],
        partitions=old['partitions'],universe='All 250 stocks; causal eligibility; no liquidity-half exclusion',
        features='Completed five-minute market-residual returns standardized using first60-session SD; W[target,source] propagated leader shocks; industry/sector leave-one-out peers',
        graph_policy='Reuse train-only stable-positive top3 lag5 networks and pure two-hop network; shuffled topology is a fixed null control, not causal evidence',
        entries='Scan completed closes every15 minutes from clock60 to clock210; require abs(leader)>=2 and signed follower response<25% of leader; first jointly feasible long/short event per day; at most top3 per side, one entry per stock/day',
        controls='Self reversal/momentum use abs(own standardized five-minute residual)>=2 at the same scan clocks',
        sizing='Inverse trailing60-minute volatility: min(4%,0.005/(minuteSD*sqrt390)); project operating gross90%, stock4%, sector net4%, net band0.1%; whole shares',
        holding='Scheduled signal exit after60/120 minutes; mandatory exit retries never expire; hard session flattening in last5 minutes; no overnight exposure',
        fees=asdict(Execution()),sources=SOURCES,
        scaling='Causal20-session10%-annual-vol target cap1; first20 sessions cash; train60 excluded from evaluation',
        screening='All14 policies at five flat per-side cost scenarios on earlier/initial development. Two diagnostic quote candidates: best graph-family and best self-control by five-block median SR at1bp, with all blocks defined, positive both cohorts, accounting valid and >=20 active development days; fallback highest valid median with qualification=False. No later selection.',
        qualification='Graph candidate requires development median SR>=1 at1bp and positive both cohorts; later SR>=1 and positive at1bp. Quote validation also requires valid accounting, >=99% five-second fresh quote coverage, positive all cohorts and +0.5bp stress positive. A60-second-age scenario is sensitivity only.',
        quote_policy='Cache as-of SIP snapshots; five-second query then60-second fallback only for missing symbols; primary freshness stays5seconds. Iteratively fetch every intended order timestamp from unscaled/scaled executions of all frozen quote scenarios until no uncached requests remain. Never infer a fill from an unqueried window.',
        auction_policy='Availability and closing-price benchmark only; observed closing auction price is an outcome, never a signal. No auction imbalance data inferred from bars. Two prespecified closing baskets: early relative momentum and last60-minute relative momentum. Benchmark entry minute open, exit largest reported closing trade (condition6), whole shares, full-fill assumption. No auction diagnostic can qualify as execution validated.',
        data_status='All historical cohorts were research-exposed; reserved future sessions never accessed',
        source={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        dependencies={str(p.relative_to(Path(__file__).parents[1])):sha(p) for folder in ['closing_followup','strategy_search','strategy_validation'] for p in sorted((Path(__file__).parents[1]/folder).glob('*.py'))})
    binding=json.loads(json.dumps(binding));dest=output/'protocol.json'
    if dest.exists():
        d=read(dest)
        if d['binding']!=binding:raise ValueError('Frozen protocol/source changed; use a new output directory')
        return d
    d=dict(frozen_UTC=datetime.now(timezone.utc).isoformat(),binding=binding);atomic(dest,d);return d
