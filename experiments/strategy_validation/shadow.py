"""After-close prospective replay. It never contacts a brokerage trading API."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

from experiments.strategy_search.features import returns, peers, stock_volatility, clocks
from experiments.strategy_search.signals import build
from experiments.strategy_search.storage import atomic, read, sha, arrays
from .market import order_windows
from .quotes import credentials, request, download


def initialize(reference, output, protocol):
    reservation_path = Path(output) / 'protected_sessions.json'
    expected = protocol['binding'].get('protected_sessions_sha256')
    if not reservation_path.exists() or expected is None or sha(reservation_path) != expected:
        raise ValueError('Forward collection requires a frozen --reservation-manifest audit, including explicit confirmation when no reservations exist')
    reservation = read(reservation_path)
    protected = sorted(set(reservation.get('reserved_session_dates', [])))
    if not protected and not reservation.get('verified_no_existing_reservation', False):
        raise ValueError('Reservation audit must list protected dates or explicitly confirm none exist')
    for day in protected:
        datetime.strptime(day, '%Y-%m-%d')
    path = Path(output) / 'shadow' / 'protocol.json'
    if path.exists():
        saved = read(path)
        if saved['parent_protocol_sha256'] != sha(Path(output) / 'protocol.json'):
            raise ValueError('Forward protocol mismatch')
        return saved
    frozen = datetime.fromisoformat(protocol['frozen_UTC'])
    # Start after the local trading date at freeze; never backdate a forward run.
    from zoneinfo import ZoneInfo
    first = frozen.astimezone(ZoneInfo('America/New_York')).date() + timedelta(days=1)
    if protected:
        first = max(first, datetime.strptime(protected[-1], '%Y-%m-%d').date() + timedelta(days=1))
    saved = dict(parent_protocol_sha256=sha(Path(output) / 'protocol.json'),
                 earliest_session=str(first), warmup_sessions=20, evaluation_sessions=60,
                 protected_sessions_sha256=expected, latest_protected_session=protected[-1] if protected else None,
                 execution='After-close delayed-SIP forward replay; no live or paper orders',
                 data_feed='sip', corporate_action_status='must verify separately for advancement',
                 frozen_strategy=protocol['binding']['config'], published_summary_checkpoints=[20, 40, 60])
    atomic(path, saved)
    return saved


def collect(reference, output, protocol, env_file, day):
    """Persist one newly observed exchange session, then rebuild its causal state."""
    import exchange_calendars as xcals
    import pandas as pd
    import json
    root = Path(output) / 'shadow'
    frozen = initialize(reference, output, protocol)
    if day < frozen['earliest_session']:
        raise ValueError('Requested date precedes prospective freeze')
    cal = xcals.get_calendar('XNYS')
    stamp = pd.Timestamp(day)
    if not cal.is_session(stamp):
        raise ValueError('Not an exchange session')
    opened, closed = cal.session_open(stamp), cal.session_close(stamp)
    if datetime.now(timezone.utc) < closed.to_pydatetime() + timedelta(minutes=20):
        raise ValueError('Wait until 20 minutes after the session close for delayed SIP data')
    expected = [str(x.date()) for x in cal.sessions_in_range(frozen['earliest_session'], day)]
    if len(expected) > 80:
        raise ValueError('The preregistered 20 warmup plus 60 evaluation sessions are complete')
    available = {p.stem for p in (root / 'sessions').glob('*.npz')}
    if any(d not in available for d in expected[:-1]):
        raise ValueError('Missing preceding forward session; collect chronologically without skipping failures')
    path = root / 'sessions' / (day + '.npz')
    receipt = path.with_suffix('.json')
    if path.exists():
        if not receipt.exists() or sha(path) != read(receipt)['sha256']:
            raise ValueError('Changed prospective input')
        return dict(status='already_collected', session=day, sessions=len(available))
    ref_protocol = read(Path(reference) / 'protocol.json')
    symbols = read(Path(ref_protocol['market_root']) / 'metadata.json')['symbols']
    headers = credentials(env_file)
    params = dict(symbols=','.join(symbols), start=opened.isoformat(), end=closed.isoformat(),
                  timeframe='1Min', adjustment='raw', feed='sip', limit=10000)
    observations = {s: [] for s in symbols}
    pages = 0
    while True:
        response = request('/v2/stocks/bars', params, headers)
        pages += 1
        for symbol, bars in response.get('bars', {}).items():
            observations[symbol].extend(bars)
        token = response.get('next_page_token')
        if not token:
            break
        params['page_token'] = token
    starts = pd.date_range(opened, closed, freq='min', inclusive='left').asi8
    bars = np.full((len(starts), len(symbols), 7), np.nan)
    lookup = {int(t): i for i, t in enumerate(starts)}
    for j, symbol in enumerate(symbols):
        for bar in observations[symbol]:
            i = lookup.get(int(pd.Timestamp(bar['t']).value))
            if i is not None:
                bars[i, j] = [bar[k] for k in ('o', 'h', 'l', 'c', 'v', 'n', 'vw')]
    actions = []
    for offset in range(0, len(symbols), 50):
        # The endpoint filters PROCESS dates, which can precede an ex-date.
        # Fetch its historical coverage before filtering actual event dates.
        query = dict(symbols=','.join(symbols[offset:offset + 50]), start='2016-01-01',
                     end=str(datetime.now(timezone.utc).date()), limit=1000, sort='asc', data_quality='all')
        while True:
            response = request('/v1/corporate-actions', query, headers)
            for kind, records in response.get('corporate_actions', {}).items():
                for record in records:
                    dates = [str(record.get(k, ''))[:10] for k in ('ex_date', 'effective_date')]
                    if day in dates:
                        actions.append(dict(type=kind, **record))
            token = response.get('next_page_token')
            if not token:
                break
            query['page_token'] = token
    # Opening after the action normally avoids overnight economics. Preserve
    # every event and conservatively exclude affected symbols for that session.
    excluded = sorted({a.get(k) for a in actions for k in ('symbol', 'source_symbol', 'target_symbol') if a.get(k) in symbols})
    arrays(path, bars=bars, starts=starts)
    atomic(root / 'sessions' / (day + '.actions.json'), dict(records=actions, excluded_symbols=excluded))
    atomic(receipt, dict(session=day, sha256=sha(path), feed='sip', pages=pages,
                         retrieved_UTC=datetime.now(timezone.utc).isoformat(),
                         missing_fraction=float((~np.isfinite(bars[:, :, 0])).mean()),
                         corporate_action_status='verified_API_events_excluded', actions_sha256=sha(root / 'sessions' / (day + '.actions.json')),
                         benchmark='delayed after-close observations'))
    count = len(available) + 1
    return dict(status='collected', session=day, sessions=count,
                warmup_remaining=max(0, 20 - count), evaluated_sessions=max(0, count - 20))


def replay_forward(reference, output, protocol, env_file):
    """Use the exact fitted signal construction and the practical replay engine."""
    from .config import Execution
    from .engine import replay, PATH_COLUMNS, ORDER_COLUMNS
    from .analysis import dollar_daily, normalized_summary
    from experiments.strategy_search.statistics import scales
    root = Path(output) / 'shadow'
    inputs = sorted((root / 'sessions').glob('*.npz'))
    if not inputs:
        return dict(status='awaiting_first_session', evaluated_sessions=0)
    config = protocol['binding']['config']
    ref = read(Path(reference) / 'protocol.json')
    from .market import load
    old, _, _ = load(reference, 'later_development64')
    collected, times, sessions, exclusions = [], [], [], []
    for sid, p in enumerate(inputs):
        if sha(p) != read(p.with_suffix('.json'))['sha256']:
            raise ValueError('Changed forward observations')
        with np.load(p) as z:
            collected.append(z['bars']); times.append(z['starts']); sessions.extend([sid] * len(z['starts']))
            exclusions.append(np.zeros((len(z['starts']), len(old['symbols'])), bool))
        action_path = p.with_suffix('.actions.json')
        if sha(action_path) != read(p.with_suffix('.json'))['actions_sha256']:
            raise ValueError('Changed forward corporate-action receipt')
        for symbol in read(action_path)['excluded_symbols']:
            exclusions[-1][:, old['symbols'].index(symbol)] = True
    bars = np.concatenate(collected)
    session = np.asarray(sessions, np.int64)
    raw, valid = returns(bars[:, :, 3], session)
    sector, sok = peers(raw, valid, old['sector'])
    market, mok = peers(raw, valid, np.zeros(len(old['sector']), int))
    vol, vok = stock_volatility(raw, valid, session)
    m = dict(open=bars[:, :, 0], close=bars[:, :, 3], volume=bars[:, :, 4], sessions=session,
             starts=np.concatenate(times), actions=np.zeros_like(valid), sector=old['sector'], industry=old['industry'],
             raw=raw, valid=valid, residual=raw - .5 * sector - .5 * market,
             residual_valid=valid & sok & mok, vol=vol, eligible_full=valid & vok & ~np.concatenate(exclusions), clock=clocks(session),
             symbols=old['symbols'], meta={'session_ids': list(range(len(inputs)))},
             evaluate_ids=list(range(20, min(80, len(inputs)))), scope='forward')
    with np.load(Path(reference) / 'fit.npz') as z:
        fitted = {k: z[k] for k in z.files}
    class Adapter:
        def market(self, scope):
            return m
    base, _ = build(Adapter(), config, 'forward', fitted)
    quotes = download(m, order_windows(m, base), root, env_file)
    settings = Execution()
    shadow, _, _ = replay(m, base, np.ones(len(inputs)), settings, quotes)
    scale = scales(dollar_daily(shadow, session, settings.capital))
    path, orders, failures = replay(m, base, scale, settings, quotes)
    arrays(root / 'replay.npz', equity=path, orders=orders, scales=scale, sessions=session)
    result = dict(status='warmup' if len(inputs) <= 20 else 'evaluating', collected_sessions=len(inputs),
                  evaluated_sessions=max(0, min(60, len(inputs) - 20)), failures=failures,
                  corporate_actions_verified=True, borrow_availability_verified=False, data_status='prospective delayed-SIP replay',
                  columns=PATH_COLUMNS, order_columns=ORDER_COLUMNS, source_protocol_sha256=sha(Path(output) / 'protocol.json'))
    if len(inputs) > 20:
        result['metrics'], _ = normalized_summary(m, path, orders, failures, settings.capital)
    if len(inputs) >= 80:
        metrics = result['metrics']
        result['status'] = 'completed_forward_research_candidate' if (metrics['accounting_valid'] and metrics['quote_coverage'] >= .99 and metrics['Sharpe'] is not None and metrics['Sharpe'] >= 1 and metrics['total_return'] > 0) else 'completed_forward_failed_or_inconclusive'
    atomic(root / 'status.json', result)
    return result
