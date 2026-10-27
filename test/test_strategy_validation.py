"""Cash, timing, quote causality and operational invariants."""
from dataclasses import replace
import numpy as np
import pytest

pytest.importorskip('numba')
from experiments.strategy_validation.config import Execution
from experiments.strategy_validation.fees import charges, cumulative_charge_delta
from experiments.strategy_validation.quotes import normalized_quote, QUOTE_DTYPE
from experiments.strategy_validation.engine import replay, feasible_round, limits
from experiments.strategy_validation.analysis import normalized_summary, dollar_daily, gate
from experiments.strategy_validation.baseline import replay_with_orders
from experiments.strategy_search.execution import replay as original_replay


def market():
    t, n = 45, 4
    price = np.full((t, n), 100.)
    start = np.datetime64('2026-09-25T19:15', 'ns').astype(np.int64)
    m = dict(open=price.copy(), close=price.copy(), volume=np.full((t, n), 1e6),
             sessions=np.zeros(t, np.int64), starts=start + np.arange(t) * 60_000_000_000,
             actions=np.zeros((t, n), bool), sector=np.zeros(n, np.int64),
             meta={'session_ids': [0]}, evaluate_ids=[0])
    base = np.zeros_like(price); base[15:, :2] = [.04, -.04]
    quotes = np.zeros((t, n), dtype=QUOTE_DTYPE)
    quotes['bid'] = 99.99; quotes['ask'] = 100.01
    quotes['bid_size'] = quotes['ask_size'] = 1000
    quotes['timestamp'] = m['starts'][:, None] - 1_000_000
    return m, base, quotes


def test_minimum_and_fractional_fees_and_sell_fees():
    s = Execution(exchange_taker_share=0.)
    assert charges(10, 100, s)[0] == .35
    assert charges(1000, 100, s)[0] == 3.5
    assert charges(.00001, 100, s, fractional=True)[0] == .01
    assert charges(-10, 100, s)[3] > charges(10, 100, s)[3]


def test_partial_fills_pay_one_parent_minimum():
    s = Execution()
    a = cumulative_charge_delta(0, 0, 10, 1000, 1, s)
    b = cumulative_charge_delta(10, 1000, 10, 1000, 1, s)
    np.testing.assert_allclose(np.array(a) + b, charges(20, 100, s))
    assert b[0] == 0


def test_quote_size_change_and_future_quote_timestamp_preserved():
    q = dict(bp=100, ap=101, bs=2, **{'as': 3}, t='2025-10-31T19:00:00Z')
    assert normalized_quote(q, 0)[2:4] == (200, 300)
    q['t'] = '2025-11-03T19:00:00Z'
    assert normalized_quote(q, 0)[2:4] == (2, 3)


def test_whole_shares_delay_bid_ask_and_no_overnight():
    m, base, quotes = market()
    path, orders, failures = replay(m, base, np.ones(1), Execution(), quotes)
    fills = orders[orders[:, 5] != 0]
    assert fills[:, 0].min() == 17
    assert np.all(fills[:, 5] == np.trunc(fills[:, 5]))
    assert np.all(fills[fills[:, 5] > 0, 7] == 100.01)
    assert np.all(fills[fills[:, 5] < 0, 7] == 99.99)
    assert path[-1, 3] == 0 and failures == 0
    assert path[-1, 1] < 100000
    metrics, daily = normalized_summary(m, path, orders, failures, 100000.)
    assert metrics['total_return'] == pytest.approx(path[-1, 1] / 100000 - 1)
    assert daily[0, 4] == pytest.approx(path[:, 8:12].sum())


def test_future_quotes_and_missing_quotes_cannot_fill():
    m, base, quotes = market()
    quotes['timestamp'] = m['starts'][:, None] + 1
    path, orders, failures = replay(m, base, np.ones(1), Execution(), quotes)
    assert not orders[:, 5].any()
    assert path[-1, 1] == 100000
    assert not orders[:, 9].any()


def test_partial_fills_respect_depth_and_retry_without_duplicate_minimums():
    m, base, quotes = market()
    quotes['bid_size'] = quotes['ask_size'] = 3
    path, orders, failures = replay(m, base, np.ones(1), Execution(), quotes)
    filled = orders[orders[:, 5] != 0]
    assert abs(filled[:, 5]).max() <= 3
    first = filled[filled[:, 2] == filled[0, 2]]
    assert first[:, 10].sum() <= .35 + 1e-10
    assert len(first) > 1
    assert failures > 0  # insufficient closing depth leaves a disclosed residual


def test_same_quote_displayed_depth_cannot_be_reused():
    m, base, quotes = market()
    quotes['bid_size'] = quotes['ask_size'] = 3
    quotes['timestamp'] = m['starts'][17] - 1_000_000
    path, orders, _ = replay(m, base, np.ones(1), Execution(quote_max_age_seconds=3600.), quotes)
    # Entries consume the ask on the long and bid on the short only once.
    assert orders[(orders[:, 1] == 0) & (orders[:, 5] > 0), 5].sum() <= 3


def test_future_prices_and_quotes_cannot_change_past_orders():
    m, base, quotes = market()
    a, ao, _ = replay(m, base, np.ones(1), Execution(), quotes)
    m['open'][30:] *= 2; m['close'][30:] *= 2; quotes['bid'][30:] *= 2; quotes['ask'][30:] *= 2
    b, bo, _ = replay(m, base, np.ones(1), Execution(), quotes)
    np.testing.assert_array_equal(a[:30], b[:30])
    np.testing.assert_array_equal(ao[ao[:, 0] < 30], bo[bo[:, 0] < 30])


def test_original_instrumentation_is_bitwise_identical():
    m, base, _ = market()
    args = (base, m['open'], m['close'], m['sessions'], m['actions'], m['sector'], True, np.ones(1), .25)
    original = original_replay(*args)
    instrumented = replay_with_orders(*args)
    for a, b in zip(original, instrumented[:5]): np.testing.assert_array_equal(a, b)
    assert len(instrumented[5]) == int(original[0][:, 10].sum())


def test_rounding_and_minimum_discretionary_order():
    m, base, quotes = market()
    target = feasible_round(np.array([41., -40., 0, 0]), np.array([100., 103., 1., 1.]), 100000., m['sector'], .0025)
    assert limits(target, np.array([100., 103., 1., 1.]), 100000., m['sector'], .0025) < 1e-9
    base *= .01
    path, orders, _ = replay(m, base, np.ones(1), Execution(), quotes)
    assert len(orders) == 0


def test_unavailable_closing_is_not_replaced_by_close_proxy():
    m, base, quotes = market()
    quotes['bid'][-5:] = np.nan
    path, orders, failures = replay(m, base, np.ones(1), Execution(), quotes)
    assert failures > 0 and path[-1, 16] > 0


def test_missing_cohorts_never_pass():
    assert gate([], {})['status'] == 'inconclusive'


def test_trial_resume_verifies_archive_and_keeps_dollar_units(tmp_path):
    import json
    from experiments.strategy_validation.runner import execute_one
    from experiments.strategy_search.storage import atomic
    m, base, _ = market()
    m['scope'] = 'initial_validation'; m['symbols'] = ['A', 'B', 'C', 'D']
    reference = tmp_path / 'reference'; reference.mkdir()
    atomic(reference / 'protocol.json', {'partitions': {'initial_validation': {'block': {'session_ids': [0]}}}})
    output = tmp_path / 'result'; output.mkdir()
    protocol = {'binding': {'reference': str(reference)}}
    atomic(output / 'protocol.json', protocol)
    a = execute_one(m, base, Execution(), 'primary', 'bar_proxy', None, output, protocol)
    b = execute_one(m, base, Execution(), 'primary', 'bar_proxy', None, output, protocol)
    assert a == b
    with np.load(a['archive']) as z:
        assert z['equity'][-1, 1] == 100000
        assert z['daily'][0, 1] == 0
    with open(a['archive'], 'ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='Changed trial'):
        execute_one(m, base, Execution(), 'primary', 'bar_proxy', None, output, protocol)


def test_quote_assembly_detects_tampering(tmp_path):
    from experiments.strategy_validation.quotes import assemble
    from experiments.strategy_search.storage import atomic
    m, _, _ = market(); m['scope'] = 'toy'; m['symbols'] = ['A', 'B', 'C', 'D']
    stamp = int(m['starts'][0])
    p = tmp_path / f'{stamp}.json'
    atomic(p, dict(feed='sip', arrival_ns=stamp, quotes={'A': [99., 100., 100., 100., stamp - 1]}))
    result = assemble(m, [(0, [0])], tmp_path)
    assert result['timestamp'][0, 0] == stamp - 1
    atomic(p, dict(feed='sip', arrival_ns=stamp, quotes={'A': [98., 100., 100., 100., stamp - 1]}))
    with pytest.raises(ValueError, match='Changed cached'):
        assemble(m, [(0, [0])], tmp_path)


def test_prospective_freeze_rejects_backdating(tmp_path):
    from experiments.strategy_validation.shadow import initialize, collect
    from experiments.strategy_search.storage import atomic, sha
    protocol = {'frozen_UTC': '2026-10-26T23:00:00+00:00', 'binding': {'config': {'family': 'S10'}}}
    atomic(tmp_path / 'protected_sessions.json', {'reserved_session_dates': [], 'verified_no_existing_reservation': True})
    protocol['binding']['protected_sessions_sha256'] = sha(tmp_path / 'protected_sessions.json')
    atomic(tmp_path / 'protocol.json', protocol)
    p = initialize(tmp_path / 'reference', tmp_path, protocol)
    assert p['earliest_session'] == '2026-10-27'
    with pytest.raises(ValueError, match='precedes prospective'):
        collect(tmp_path / 'reference', tmp_path, protocol, None, '2026-10-26')


def test_forward_collection_caches_bars_and_prior_announced_actions(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import experiments.strategy_validation.shadow as shadow
    from experiments.strategy_search.storage import atomic, read, sha
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 28, 23, tzinfo=timezone.utc)
    monkeypatch.setattr(shadow, 'datetime', Clock)
    monkeypatch.setattr(shadow, 'credentials', lambda _: {})
    calls = []
    def response(path, params, headers):
        calls.append((path, params.copy()))
        if path == '/v1/corporate-actions':
            return {'corporate_actions': {'forward_splits': [{'symbol': 'A', 'ex_date': '2026-10-27', 'process_date': '2026-09-01'}]}}
        assert path == '/v2/stocks/bars'
        return {'bars': {'A': [dict(t='2026-10-27T13:30:00Z', o=100, h=101, l=99, c=100, v=1000, n=20, vw=100)]}}
    monkeypatch.setattr(shadow, 'request', response)
    reference = tmp_path / 'reference'; market = tmp_path / 'market'
    atomic(market / 'metadata.json', {'symbols': ['A', 'B']})
    atomic(reference / 'protocol.json', {'market_root': str(market)})
    output = tmp_path / 'output'
    protocol = {'frozen_UTC': '2026-10-26T23:00:00+00:00', 'binding': {'config': {'family': 'S10'}}}
    atomic(output / 'protected_sessions.json', {'reserved_session_dates': [], 'verified_no_existing_reservation': True})
    protocol['binding']['protected_sessions_sha256'] = sha(output / 'protected_sessions.json')
    atomic(output / 'protocol.json', protocol)
    result = shadow.collect(reference, output, protocol, None, '2026-10-27')
    assert result['sessions'] == 1
    action = read(output / 'shadow/sessions/2026-10-27.actions.json')
    assert action['excluded_symbols'] == ['A']
    assert next(params for path, params in calls if path == '/v1/corporate-actions')['start'] < '2026-09-01'
    count = len(calls)
    assert shadow.collect(reference, output, protocol, None, '2026-10-27')['status'] == 'already_collected'
    assert len(calls) == count


def test_forward_collection_respects_sealed_sessions_and_requires_audit(tmp_path):
    from experiments.strategy_validation.shadow import initialize, collect
    from experiments.strategy_search.storage import atomic, sha
    protocol = {'frozen_UTC': '2026-10-26T23:00:00+00:00', 'binding': {'config': {'family': 'S10'}}}
    atomic(tmp_path / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='reservation-manifest'):
        initialize(tmp_path / 'reference', tmp_path, protocol)
    atomic(tmp_path / 'protected_sessions.json', {'reserved_session_dates': ['2026-10-27', '2027-01-13']})
    protocol['binding']['protected_sessions_sha256'] = sha(tmp_path / 'protected_sessions.json')
    atomic(tmp_path / 'protocol.json', protocol)
    p = initialize(tmp_path / 'reference', tmp_path, protocol)
    assert p['earliest_session'] == '2027-01-14'
    with pytest.raises(ValueError, match='precedes prospective'):
        collect(tmp_path / 'reference', tmp_path, protocol, None, '2026-10-27')
    atomic(tmp_path / 'protected_sessions.json', {'reserved_session_dates': []})
    with pytest.raises(ValueError, match='reservation-manifest'):
        initialize(tmp_path / 'reference', tmp_path, protocol)
