"""Causal data, full-universe ranking metrics and stateful daily portfolio accounting."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd


@dataclass
class Panel:
    name: str
    dates: pd.DatetimeIndex
    tickers: list
    close: np.ndarray
    volume: np.ndarray
    benchmark: np.ndarray
    notes: list


def load_panel(name):
    if name in ('NYSE_recent', 'NASDAQ_recent'):
        root = Path('data/recent_cohorts')
        manifest = json.loads((root/'manifest.json').read_text())
        names = manifest['selection'][name]
        read = lambda field: pd.read_csv(root/f'{field}.csv', index_col=0, parse_dates=True)
        adj, close, vol = read('adj_close'), read('close'), read('volume')
        exclusions = json.loads(Path('config/recent_identity_exclusions.json').read_text())
        for ticker in exclusions:
            if ticker in names:
                adj[ticker] = np.nan
                close[ticker] = np.nan
                vol[ticker] = np.nan
        return Panel(name, adj.index, names, adj[names].to_numpy(),
                     (close[names]*vol[names]).to_numpy(), adj['SPY'].to_numpy(),
                     manifest['limitations'] + ['SPY total-return approximation benchmark for both cohorts.',
                     'AVX/CAMP current-vendor histories rejected for issuer mismatch; original histories remain unavailable.'])
    if name == 'modern12':
        root = Path('data/research_panel')
        names = json.loads((root/'manifest.json').read_text())['tickers']
        read = lambda field: pd.read_csv(root/f'{field}.csv', index_col=0, parse_dates=True)
        adj, close, vol = read('adj_close'), read('close'), read('volume')
        # Use split/dividend-adjusted marks, but raw close*volume for liquidity.
        return Panel(name, adj.index, names, adj[names].to_numpy(),
                     (close[names]*vol[names]).to_numpy(), adj['SPY'].to_numpy(),
                     ['Fixed survivor basket; revised Yahoo adjusted total-return marks.',
                      'SPY adjusted-close benchmark; no historical constituent/delisting database.'])
    root = Path('data/research_cohorts')
    names = json.loads((root/'selection.json').read_text())[name]
    benchmark = pd.read_csv(root/'benchmark.csv', index_col=0, parse_dates=True)
    dates = benchmark.loc['2012-01-01':'2017-12-08'].index
    cs, vs = [], []
    for ticker in names:
        f = pd.read_csv(root/'raw'/f'{name}_{ticker}.csv', index_col=0)
        f.index = pd.to_datetime(f.index.str[:10])
        if f.index.has_duplicates:
            raise ValueError('Duplicate raw stock dates')
        f = f.reindex(dates)
        c = f['Close'].where(f['Close'] > 0)
        cs.append(c.to_numpy())
        vs.append((c*f['Volume']).to_numpy())
    return Panel(name, dates, names, np.array(cs).T, np.array(vs).T,
                 benchmark.reindex(dates)['QQQ' if name == 'NASDAQ' else 'SPY'].to_numpy(),
                 ['100 names chosen by SHA256(ticker) from published survivor cohort.',
                  'Google historical close adjustment/dividend conventions unverified: price-return diagnostic.',
                  'Raw historical dollar-volume validity is conditional on vendor adjustment conventions.',
                  'NASDAQ uses QQQ proxy; NYSE uses broad US SPY proxy, not exact exchange index.',
                  '2017 was previously examined in this workspace; not a virgin confirmation sample.'])


def prepare_panel(panel, horizon=5, lookback=16):
    raw = pd.DataFrame(panel.close, index=panel.dates)
    valid = raw.notna() & (raw > 0)
    marks = raw.ffill()  # Only valuation carries forward; eligibility never uses filled prices.
    r = marks.pct_change(fill_method=None)
    ma5 = marks/marks.rolling(5).mean()-1
    ma20 = marks/marks.rolling(20).mean()-1
    vol20 = r.rolling(20).std()
    dv = pd.DataFrame(panel.volume, index=panel.dates)
    adv = dv.rolling(20, min_periods=20).mean()
    vratio = np.log((dv/adv).clip(lower=.01, upper=100))
    features = np.stack([r, ma5, ma20, vol20, vratio], axis=-1)
    eligible = (valid.rolling(40, min_periods=40).sum() == 40).to_numpy(copy=True)
    eligible &= np.isfinite(features).all(-1) & (adv.to_numpy() >= 1_000_000)
    # Fixed 5-session calendar; trading takes place at the following session's close.
    origins = np.arange(260, len(raw)-horizon-1, horizon)
    if not len(origins):
        raise ValueError('Insufficient history')
    x = np.stack([features[t-lookback+1:t+1] for t in origins])
    mask = eligible[origins] & np.isfinite(x).all((1, 3))
    if np.any(mask.sum(1) < 2):
        raise ValueError('Fewer than two eligible names: explicit cash-only support required')
    entry, exit_ = origins+1, origins+1+horizon
    # Unfilled true marks for labels. Missing future prices never change eligibility.
    y = panel.close[exit_]/panel.close[entry]-1
    labelmask = mask & np.isfinite(y)
    risk = []
    for t in origins:
        future_sq = r.iloc[t+2:t+horizon+2].to_numpy()**2
        risk.append(np.sqrt(np.nansum(future_sq, axis=0)/np.maximum(np.isfinite(future_sq).sum(0), 1)))
    risk = np.stack(risk)
    momentum = (marks/marks.shift(126)-1).to_numpy()[origins]
    reversal = -(marks/marks.shift(5)-1).to_numpy()[origins]
    return dict(panel=panel, x=np.nan_to_num(x).astype('float32'), y=np.nan_to_num(y).astype('float32'),
                mask=mask, labelmask=labelmask, risk=np.nan_to_num(risk).astype('float32'),
                origin=origins, entry=entry, exit=exit_, horizon=horizon,
                adv=adv.to_numpy(copy=True), returns=r.to_numpy(), marks=marks.to_numpy(),
                momentum=momentum, reversal=reversal, stale=~valid.to_numpy())


def fold_indices(data, test_year):
    dates = data['panel'].dates
    signal = dates[data['origin']]
    outcome = dates[data['exit']]
    vstart, tstart = pd.Timestamp(test_year-1, 1, 1), pd.Timestamp(test_year, 1, 1)
    tend = pd.Timestamp(test_year+1, 1, 1)
    train = np.flatnonzero((signal >= pd.Timestamp(test_year-5, 1, 1)) & (outcome < vstart))
    val = np.flatnonzero((signal >= vstart) & (outcome < tstart))
    test = np.flatnonzero((signal >= tstart) & (outcome < tend))
    if min(len(train), len(val), len(test)) < 15:
        raise ValueError(f'Insufficient fold history: {test_year}')
    assert outcome[train].max() < signal[val].min()
    assert outcome[val].max() < signal[test].min()
    return train, val, test


def ranks(x):
    return pd.Series(x).rank(method='average').to_numpy()


def corr(x, y):
    x, y = x-x.mean(), y-y.mean()
    den = np.sqrt(x@x*(y@y))
    return float(x@y/den) if den > 1e-15 else np.nan


def dcg(gains, scores, k):
    order = np.argsort(-scores, kind='stable')
    g, s = gains[order], scores[order]
    value, i = 0., 0
    while i < min(k, len(s)):
        j = i+1
        while j < len(s) and s[j] == s[i]:
            j += 1
        value += g[i:j].mean()*np.sum(1/np.log2(np.arange(i, min(k, j))+2))
        i = j
    return value


def ranking_metrics(scores, actual, eligible, labelmask, ks):
    rows = []
    for p, y, e, m in zip(scores, actual, eligible, labelmask):
        if not np.isfinite(p[e]).all():
            raise ValueError('Nonfinite score for eligible stock')
        row = dict(eligible=int(e.sum()), observed=int(m.sum()), unobserved=int((e & ~m).sum()))
        p, y = p[m], y[m]
        row.update(IC=corr(p, y), RankIC=corr(ranks(p), ranks(y)))
        # Percentile relevance: nonnegative, linear gains, including all-negative-return dates.
        gain = (ranks(y)-1)/max(len(y)-1, 1)
        for k in ks:
            ideal = dcg(gain, gain, k)
            row[f'NDCG@{k}'] = dcg(gain, p, k)/ideal if ideal else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {c:float(frame[c].mean()) for c in ['IC', 'RankIC']+[f'NDCG@{k}' for k in ks]}
    for c in ['IC', 'RankIC']:
        summary[c+'IR'] = float(frame[c].mean()/frame[c].std()) if frame[c].std() > 1e-12 else None
        summary[c+'_periods'] = int(frame[c].notna().sum())
    summary['unobserved_labels'] = int(frame.unobserved.sum())
    return summary, frame


def target_weights(score, mask, k, method):
    if not np.isfinite(np.asarray(score)[mask]).all():
        raise ValueError('Nonfinite score for eligible security')
    # Missing histories can yield NaN momentum/reversal scores. They must not
    # contaminate zero-weight positions via 0 * exp(NaN) in confidence weighting.
    score = np.where(mask, score, 0.)
    # Fractional allocations at a boundary tie; invariant to arbitrary ticker order.
    def side(s, budget):
        ids = np.flatnonzero(mask)
        kk = min(k, len(ids))
        boundary = np.sort(s[ids])[-kk]
        above, tied = mask & (s > boundary), mask & (s == boundary)
        w = above.astype(float) + tied*((kk-above.sum())/max(tied.sum(), 1))
        if method == 'confidence':
            z = (s-s[ids].mean())/max(s[ids].std(), 1e-8)
            w *= np.exp(np.clip(z, -2, 2))
        return w/w.sum()*budget
    if method == 'long_short':
        if mask.sum() < 2*k:
            return np.zeros(len(score))
        return side(score, .5)-side(-score, .5)
    return side(score, 1.)


def performance(daily, rf=.03):
    a = np.asarray(daily, float)
    if len(a) < 2 or np.any(a <= -1) or not np.isfinite(a).all():
        raise ValueError('Invalid portfolio path')
    equity = np.r_[1., np.cumprod(1+a)]
    vol = a.std(ddof=1)*np.sqrt(252)
    return dict(sharpe=float((a.mean()-((1+rf)**(1/252)-1))*252/vol) if vol > 1e-12 else None,
                sharpe_rf0=float(a.mean()*252/vol) if vol > 1e-12 else None,
                annualized_return=float(equity[-1]**(252/len(a))-1),
                cumulative_return=float(equity[-1]-1), annualized_volatility=float(vol),
                maximum_drawdown=float(np.min(equity/np.maximum.accumulate(equity)-1)),
                profitable_days=float(np.mean(a>0)), days=len(a))


def backtest(data, ids, scores, k, method, cost_multiplier=1., aum=1_000_000.):
    """Daily marked NAV; weights drift, trades limited using signal-time ADV.

    Signal close t; irrevocable order for close t+1; held through t+6. Fixed
    notional target orders are computed at t; fill close is not used to choose
    names. Dollar-order/auction fractional-share approximation is explicit.
    Gross and net are matched paths on identical actual dollar holdings.
    """
    ids = np.asarray(ids)
    if np.any(np.diff(ids) != 1):
        raise ValueError('Backtest expects contiguous forecast indices')
    n = len(data['panel'].tickers)
    holdings, cash, nav = np.zeros(n), aum, aum
    gross_daily, net_daily, daily_dates, turnover, periods = [], [], [], [], []
    costs, stale_marks, constrained = 0., 0, 0
    first = data['entry'][ids[0]]
    previous_day = first
    for j, i in enumerate(ids):
        signal, entry, end = data['origin'][i], data['entry'][i], data['exit'][i]
        # Consecutive periods share exit/entry close. Rebalance with already drifted dollars.
        before = nav
        move_to_fill = np.nan_to_num(data['marks'][entry]/data['marks'][signal], nan=1.)
        signal_holdings = holdings/np.maximum(move_to_fill, 1e-12)
        signal_nav = cash+signal_holdings.sum()
        if method == 'buy_hold' and j:
            target_dollars = holdings.copy()
        elif method in ('buy_hold', 'equal_universe'):
            target_dollars = data['mask'][i]/data['mask'][i].sum()*signal_nav*move_to_fill
        else:
            w = target_weights(scores[j], data['mask'][i], k, method)
            target_dollars = w*signal_nav*move_to_fill
        trade = target_dollars-holdings
        # Tradability at execution can reject orders, but never replace names using future data.
        tradable = ~data['stale'][entry]
        cap = np.nan_to_num(data['adv'][signal], nan=0.)*.01*move_to_fill
        actual_trade = np.clip(trade, -cap, cap)*tradable
        constrained += int(np.count_nonzero(np.abs(actual_trade-trade) > .01))
        participation = np.abs(actual_trade)/np.maximum(np.nan_to_num(data['adv'][signal], nan=1), 1)
        # 2bp commission, 5bp half-spread/slippage, sqrt participation impact (3bp at 1% ADV).
        rate = (.0002+.0005+.0003*np.sqrt(participation/.01))*cost_multiplier
        fee = float(np.abs(actual_trade)@rate)
        # Reserve cash to pay costs for a fully invested long-only order.
        buys = np.maximum(actual_trade, 0)
        needed = buys.sum()+fee
        available = cash-np.minimum(actual_trade, 0).sum()
        if method != 'long_short' and needed > available and buys.sum() > 0:
            actual_trade -= buys*(min((needed-available)/buys.sum(), 1.))
            participation = np.abs(actual_trade)/np.maximum(np.nan_to_num(data['adv'][signal], nan=1), 1)
            rate = (.0002+.0005+.0003*np.sqrt(participation/.01))*cost_multiplier
            fee = float(np.abs(actual_trade)@rate)
        turnover.append(float(np.abs(actual_trade).sum()/nav))
        holdings += actual_trade
        cash -= actual_trade.sum()+fee
        costs += fee
        nav = cash+holdings.sum()
        period_growth = 1.
        for day in range(entry+1, end+1):
            old = nav
            dr = np.nan_to_num(data['returns'][day], nan=0.)
            stale_marks += int(np.count_nonzero((np.abs(holdings) > .01) & data['stale'][day]))
            profit = float(holdings@dr)
            borrow = float(np.maximum(-holdings, 0).sum()*.03/252) if method == 'long_short' else 0.
            holdings *= 1+dr
            cash -= borrow
            nav = cash+holdings.sum()
            if day == entry+1:
                # Charge execution fee on first holding-day return, without dropping it from NAV.
                net = nav/before-1
                gross = profit/before
            else:
                net = nav/old-1
                gross = profit/old
            gross_daily.append(gross)
            net_daily.append(net)
            daily_dates.append(str(data['panel'].dates[day].date()))
            period_growth *= 1+net
        periods.append(period_growth-1)
        previous_day = end
    # Close all holdings and charge terminal liquidation, with cost assumed at same schedule.
    # Recorded separately; no liquidity cap liquidation claim for hypothetical open positions.
    final_part = np.abs(holdings)/np.maximum(np.nan_to_num(data['adv'][data['origin'][ids[-1]]], nan=1), 1)
    exit_fee = float(np.abs(holdings) @ ((.0007+.0003*np.sqrt(final_part/.01))*cost_multiplier))
    if exit_fee >= nav:
        raise ValueError('Terminal liquidation cost exceeds NAV')
    net_daily[-1] = (1+net_daily[-1])*(1-exit_fee/nav)-1
    periods[-1] = (1+periods[-1])*(1-exit_fee/nav)-1
    turnover[-1] += float(np.abs(holdings).sum()/nav)
    costs += exit_fee
    metrics = {f'gross_{key}':value for key, value in performance(gross_daily).items()}
    metrics.update({f'net_{key}':value for key, value in performance(net_daily).items()})
    metrics.update(turnover_mean=float(np.mean(turnover)), turnover_annual=float(np.sum(turnover)*252/len(net_daily)),
                   profitable_periods=float(np.mean(np.array(periods)>0)),
                   cost_dollars=costs, stale_position_marks=stale_marks, constrained_orders=constrained,
                   terminal_liquidation_over_cap=int(np.count_nonzero(final_part > .01)))
    daily = pd.DataFrame(dict(date=daily_dates, gross=gross_daily, net=net_daily))
    return metrics, daily, np.array(periods)


def training_groups(data, train):
    # Training-only return-correlation neighbors; no present-day industry/Wikidata edges.
    end = data['origin'][train[-1]]
    r = np.nan_to_num(data['returns'][:end+1], nan=0.)
    std = r.std(0)
    z = (r-r.mean(0))/np.maximum(std, 1e-8)
    c = z.T@z/max(len(z), 1)
    groups = []
    for i in range(c.shape[0]):
        row = c[i].copy()
        row[i] = -np.inf
        neighbors = np.argsort(-row, kind='stable')[:4]
        groups.append(set([i]+[int(j) for j in neighbors if row[j] > 0]))
    return groups
