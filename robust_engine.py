"""Extended portfolio policies; original daily accounting is preserved verbatim.

Additional policies use only signal-time returns and previous target holdings.
The buffer retains prior target names within the best 2*K ranks; unfilled orders
can remain in that target, so this is explicitly a target-retention buffer.
"""
import numpy as np
import pandas as pd
import warnings
from research_engine import performance, target_weights, ranks

METHODS = ['equal', 'confidence', 'inverse_vol', 'buffer_equal', 'buffer_inverse_vol']

def allocation(score, mask, k, method, trailing_returns, previous):
    if method in ['equal', 'confidence', 'long_short']:
        return target_weights(score, mask, k, method)
    clean = np.where(mask, score, 0.)
    if not np.isfinite(clean).all():
        raise ValueError('Invalid eligible score')
    selected = target_weights(clean, mask, k, 'equal')
    if method.startswith('buffer_'):
        rank = np.full(len(score), np.inf)
        rank[mask] = ranks(-clean[mask])
        # Rank ties remain symmetric: selection uses the existing fractional tie rule.
        retain = (previous > 0) & mask & (rank <= 2*k)
        priority = clean.copy()
        priority[mask] = ranks(clean[mask])/max(mask.sum(), 1)
        priority[retain] += 2
        selected = target_weights(priority, mask, k, 'equal')
    if method.endswith('inverse_vol'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            vol = np.nanstd(trailing_returns, axis=0, ddof=1)*np.sqrt(252)
        # Fixed 10% volatility floor limits extreme weights. No future risk estimates.
        selected /= np.maximum(np.nan_to_num(vol, nan=.1), .1)
        selected /= selected.sum()
        cap = max(2/k, 1/max(mask.sum(), 1))
        # Water-fill at 2/K per name where enough names are selected.
        if np.count_nonzero(selected)*cap >= 1:
            for _ in range(len(score)):
                over = selected > cap+1e-12
                if not over.any(): break
                excess = (selected[over]-cap).sum()
                selected[over] = cap
                free = (selected > 0) & ~over & (selected < cap-1e-12)
                if not free.any(): break
                selected[free] += excess*selected[free]/selected[free].sum()
    return selected


def backtest(data, ids, scores, k, method, cost_multiplier=1., aum=1_000_000.):
    """Daily marked NAV; weights drift, trades limited using signal-time ADV.

    Signal close t; irrevocable order for close t+1; held through t+6. Fixed
    notional target orders are computed at t; fill close is not used to choose
    names. Dollar-order/auction fractional-share approximation is explicit.
    Gross and net are matched paths on identical actual dollar holdings.
    """
    previous_target = np.zeros(len(data["panel"].tickers))
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
            w = allocation(scores[j], data['mask'][i], k, method,
                           data['returns'][max(0, signal-59):signal+1], previous_target)
            previous_target = w.copy()
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

