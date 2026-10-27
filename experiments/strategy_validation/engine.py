"""Whole-share execution of fixed S10 events with explicit partial orders."""
from dataclasses import asdict
import numpy as np

from experiments.strategy_search.execution import project
from .fees import cumulative_charge_delta

PATH_COLUMNS = ('nav_open', 'nav_close', 'cash', 'gross', 'net', 'stock', 'sector',
                'turnover', 'commission', 'exchange', 'clearing', 'regulatory', 'borrow',
                'adverse_slippage', 'constraint_error', 'action_failure', 'overnight', 'close_drift_error')
ORDER_COLUMNS = ('minute', 'stock', 'parent', 'reason', 'intended_shares', 'filled_shares',
                 'benchmark', 'fill_price', 'quote_age_ns', 'quote_valid', 'commission',
                 'exchange', 'clearing', 'regulatory', 'adverse_slippage', 'notional')
REASONS = {1: 'entry_or_event', 2: 'risk', 3: 'closing', 4: 'retry'}


def limits(shares, marks, nav, sector, net_tolerance):
    w = shares * marks / max(nav, 1e-30)
    sector_max = max((abs(w[sector == g].sum()) for g in np.unique(sector)), default=0.)
    return max(0., abs(w).sum() - 1., abs(w).max() - .05,
               abs(w.sum()) - net_tolerance, sector_max - .05)


def feasible_round(shares, marks, nav, sector, tolerance):
    """Trim existing desired exposure; never invent a new stock or direction."""
    w = project(shares * marks / max(nav, 1e-30), sector, True,
                marks <= 0, shares * marks / max(nav, 1e-30), True)
    target = np.trunc(np.divide(w * nav, marks, out=np.zeros_like(w), where=marks > 0))
    # Rounding can leave an imbalance larger than the cash neutrality band.
    for _ in range(int(np.abs(target).sum()) + 1):
        net = np.dot(target, marks)
        if abs(net) <= tolerance * nav:
            break
        eligible = np.flatnonzero((target * net > 0) & (marks > 0))
        if not len(eligible):
            break
        j = eligible[np.argmin(np.abs(net - np.sign(net) * marks[eligible]))]
        target[j] -= np.sign(target[j])
    if limits(target, marks, nav, sector, tolerance) > 1e-9:
        return np.zeros_like(target)
    return target


def replay(m, base, scale, settings, quotes=None):
    """quotes=None is an explicitly labeled minute-open diagnostic scenario."""
    t, n = base.shape
    q, marks, goal = np.zeros(n), np.zeros(n), np.zeros(n)
    cash = settings.capital
    path = np.zeros((t, len(PATH_COLUMNS)))
    journal = []
    failures = 0
    sid = -1
    end = 0
    parent = np.zeros(n, dtype=np.int64)
    filled_parent, notional_parent = np.zeros(n), np.zeros(n)
    parent_serial = 0
    last_quote = np.full(n, -1, np.int64)
    used_bid, used_ask = np.zeros(n), np.zeros(n)
    for i in range(t):
        new_session = i == 0 or m['sessions'][i] != m['sessions'][i - 1]
        if new_session:
            sid += 1
            end = i
            while end + 1 < t and m['sessions'][end + 1] == m['sessions'][i]:
                end += 1
            if i and np.any(q):
                failures += 1
            goal = q.copy()
        fresh = np.isfinite(m['open'][i]) & (m['open'][i] > 0)
        marks[fresh] = m['open'][i, fresh]
        nav = cash + np.dot(q, marks)
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError('Nonpositive or invalid portfolio NAV')
        reason = 0
        old_goal = goal.copy()
        closing = i >= end - 4
        origin = i - settings.delay
        changed = np.zeros(n, bool)
        if origin >= 0 and m['sessions'][origin] == m['sessions'][i] and i < end - 4:
            previous = base[origin - 1] if origin and m['sessions'][origin - 1] == m['sessions'][origin] else np.zeros(n)
            changed = np.abs(base[origin] - previous) > 1e-12
        if closing:
            if i == end - 4:
                goal[:] = 0
                reason = 3
            elif np.any(q):
                reason = 3
        elif changed.any():
            goal[changed] = np.trunc(np.divide(base[origin, changed] * scale[sid] * nav, marks[changed],
                                              out=np.zeros(changed.sum()), where=marks[changed] > 0))
            goal = feasible_round(goal, marks, nav, m['sector'], settings.net_tolerance)
            reason = 1
        if not closing and limits(q, marks, nav, m['sector'], settings.net_tolerance) > 1e-9:
            goal = feasible_round(q, marks, nav, m['sector'], settings.net_tolerance)
            reason = 2
        if not np.any(q) and not np.any(goal):
            close_fresh = np.isfinite(m['close'][i]) & (m['close'][i] > 0)
            marks[close_fresh] = m['close'][i, close_fresh]
            path[i, :3] = cash
            continue
        delta = goal - q
        if reason == 1:
            # Skip only discretionary adjustments; actual stock exits bypass it.
            discretionary = (goal != 0) & (np.abs(delta * marks) < settings.minimum_order_weight * nav)
            goal[discretionary] = q[discretionary]
            delta = goal - q
            if limits(goal, marks, nav, m['sector'], settings.net_tolerance) > 1e-9:
                goal = q.copy()
                delta[:] = 0
        if reason:
            reset = (goal != old_goal) | ((delta != 0) & (parent == 0))
            for j in np.flatnonzero(reset):
                parent_serial += 1
                parent[j] = parent_serial
                filled_parent[j] = notional_parent[j] = 0.
        if not reason and delta.any():
            reason = 4
        turnover = commission = exchange = clearing = regulatory = slippage = 0.
        intended = np.flatnonzero(delta != 0)
        if len(intended):
            proposal = np.zeros(n)
            prices = marks.copy()
            valid_quote = np.zeros(n, bool)
            stamps = np.full(n, np.nan)
            for j in intended:
                if not fresh[j]:
                    continue
                buy = delta[j] > 0
                volume = m['volume'][i - 1, j] if not new_session else 0.
                capacity = max(0., np.floor(volume * settings.participation)) if np.isfinite(volume) else 0.
                if quotes is None:
                    valid_quote[j] = True
                    available = capacity
                else:
                    bid, ask, bs, az, stamp = quotes[i, j]
                    age_ns = int(m['starts'][i]) - int(stamp)
                    age = age_ns / 1e9
                    good = np.isfinite([bid, ask, bs, az, stamp]).all() and 0 < bid <= ask and bs > 0 and az > 0
                    good &= 0 <= age <= settings.quote_max_age_seconds
                    if not good:
                        continue
                    stamps[j] = age_ns
                    valid_quote[j] = True
                    if int(stamp) != last_quote[j]:
                        used_bid[j] = used_ask[j] = 0.
                        last_quote[j] = int(stamp)
                    available = min(capacity, max(0., az - used_ask[j] if buy else bs - used_bid[j]))
                    prices[j] = ask if buy else bid
                prices[j] *= 1 + (1 if buy else -1) * settings.adverse_bps / 10000.
                proposal[j] = np.sign(delta[j]) * min(abs(delta[j]), np.floor(available))
            # Back off a batch when asymmetric fills would create new excessive
            # exposure. Closing/risk reductions must be attempted regardless.
            if reason not in (2, 3) and limits(q, marks, nav, m['sector'], settings.net_tolerance) <= 1e-9:
                original = proposal.copy()
                for iteration in range(25):
                    estimated_cost = 0.
                    for j in np.flatnonzero(proposal):
                        qty = abs(proposal[j])
                        estimated_cost += sum(cumulative_charge_delta(filled_parent[j], notional_parent[j], qty,
                                              qty * prices[j], np.sign(proposal[j]), settings))
                    post_nav = nav - estimated_cost - np.dot(proposal, prices - marks)
                    if limits(q + proposal, marks, post_nav, m['sector'], settings.net_tolerance) <= 1e-9:
                        break
                    proposal = np.trunc(original * .5 ** (iteration + 1))
            for j in intended:
                fill = proposal[j]
                costs = (0.,) * 4
                notional = abs(fill * prices[j])
                if fill:
                    costs = cumulative_charge_delta(filled_parent[j], notional_parent[j], abs(fill), notional,
                                                    np.sign(fill), settings)
                    filled_parent[j] += abs(fill)
                    notional_parent[j] += notional
                    cash -= fill * prices[j] + sum(costs)
                    q[j] += fill
                    turnover += notional
                    commission += costs[0]; exchange += costs[1]; clearing += costs[2]; regulatory += costs[3]
                    if quotes is not None:
                        if fill > 0: used_ask[j] += abs(fill)
                        else: used_bid[j] += abs(fill)
                penalty = notional * settings.adverse_bps / (10000. + np.sign(fill) * settings.adverse_bps) if fill else 0.
                slippage += penalty
                journal.append((i, j, parent[j], reason, delta[j], fill, marks[j], prices[j] if fill else np.nan,
                                stamps[j], float(valid_quote[j]), *costs, penalty, notional))
        nav_open = cash + np.dot(q, marks)
        post_error = limits(q, marks, nav_open, m['sector'], settings.net_tolerance)
        close_fresh = np.isfinite(m['close'][i]) & (m['close'][i] > 0)
        marks[close_fresh] = m['close'][i, close_fresh]
        borrow = -np.minimum(q * marks, 0).sum() * settings.borrow / (365 * 1440.)
        cash -= borrow
        nav_close = cash + np.dot(q, marks)
        error = limits(q, marks, nav_close, m['sector'], settings.net_tolerance)
        action = int(np.any((q != 0) & m['actions'][i]))
        overnight = float(abs(q * marks).sum()) if i == end else 0.
        failures += int(action or overnight > 1e-8 or post_error > 1e-7)
        w = q * marks / nav_close
        sm = max(abs(w[m['sector'] == g].sum()) for g in np.unique(m['sector']))
        path[i] = (nav_open, nav_close, cash, abs(w).sum(), w.sum(), abs(w).max(), sm,
                   turnover, commission, exchange, clearing, regulatory, borrow, slippage, post_error, action, overnight, error)
    return path, np.asarray(journal, float).reshape(-1, len(ORDER_COLUMNS)), failures
