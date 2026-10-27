# Instrumented copy of strategy_search/execution.py (financial source 7037aac085c3eec8decbe51373b82fc1f4ee573594eaf30244c9a45239d0ea5f).
# Additive order journal only; exact daily reproduction is required before use.
"""Delayed shares/cash replay, full-fill proxies and explicit turnover budgets.

No quote, queue, market-impact, participation or dollar-capacity model is present.
Event sleeves hold shares between event changes. Stops use completed closes and
the same execution delay as entries. Closing/risk orders cannot be suppressed.
"""
import numpy as np
from numba import njit

COLUMNS = ('nav_open', 'nav_close', 'cash', 'gross', 'net', 'max_stock', 'max_sector',
           'turnover_dollars', 'transaction_cost', 'borrow_cost', 'orders', 'stale_gross',
           'posttrade_constraint_error', 'pretrade_drift_error', 'liquidation_failure_gross',
           'valuation_failures', 'action_failures')


@njit(cache=True)
def constraint_error(w, sector, neutral):
    error = max(0., np.abs(w).sum() - 1., np.abs(w).max() - .05)
    error = max(error, abs(w.sum()) if neutral else -w.min())
    for k in range(int(sector.max()) + 1):
        total = w[sector == k].sum()
        error = max(error, abs(total) - .05 if neutral else total - .25)
    return max(0., error)


@njit(cache=True)
def project(weights, sector, neutral, fixed, fixed_weights, operating=False):
    w = weights.copy()
    stock_cap = .04 if operating else .05
    gross_cap = .9 if operating else 1.
    sector_cap = (.04 if neutral else .2) if operating else (.05 if neutral else .25)
    for j in range(len(w)):
        w[j] = fixed_weights[j] if fixed[j] else min(stock_cap, max(-stock_cap if neutral else 0., w[j]))
    for iteration in range(160):
        before = w.copy()
        gross = np.abs(w).sum()
        if gross > gross_cap:
            locked = np.abs(w[fixed]).sum()
            adjustable = np.abs(w[~fixed]).sum()
            scale = min(1., max(0., gross_cap - locked) / max(adjustable, 1e-30))
            w[~fixed] *= scale
        for k in range(int(sector.max()) + 1):
            mask = sector == k
            total = w[mask].sum()
            positive = mask & ~fixed & (w > 0)
            negative = mask & ~fixed & (w < 0)
            if total > sector_cap:
                scale = max(0., 1 - (total - sector_cap) / max(w[positive].sum(), 1e-30))
                w[positive] *= scale
            if neutral and total < -sector_cap:
                scale = max(0., 1 - (-total - sector_cap) / max(-w[negative].sum(), 1e-30))
                w[negative] *= scale
        if neutral:
            net = w.sum()
            positive = ~fixed & (w > 0)
            negative = ~fixed & (w < 0)
            if net > 0:
                w[positive] *= max(0., 1 - net / max(w[positive].sum(), 1e-30))
            elif net < 0:
                w[negative] *= max(0., 1 + net / max(-w[negative].sum(), 1e-30))
        # Operating projection also has to converge below its tighter limits.
        err = max(0., np.abs(w).sum() - gross_cap, np.abs(w).max() - stock_cap)
        if neutral:
            err = max(err, abs(w.sum()))
        for k in range(int(sector.max()) + 1):
            total = w[sector == k].sum()
            err = max(err, abs(total) - sector_cap if neutral else total - sector_cap)
        # A deterministic pass at an exact fixed point cannot improve on a
        # subsequent pass. Preserve the infeasibility for the accounting audit.
        if err < 1e-10 or np.array_equal(w, before):
            break
    return w


@njit(cache=True)
def trade(q, cash, marks, fresh, wanted, sector, neutral, cost, minute, entry_time, journal, reason):
    nav = cash + np.dot(q, marks)
    after = nav
    fixed = ~fresh
    for iteration in range(12):
        locked = q * marks / max(after, 1e-30)
        w = project(wanted, sector, neutral, fixed, locked)
        turnover = np.abs((w * after - q * marks)[fresh]).sum()
        updated = nav - cost * turnover
        if abs(updated - after) < 1e-14 * max(nav, 1e-300):
            after = updated
            break
        after = updated
    turnover = 0.
    orders = 0
    closed = 0.
    duration = 0.
    for j in range(len(q)):
        if not fresh[j]:
            continue
        target = w[j] * after / marks[j]
        delta = target - q[j]
        if abs(delta * marks[j]) < 1e-13 * max(nav, 1e-30):
            continue
        journal.append((minute, float(j), delta, marks[j], abs(delta * marks[j]), float(reason)))
        old = q[j]
        if old * target > 0 and abs(target) > abs(old):
            entry_time[j] = (abs(old) * entry_time[j] + abs(delta) * minute) / abs(target)
        elif old == 0:
            entry_time[j] = minute
        else:
            dollars = min(abs(old), abs(delta)) * marks[j]
            closed += dollars
            duration += dollars * (minute - entry_time[j])
            if old * target < 0:
                entry_time[j] = minute
        dollars = abs(delta) * marks[j]
        turnover += dollars
        orders += 1
        cash -= delta * marks[j] + cost * dollars
        q[j] = target
    nav = cash + np.dot(q, marks)
    error = constraint_error(q * marks / max(nav, 1e-30), sector, neutral)
    return cash, turnover, orders, error, closed, duration


@njit(cache=True)
def replay_with_orders(base, open_price, close_price, sessions, actions, sector, neutral, scales,
           cost_bps=0., borrow_annual=.03, delay=2, partial=1., interval=1, day_cap=0., inventory=False):
    journal = [(0., 0., 0., 0., 0., 0.)]
    t, n = open_price.shape
    path = np.zeros((t, len(COLUMNS)))
    q = np.zeros(n)
    cash = 1.
    marks = np.zeros(n)
    entry_time = np.zeros(n)
    pending = np.zeros((8, n))
    changes = np.zeros((8, n), np.bool_)
    due = np.full(8, -1, np.int64)
    previous = np.zeros(n)
    session_no = -1
    session_end = 0
    session_start = 0
    session_nav = 1.
    spent = 0.
    total_closed = 0.
    duration = 0.
    failures = 0
    budget_breaches = 0
    for i in range(t):
        if i == 0 or sessions[i] != sessions[i - 1]:
            session_no += 1
            session_start = i
            session_end = i
            while session_end + 1 < t and sessions[session_end + 1] == sessions[i]:
                session_end += 1
            if i and np.abs(q * marks).sum() > 1e-10 * max(path[i - 1, 1], 1e-300):
                failures += 1
            session_nav = cash + np.dot(q, marks)
            spent = 0.
            previous[:] = 0
            due[:] = -1
        fresh = np.isfinite(open_price[i]) & (open_price[i] > 0)
        marks[fresh] = open_price[i, fresh]
        pre_nav = cash + np.dot(q, marks)
        if pre_nav <= 0 or not np.isfinite(pre_nav):
            failures += 1
            path[i:, :2] = np.nan
            break
        held = q * marks / pre_nav
        drift = constraint_error(held, sector, neutral)
        valuation = int(((np.abs(q) > 0) & (marks <= 0)).sum())
        action_failure = int(((np.abs(q * marks) > 1e-12 * pre_nav) & actions[i]).sum())
        slot = i % 8
        scheduled = due[slot] == i
        flatten = i >= session_end - 4
        turnover = 0.
        fees = 0.
        orders = 0
        error = 0.
        if scheduled or drift > 1e-8 or (flatten and np.abs(q * marks).sum() > 1e-12 * pre_nav):
            wanted = held.copy()
            if flatten:
                wanted[:] = 0
            elif scheduled:
                for j in range(n):
                    if changes[slot, j]:
                        wanted[j] = pending[slot, j]
                wanted = project(wanted, sector, neutral, ~fresh, held, True)
                if inventory:
                    wanted = held + partial * (wanted - held)
                    # Reserve enough turnover to exit; mandatory risk/closing trades
                    # are never blocked. Any actual cap breach invalidates the run.
                    distance = np.abs(wanted - held).sum()
                    spare = day_cap * session_nav / pre_nav - spent / pre_nav - np.abs(held).sum()
                    alpha = min(1., max(0., spare) / max(2 * distance, 1e-30))
                    wanted = held + alpha * (wanted - held)
            cash, tr, count, error, closed, elapsed = trade(q, cash, marks, fresh, wanted, sector,
                                                           neutral, cost_bps / 10000., float(i), entry_time, journal,
                                                           3 if flatten else (1 if scheduled else 2))
            turnover += tr
            orders += count
            fees += tr * cost_bps / 10000.
            total_closed += closed
            duration += elapsed
            if scheduled:
                due[slot] = -1
        nav_open = cash + np.dot(q, marks)
        close_fresh = np.isfinite(close_price[i]) & (close_price[i] > 0)
        marks[close_fresh] = close_price[i, close_fresh]
        borrow = -np.minimum(q * marks, 0).sum() * borrow_annual / (365 * 1440.)
        cash -= borrow
        nav = cash + np.dot(q, marks)
        stale = np.abs((q * marks)[~close_fresh]).sum() / max(nav, 1e-30)
        liquidation_failure = 0.
        if i == session_end:
            for j in range(n):
                if close_fresh[j] and abs(q[j] * marks[j]) > 1e-13 * max(nav, 1e-30):
                    dollars = abs(q[j] * marks[j])
                    journal.append((float(i), float(j), -q[j], marks[j], dollars, 4.))
                    fee = dollars * cost_bps / 10000.
                    cash += q[j] * marks[j] - fee
                    turnover += dollars
                    fees += fee
                    orders += 1
                    total_closed += dollars
                    duration += dollars * (i + 1 - entry_time[j])
                    q[j] = 0.
            nav = cash + np.dot(q, marks)
            liquidation_failure = np.abs(q * marks).sum() / max(nav, 1e-30)
            if liquidation_failure > 1e-10:
                failures += 1
        spent += turnover
        if i == session_end and day_cap and spent > day_cap * session_nav + 1e-8:
            budget_breaches += 1
        w = q * marks / max(nav, 1e-30)
        max_sector = 0.
        for k in range(int(sector.max()) + 1):
            max_sector = max(max_sector, abs(w[sector == k].sum()))
        path[i] = np.array([nav_open, nav, cash, np.abs(w).sum(), w.sum(), np.abs(w).max(), max_sector,
                            turnover, fees, borrow, orders, stale, error, drift, liquidation_failure,
                            float(valuation), float(action_failure)])
        changed = np.abs(base[i] - previous) > 1e-12
        should_schedule = (inventory and (i - session_start) % interval == 0) or np.any(changed)
        if should_schedule and i + delay < session_end - 4:
            execute = i + delay
            pending[execute % 8] = base[i] * scales[session_no]
            changes[execute % 8] = np.ones(n, np.bool_) if inventory else changed
            due[execute % 8] = execute
        previous = base[i].copy()
    return path, total_closed, duration, failures, budget_breaches, journal[1:]
