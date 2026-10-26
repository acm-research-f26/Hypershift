"""Completed-bar features and fit-only normalizers. Missing windows stay missing."""
import numpy as np
from numba import njit


@njit(cache=True)
def clocks(sessions):
    result = np.zeros(len(sessions), np.int64)
    for i in range(1, len(sessions)):
        if sessions[i] == sessions[i - 1]:
            result[i] = result[i - 1] + 1
    return result


def returns(close, sessions):
    good = np.isfinite(close) & (close > 0)
    valid = np.zeros_like(good)
    valid[1:] = good[1:] & good[:-1] & (sessions[1:] == sessions[:-1])[:, None]
    result = np.zeros_like(close, dtype=np.float64)
    np.log(np.divide(close[1:], close[:-1], out=np.ones_like(result[1:]), where=valid[1:]), out=result[1:])
    return result, valid


def peers(values, valid, groups):
    result = np.zeros_like(values)
    available = np.zeros_like(valid)
    for group in np.unique(groups):
        ix = np.flatnonzero(groups == group)
        v, ok = values[:, ix], valid[:, ix]
        count = ok.sum(1)[:, None] - ok
        total = np.where(ok, v, 0).sum(1)[:, None] - np.where(ok, v, 0)
        result[:, ix] = np.divide(total, count, out=np.zeros_like(v), where=count >= 2)
        available[:, ix] = count >= 2
    return result, available


@njit(cache=True)
def window_sum(values, valid, sessions, window):
    t, n = values.shape
    result = np.zeros_like(values)
    available = np.zeros_like(valid)
    total = np.zeros(n)
    count = np.zeros(n, np.int64)
    for i in range(t):
        if i == 0 or sessions[i] != sessions[i - 1]:
            total[:] = 0
            count[:] = 0
        for j in range(n):
            if valid[i, j]:
                total[j] += values[i, j]
                count[j] += 1
            if i >= window and sessions[i - window] == sessions[i] and valid[i - window, j]:
                total[j] -= values[i - window, j]
                count[j] -= 1
            if count[j] == window:
                result[i, j] = total[j]
                available[i, j] = True
    return result, available


def moments(values, valid, train):
    mask = valid & train[:, None]
    count = mask.sum(0)
    mean = np.divide(np.where(mask, values, 0).sum(0), count, out=np.zeros(values.shape[1]), where=count > 0)
    variance = np.divide(np.where(mask, (values - mean) ** 2, 0).sum(0), np.maximum(count - 1, 1))
    return mean, np.where((count >= 100) & (variance > 1e-16), np.sqrt(variance), np.nan)


def normalize(values, valid, mean, sd):
    ok = valid & np.isfinite(sd)[None, :]
    return np.where(ok, (values - mean) / sd, 0), ok


def stock_volatility(raw, valid, sessions):
    total, ok = window_sum(raw, valid, sessions, 60)
    squares, _ = window_sum(raw * raw, valid, sessions, 60)
    sd = np.sqrt(np.maximum((squares - total * total / 60) / 59, 0))
    return sd, ok & (sd > 1e-8)


@njit(cache=True)
def event_targets(score, eligible, exit_long, exit_short, volatility, sessions,
                  sector, industry, long_short, max_hold, max_entries, sector_events,
                  inventory, interval, basket, basket_members):
    """Freeze entry sleeve sizes; keep events until declared completed-close exits."""
    t, n = score.shape
    result = np.zeros_like(score)
    position = np.zeros(n)
    entered = np.zeros(n, np.int64)
    entries = np.zeros(n, np.int64)
    sector_count = np.zeros(int(sector.max()) + 1, np.int64)
    events = 0
    clock = 0
    for i in range(t):
        if i == 0 or sessions[i] != sessions[i - 1]:
            position[:] = 0
            entries[:] = 0
            sector_count[:] = 0
            clock = 0
        else:
            clock += 1
        exited = np.zeros(n, np.bool_)
        for j in range(n):
            if position[j] and not inventory:
                leave = i - entered[j] >= max_hold or (exit_long[i, j] if position[j] > 0 else exit_short[i, j])
                if leave:
                    position[j] = 0
                    exited[j] = True
        if not inventory or clock % interval == 0:
            hi = np.full(n, -np.inf)
            lo = np.full(n, np.inf)
            count = 0
            for j in range(n):
                if eligible[i, j]:
                    count += 1
                    hi[j] = score[i, j]
                    lo[j] = score[i, j]
            top = np.argsort(-hi)
            bottom = np.argsort(lo)
            k = max(1, int(np.ceil(.1 * count)))
            keep = max(k, int(np.ceil(.3 * count)))
            want = np.zeros(n, np.int64)
            for rank in range(min(n, keep if inventory else k)):
                j = top[rank]
                if eligible[i, j] and score[i, j] > 0 and (rank < k or position[j] > 0):
                    want[j] = 1
                j = bottom[rank]
                if long_short and eligible[i, j] and score[i, j] < 0 and (rank < k or position[j] < 0):
                    want[j] = -1
            used = np.zeros(len(sector_count), np.bool_)
            for j in range(n):
                if inventory and eligible[i, j] and (want[j] == 0 or position[j] * want[j] < 0):
                    position[j] = 0
                if want[j] and not position[j] and not exited[j] and entries[j] < max_entries:
                    group = sector[j]
                    if sector_events and sector_count[group] >= sector_events:
                        continue
                    if volatility[i, j] <= 1e-8 or not np.isfinite(volatility[i, j]):
                        continue
                    position[j] = want[j] * min(.04, .005 / (volatility[i, j] * np.sqrt(390.)))
                    entered[j] = i
                    entries[j] += 1
                    events += 1
                    used[group] = True
            for group in range(len(sector_count)):
                if used[group]:
                    sector_count[group] += 1
        result[i] = position
        if basket and long_short:
            for group in range(int(industry.max()) + 1):
                total = 0.
                count = 0
                for j in range(n):
                    if industry[j] == group and basket_members[j]:
                        total += position[j]
                        count += 1
                if count > 1:
                    for j in range(n):
                        if industry[j] == group and basket_members[j]:
                            result[i, j] = position[j] * count / (count - 1) - total / (count - 1)
    return result, events
