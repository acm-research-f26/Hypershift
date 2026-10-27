"""Dollar fees per submitted order; partial fills share their parent minimum."""
import math


def charges(shares, price, settings, *, fractional=False):
    quantity, notional = abs(shares), abs(shares * price)
    if not quantity:
        return (0.,) * 4
    if settings.commission_free:
        commission = 0.
    elif fractional and quantity < 1:
        commission = max(settings.fractional_minimum, .01 * notional)
    else:
        commission = min(max(settings.commission_minimum, settings.commission_share * quantity),
                         settings.commission_maximum_fraction * notional)
    # Route is unknown. This explicit taker scenario excludes rebates and can
    # be replaced by a versioned schedule after a broker has been selected.
    exchange = quantity * settings.exchange_taker_share
    clearing = quantity * (settings.clearing_share + settings.cat_share)
    regulatory = commission * settings.commission_pass_through
    if shares < 0:
        regulatory += settings.sec_sell_fraction * notional + min(settings.finra_sell_cap, quantity * settings.finra_sell_share)
    return commission, exchange, clearing, regulatory


def cumulative_charge_delta(old_shares, old_notional, new_shares, new_notional, side, settings):
    """Only the incremental fee on a partial execution of the same order."""
    old = charges(side * old_shares, old_notional / old_shares, settings) if old_shares else (0.,) * 4
    total = old_shares + new_shares
    new = charges(side * total, (old_notional + new_notional) / total, settings)
    return tuple(b - a for a, b in zip(old, new))
