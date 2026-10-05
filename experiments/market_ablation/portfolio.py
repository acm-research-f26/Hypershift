"""Delayed, self-financing portfolio comparisons on archived raw prices."""
import numpy as np
import pandas as pd
from .storage import read_json, atomic_npz, atomic_json,reserve
from numba import njit


@njit(cache=True)
def execution_path(desired, visible, returns, sessions, duration, intraday, cost_bps, borrow,
                   capital_returns=None, liquidation_yields=None, corporate_costs_output=None, cash_flows_output=None):
    """Carry stock holdings, credit corporate cash separately, and fund costs."""
    t,n=desired.shape
    net,gross,turnover,long_exposure,short_exposure,costs,borrowing,stale_exposure=[np.zeros(t) for _ in range(8)]
    held=np.zeros(n)
    actual=np.zeros((t,n),np.float32)
    for i in range(t):
        if intraday and (i==0 or sessions[i]!=sessions[i-1]):
            held[:]=0
        fixed=np.where(visible[i],0.,held)
        fixed_long=np.maximum(fixed,0).sum()
        fixed_short=np.maximum(-fixed,0).sum()
        wanted_long=np.maximum(desired[i],0).sum()
        wanted_short=np.maximum(-desired[i],0).sum()
        cap_long=1. if wanted_short==0 else .5
        cap_short=.5 if wanted_short>0 else 0.
        q=1.
        dollar=np.zeros(n)
        for _ in range(5):
            long_scale=min(1.,max(0.,q*cap_long-fixed_long)/max(q*wanted_long,1e-30))
            short_scale=min(1.,max(0.,q*cap_short-fixed_short)/max(q*wanted_short,1e-30))
            for j in range(n):
                dollar[j]=fixed[j] if not visible[i,j] else q*desired[i,j]*(long_scale if desired[i,j]>=0 else short_scale)
            trade=np.abs(dollar-held).sum()
            q=1-cost_bps/10000*trade
        if q<=0:
            net[i:]=-1.
            break
        w=dollar/q
        actual[i]=w
        pnl=(w*returns[i]).sum()
        capital=returns[i] if capital_returns is None else capital_returns[i]
        liquidation_trade=0. if liquidation_yields is None else (np.abs(w)*liquidation_yields[i]).sum()
        corporate_cost=liquidation_trade*cost_bps/10000
        loan=np.maximum(-w,0).sum()*borrow*duration[i]
        closing=0.
        final=intraday and (i==t-1 or sessions[i+1]!=sessions[i])
        if final:
            closing=np.abs(w*(1+capital)).sum()*cost_bps/10000
        net[i]=q*(1+pnl-loan-closing-corporate_cost)-1
        gross[i]=pnl
        turnover[i]=trade+q*liquidation_trade+(np.abs(dollar*(1+capital)).sum() if final else 0.)
        long_exposure[i]=np.maximum(w,0).sum()
        short_exposure[i]=np.maximum(-w,0).sum()
        stale_exposure[i]=np.abs(np.where(visible[i],0.,w)).sum()
        costs[i]=1-q+q*(closing+corporate_cost)
        if corporate_costs_output is not None:
            corporate_costs_output[i]=q*corporate_cost
        if cash_flows_output is not None:
            cash_flows_output[i]=(dollar*(returns[i]-capital)).sum()
        borrowing[i]=q*loan
        denominator=1+net[i]
        held=dollar*np.maximum(1+capital,0)/denominator if denominator>0 and not final else np.zeros(n)
    return net,gross,turnover,long_exposure,short_exposure,costs,borrowing,stale_exposure,actual


def risk_metrics(returns, benchmark=None, wealth_path=None):
    r = np.asarray(returns, float)
    if not len(r) or not np.isfinite(r).all():
        return {"status": "undefined", "reason": "nonfinite_or_empty_returns"}
    wealth = np.r_[1.,np.cumprod(1+r)] if wealth_path is None else np.asarray(wealth_path)
    high = np.maximum.accumulate(wealth)
    drawdown = np.divide(wealth, high, out=np.ones_like(wealth), where=high>0)-1
    maxdd = float(-drawdown.min())
    duration, longest, recovery = 0, 0, []
    for value in drawdown:
        if value < -1e-12:
            duration += 1
            longest = max(longest, duration)
        elif duration:
            recovery.append(duration)
            duration = 0
    mean, std = float(r.mean()), float(r.std(ddof=1)) if len(r)>1 else 0.
    downside = float(np.sqrt(np.mean(np.minimum(r, 0)**2)))
    cagr = float(wealth[-1]**(252/len(r))-1) if wealth[-1]>0 else -1.
    result = {"status": "ok", "sessions": len(r), "total_return": float(wealth[-1]-1), "annual_return": cagr,
              "annual_volatility": std*np.sqrt(252), "Sharpe": mean/std*np.sqrt(252) if std>0 else None,
              "Sortino": mean/downside*np.sqrt(252) if downside>0 else None,
              "MaxDD": maxdd, "Calmar": cagr/maxdd if maxdd>0 else None, "IR": None,
              "max_drawdown_duration_observations": longest, "unrecovered_drawdown_observations": duration,
              "completed_recovery_durations": recovery, "risk_free_rate": 0., "minimum_acceptable_return": 0.,
              "annualization_sessions": 252, "positive_session_fraction": float(np.mean(r>0))}
    for confidence in (.95,.99):
        threshold = float(np.quantile(-r, confidence))
        tail = -r[-r >= threshold]
        result[f"VaR{int(confidence*100)}"] = threshold
        result[f"ES{int(confidence*100)}"] = float(tail.mean()) if len(tail) else None
    if benchmark is not None:
        active = r-np.asarray(benchmark)
        tracking = float(active.std(ddof=1)) if len(active)>1 else 0
        result["IR"] = float(active.mean()/tracking*np.sqrt(252)) if tracking>0 else None
        result["tracking_error"] = tracking*np.sqrt(252)
    return result


def desired_weights(prediction, available, kind):
    p = np.asarray(prediction)
    observed = np.asarray(available)
    n = observed.sum(axis=1)
    limit = np.maximum(1, np.ceil(n*.1).astype(int))
    informative=np.max(np.where(observed,p,-np.inf),axis=1)>np.min(np.where(observed,p,np.inf),axis=1)
    positive = observed & (p>0) if kind!="long_short" else observed & informative[:,None]
    order = np.argsort(np.where(positive, -p, np.inf), axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(p.shape[1]), order.shape), axis=1)
    long = positive & (ranks < limit[:,None])
    weights = long.astype(float)/np.maximum(long.sum(axis=1), 1)[:,None]
    if kind == "long_short":
        negative = observed & informative[:,None]
        order = np.argsort(np.where(negative, p, np.inf), axis=1, kind="stable")
        np.put_along_axis(ranks, order, np.broadcast_to(np.arange(p.shape[1]), order.shape), axis=1)
        short = negative & (ranks < limit[:,None])
        weights = .5*weights-.5*short/np.maximum(short.sum(axis=1), 1)[:,None]
    return weights


def daily_economic_returns(config, dataset, begin, end, *, include_flows=False):
    """Open-to-open total value; noncash entitlements liquidated at recipient open.

    Unavailable valuations are explicitly invalid; they never become zero flow.
    Informational identity changes are reconciled separately from economic events.
    """
    entry = np.array(dataset.bars[begin:end, :, 0], np.float64)
    exit = np.array(dataset.bars[begin+1:end+1, :, 0], np.float64)
    available = (entry>0) & (exit>0)
    payout = np.zeros_like(entry)
    liquidated = np.zeros_like(entry)
    ratio = np.ones_like(entry)
    unsupported = np.zeros_like(entry, bool)
    audit = []
    dates = pd.to_datetime(dataset.session_dates[dataset.sessions[begin+1:end+1]]).strftime("%Y-%m-%d")
    rows = {date:i for i,date in enumerate(dates)}
    stocks = {symbol:i for i,symbol in enumerate(dataset.symbols)}
    for action in read_json(config.root / "corporate_actions/records.json", []):
        kind = action["type"]
        day = action.get("ex_date", action.get("effective_date", action.get("process_date")))
        symbol = action.get("symbol", action.get("source_symbol", action.get("acquiree_symbol", action.get("old_symbol"))))
        if day not in rows or symbol not in stocks:
            continue
        row, column = rows[day], stocks[symbol]
        status = "accounted"
        if kind in ("forward_splits", "reverse_splits"):
            ratio[row,column] *= float(action["new_rate"])/float(action["old_rate"])
        elif kind == "cash_dividends":
            payout[row,column] += float(action["rate"])
        elif kind == "stock_dividends":
            # Alpaca rate is the resulting shares per old share (e.g. 1.3).
            ratio[row,column] *= float(action["rate"])
        elif kind == "spin_offs":
            recipient = action.get("new_symbol")
            data = read_json(config.root / "corporate_actions/recipient_prices" / f"{recipient}-{day}.json", {})
            bars = data.get("response", {}).get("bars", {}).get(recipient, [])
            if not bars:
                fallback=read_json(config.root/"corporate_actions/recipient_prices"/f"{recipient}-{day}.yahoo.json",{})
                if fallback.get("currency")=="USD":
                    bars=fallback.get("bars",[])
            candidates = [b for b in bars if pd.Timestamp(b["t"]).tz_convert("America/New_York").strftime("%Y-%m-%d") == day]
            if candidates:
                value=float(action["new_rate"])/float(action.get("source_rate", 1))*float(candidates[0]["o"])
                payout[row,column] += value
                liquidated[row,column] += value
            else:
                unsupported[row,column] = True
                status = "unvalued_recipient_entitlement"
        elif kind in ("stock_mergers", "stock_and_cash_mergers"):
            recipient = action.get("acquirer_symbol")
            if recipient in stocks:
                ratio[row,column] = 0
                value=float(action["acquirer_rate"])/float(action["acquiree_rate"])*exit[row,stocks[recipient]]
                payout[row,column] += value + float(action.get("cash_rate",0))
                liquidated[row,column] += value
                if not exit[row,stocks[recipient]]>0:
                    unsupported[row,column] = True
                    status = "unvalued_merger_recipient"
            elif not recipient and float(action.get("acquirer_rate",0)) == float(action.get("acquiree_rate",1)):
                status = "identity_reorganization_price_continuity"
            else:
                unsupported[row,column] = True
                status = "unvalued_merger_recipient"
        elif kind == "cash_mergers":
            ratio[row,column] = 0
            payout[row,column] += float(action["rate"])
        elif kind == "name_changes":
            status = "informational_identity_record"
        else:
            unsupported[row,column] = True
            status = "unsupported_action_type"
        audit.append({"id": action["id"], "type": kind, "symbol": symbol, "date": day, "status": status,
                      "share_ratio": float(ratio[row,column]), "cash_or_liquidated_entitlement": float(payout[row,column]),
                      "liquidated_non_cash_entitlement":float(liquidated[row,column])})
    available = (entry>0)&((exit>0)|((ratio==0)&(payout>0)))
    total = np.divide(exit*ratio+payout, entry, out=np.ones_like(entry), where=available)-1
    result=total, available, unsupported, audit
    if include_flows:
        capital=np.divide(exit*ratio,entry,out=np.ones_like(entry),where=available)-1
        liquidation=np.divide(liquidated,entry,out=np.zeros_like(entry),where=entry>0)
        return (*result,{"capital_returns":capital,"liquidation_yields":liquidation})
    return result


def portfolios(config, dataset, origins, predictions, active, directory):
    predictions = np.asarray(predictions, np.float32)
    if dataset.interval == "1d":
        # Close i is known strictly before session i+1 open.
        execution = origins+1
        keep = execution+1 < len(dataset.times)
        keep[keep] &= dataset.starts[execution[keep]+1] <= dataset.times[origins[-1]+1]
        execution, predictions, active = execution[keep], predictions[keep], active[keep]
        if not len(execution):
            return {"status":"undefined","reason":"no_executable_signals"}
        returns, quoted, unsupported, audit,flows = daily_economic_returns(config, dataset, int(execution[0]), int(execution[-1])+1,include_flows=True)
        selection = execution-execution[0]
        returns, quoted, unsupported = returns[selection], quoted[selection], unsupported[selection]
        capital_returns=flows["capital_returns"][selection]
        liquidation_yields=flows["liquidation_yields"][selection]
        duration = (dataset.starts[execution+1]-dataset.starts[execution])/1e9/(365*86400)
        sessions = dataset.sessions[execution]
    else:
        execution = origins+2
        keep = (execution < len(dataset.times))
        keep[keep] &= dataset.sessions[execution[keep]] == dataset.sessions[origins[keep]]
        execution, predictions, active = execution[keep], predictions[keep], active[keep]
        if not len(execution):
            return {"status":"undefined","reason":"no_executable_signals"}
        last = (execution+1 >= len(dataset.times))
        last[~last] = dataset.sessions[execution[~last]+1] != dataset.sessions[execution[~last]]
        entry = np.array(dataset.bars[execution, :, 0], np.float64)
        exit = np.array(dataset.bars[np.minimum(execution+1,len(dataset.times)-1), :, 0], np.float64)
        exit[last] = dataset.bars[execution[last], :, 3]
        quoted = (entry>0) & (exit>0)
        # Missing trade bars are not evidence of a zero asset price. Mark at the
        # last completed trade and leave holdings unchanged without a fresh open.
        start=max(0,int(execution[0])-1); end=int(execution[-1])+2
        raw=np.array(dataset.bars[start:min(end,len(dataset.times))],np.float64)
        previous_close=np.r_[np.zeros((1,250)),raw[:-1,:,3]]
        marked=pd.DataFrame(np.where(raw[:,:,0]>0,raw[:,:,0],np.where(previous_close>0,previous_close,np.nan))).ffill().fillna(0).to_numpy()
        marked_entry=marked[execution-start]
        marked_exit=marked[np.minimum(execution+1-start,len(marked)-1)]
        close=pd.DataFrame(np.where(raw[:,:,3]>0,raw[:,:,3],np.nan)).ffill().fillna(0).to_numpy()
        marked_exit[last]=close[execution[last]-start]
        returns = np.divide(marked_exit,marked_entry,out=np.ones_like(marked_exit),where=marked_entry>0)-1
        unsupported, audit = np.zeros_like(quoted), []
        capital_returns=liquidation_yields=None
        duration = dataset.expected[execution]/(365*1440)
        sessions = dataset.sessions[execution]
    if not len(execution):
        return {"status": "undefined", "reason": "no_executable_signals"}
    # Execution can use quotes visible at the open, never target eligibility.
    visible = dataset.bars[execution, :, 0] > 0
    baseline_weights = visible/np.maximum(visible.sum(axis=1),1)[:,None]
    benchmark_path=execution_path(baseline_weights.astype(np.float64),visible,returns,sessions,duration,dataset.interval!="1d",0.,0.,capital_returns,liquidation_yields)
    benchmark_step=benchmark_path[0]
    session_ids = np.unique(sessions)
    def session_returns(step):
        if np.any(step<=-1):
            return None
        logs = np.bincount(sessions, weights=np.log1p(step), minlength=len(dataset.session_dates))
        return np.expm1(logs[session_ids])
    benchmark = session_returns(benchmark_step)
    benchmark_complete = not np.any((np.abs(benchmark_path[-1])>0) & unsupported)
    if dataset.interval=="1d":
        benchmark_complete &= not np.any((np.abs(benchmark_path[-1])>0)&~quoted)
    del benchmark_path
    result = {"execution": "next_session_open" if dataset.interval == "1d" else "one_complete_bar_safety_lag",
              "corporate_action_audit": audit, "cost_basis": "per_dollar_one_way_turnover", "strategies": {},
              "benchmark_accounting_complete": benchmark_complete,
              "corporate_cash_policy":"ex_date_entitlements_credited_as_cash; payment_lag_funding_not_modeled",
              "noncash_entitlement_policy":"liquidate_at_recipient_open_and_charge_one_way_transaction_cost",
              "missing_trade_bar_policy":"mark_last_completed_trade_and_carry_holdings_until_fresh_open",
              "intraday_closing_policy":"liquidation_at_final_completed_trade_mark; OHLCV_fill_proxy"}
    for kind in ("long_only", "long_short", "persistence_cash"):
        weights = desired_weights(predictions if kind != "persistence_cash" else np.zeros_like(predictions), active & visible,
                                  "long_short" if kind == "long_short" else "long_only")
        sensitivities = {}
        for cost in (0,1,5,10):
            for borrow in (0,.03,.10):
                corporate_costs=np.zeros(len(execution))
                cash_flows=np.zeros(len(execution))
                net,gross,turnover,long_exposure,short_exposure,costs,borrowing,stale_exposure,actual=execution_path(
                    weights.astype(np.float64),visible,returns,sessions,duration,dataset.interval!="1d",float(cost),borrow,
                    capital_returns,liquidation_yields,corporate_costs,cash_flows)
                bad_quotes=np.any((np.abs(actual)>0)&~quoted,axis=1)
                bad_actions=np.any((np.abs(actual)>0)&unsupported,axis=1)
                daily = session_returns(net)
                metrics = risk_metrics(daily, benchmark) if daily is not None else {"status":"undefined","reason":"portfolio_ruin"}
                metrics.update(turnover=float(turnover.sum()), mean_gross_exposure=float((long_exposure+short_exposure).mean()),
                               mean_net_exposure=float((long_exposure-short_exposure).mean()), cost_bps=cost, borrow_annual=borrow,
                               transaction_cost_fraction=float(costs.sum()), borrowing_fraction=float(borrowing.sum()),
                               corporate_action_liquidation_cost_fraction=float(corporate_costs.sum()),
                               corporate_cash_flow_fraction=float(cash_flows.sum()),
                               stale_trade_mark_steps=int(bad_quotes.sum()), carried_unquoted_exposure=float(stale_exposure.sum()),
                               unvalued_action_steps=int(bad_actions.sum()))
                if not benchmark_complete:
                    metrics["IR"] = None
                    metrics["IR_undefined_reason"] = "benchmark_economic_valuation_incomplete"
                if bad_actions.any() or (dataset.interval=="1d" and bad_quotes.any()):
                    metrics["accounting_status"] = "incomplete_economic_valuation"
                    metrics["interpretation"] = "diagnostic_path_only; economic risk ratios unavailable on this support"
                    for name in ("Sharpe","Sortino","IR","Calmar","MaxDD","VaR95","VaR99","ES95","ES99"):
                        metrics[name] = None
                else:
                    metrics["accounting_status"] = "complete_on_executable_support"
                sensitivities[f"cost{cost}-borrow{borrow}"] = metrics
                reserve(config,net.nbytes+bad_quotes.nbytes+bad_actions.nbytes+corporate_costs.nbytes+cash_flows.nbytes)
                atomic_npz(directory/"sensitivities"/kind/f"cost{cost}-borrow{borrow}.npz",net_returns=net,
                           session_returns=daily if daily is not None else np.array([]),bad_quotes=bad_quotes,bad_actions=bad_actions,
                           corporate_action_liquidation_costs=corporate_costs,corporate_cash_flows=cash_flows)
                if cost==5 and borrow==.03:
                    wealth = np.r_[1.,np.cumprod(1+net)]
                    drawdown = wealth/np.maximum.accumulate(wealth)-1
                    reserve(config,sum(a.nbytes for a in (net,gross,wealth,drawdown,turnover,costs,borrowing,long_exposure,short_exposure,corporate_costs,cash_flows)))
                    atomic_npz(directory / f"{kind}.npz", execution=execution, sessions=sessions, session_ids=session_ids,
                               net_returns=net, gross_returns=gross, session_returns=daily if daily is not None else np.array([]),
                               benchmark_sessions=benchmark if benchmark is not None else np.array([]), wealth=wealth, drawdown=drawdown,
                               turnover=turnover, gross_exposure=long_exposure+short_exposure, net_exposure=long_exposure-short_exposure,
                               transaction_costs=costs,borrowing_costs=borrowing,carried_unquoted_exposure=stale_exposure,
                               corporate_action_liquidation_costs=corporate_costs,corporate_cash_flows=cash_flows,
                               bad_quotes=bad_quotes, bad_actions=bad_actions)
                    metrics["MaxDD_all_execution_marks"] = float(-drawdown.min()) if not bad_actions.any() and (dataset.interval!="1d" or not bad_quotes.any()) else None
        result["strategies"][kind] = sensitivities
    atomic_json(directory / "metrics.json", result)
    return result
