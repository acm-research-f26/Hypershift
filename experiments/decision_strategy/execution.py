"""Sequential whole-share fills: no atomic stock-basket assumption."""
from dataclasses import dataclass, field
import numpy as np
from experiments.strategy_validation.fees import cumulative_charge_delta, charges


def exposures(q, marks, cash, sectors):
    dollars = q * marks
    nav = cash + dollars.sum()
    if nav <= 0:
        return dict(nav=nav, gross=np.inf, net=np.inf, stock=np.inf, sector=np.inf, error=np.inf)
    gross, net = abs(dollars).sum()/nav, dollars.sum()/nav
    stock = abs(dollars).max(initial=0)/nav
    sector = max((abs(dollars[sectors == g].sum())/nav for g in np.unique(sectors)), default=0.)
    error = max(0., gross-1., abs(net)-.0025, stock-.05, sector-.05)
    return dict(nav=float(nav), gross=float(gross), net=float(net), stock=float(stock), sector=float(sector), error=float(error))


def integer_targets(prices, sides, weights, dollars, nav, sectors, existing=None, capacity=None):
    """Enumerate decreasing feasible whole-share quantities, including all hedges."""
    prices, sides, weights = map(np.asarray, (prices, sides, weights))
    if not np.isfinite(prices).all() or (prices <= 0).any() or len(prices)<2:
        return np.zeros(len(prices), dtype=int)
    existing = np.zeros(len(prices), int) if existing is None else np.asarray(existing)
    capacity=np.full(len(prices),np.inf) if capacity is None else np.asarray(capacity)
    if (capacity<1).any():return np.zeros(len(prices),dtype=int)
    dollars=min(dollars,float(np.min(capacity*prices/weights)))
    # A first child must fit the hard net band on its own. This rejects a
    # nominally neutral but unstartable high-price basket before submission.
    net=float(existing@prices)
    if not any(abs(net+s*p)<=.00249*nav for s,p in zip(sides,prices)):
        return np.zeros(len(prices),dtype=int)
    maximum = int(np.floor(min(dollars,.0399*nav)/prices[0]))
    for lead in range(maximum,0,-1):
        budget = lead*prices[0]
        q = np.r_[lead*sides[0], np.rint(budget*weights[1:]/prices[1:])*sides[1:]].astype(int)
        if (q == 0).any() or (abs(q)>capacity).any() or abs(q@prices)>.001*nav:
            continue
        book=(existing+q)*prices
        if abs(book).max()>.04*nav or abs(book).sum()>.30*nav:
            continue
        if any(abs(book[sectors==g].sum())>.03*nav for g in np.unique(sectors)):
            continue
        # Prove a bounded sequential net path before promising a basket size.
        # Conservative failure is explicit; unrelated stocks are never netted
        # into a fictional simultaneous execution.
        remaining=abs(q).copy();net=float(existing@prices);band=.00249*nav
        feasible=True
        while remaining.any():
            candidates=np.flatnonzero((remaining>0)&(abs(net+np.sign(q)*prices)<=band))
            if not len(candidates):feasible=False;break
            k=candidates[np.argmin(abs(net+np.sign(q[candidates])*prices[candidates]))]
            net+=np.sign(q[k])*prices[k];remaining[k]-=1
        if not feasible:return np.zeros(len(prices),dtype=int)
        return q
    return np.zeros(len(prices),dtype=int)


def no_trade_targets(current,desired,marks,*,new_signal=False,required=False,band_dollars=50.):
    """A discretionary discrepancy band; required hedges/exits always bypass."""
    desired=np.asarray(desired).copy();current=np.asarray(current)
    if not new_signal and not required:
        small=abs((desired-current)*np.asarray(marks))<band_dollars
        desired[small]=current[small]
    return desired


def round_trip_cost(q, prices, entry_spreads, exit_spread_bps, horizon, settings):
    """Dollar estimate; exit spreads are past-only estimates, not realized exits."""
    q, prices = np.asarray(q), np.asarray(prices)
    notional = abs(q*prices)
    fees = sum(sum(charges(int(a),float(p),settings))+sum(charges(int(-a),float(p),settings)) for a,p in zip(q,prices))
    spread = float((abs(q)*np.asarray(entry_spreads)/2 + notional*np.asarray(exit_spread_bps)/20000).sum())
    borrow = float(notional[q<0].sum()*settings.borrow*horizon/(365*1440))
    buffer = float(notional.sum()*2*.5/10000)
    return dict(fees=float(fees),spread=spread,borrow=borrow,buffer=buffer,total=float(fees+spread+borrow+buffer))


@dataclass
class Book:
    n: int
    settings: object
    sectors: np.ndarray
    cash: float = 100000.
    q: np.ndarray = field(init=False)
    parents: dict = field(default_factory=dict)
    used_depth: dict = field(default_factory=dict)
    used_volume: dict = field(default_factory=dict)
    fills: list = field(default_factory=list)
    attempts: list = field(default_factory=list)

    def __post_init__(self):
        self.cash=self.settings.capital
        self.q=np.zeros(self.n,dtype=np.int64)

    def execute(self, minute, stamp, targets, marks, volume, quotes, parent, reason, stress='normal'):
        """One snapshot, sequential child fills. A missing hedge leaves real exposure.

        Every child is checked before submission against actual current holdings.
        Risk reductions may proceed from an already invalid mark if they reduce
        its excess. Snapshot order is an assumption; adverse order is explicit.
        """
        desired={int(j):int(v) for j,v in targets.items()}
        if reason=='rebalance':
            js=list(desired);target=no_trade_targets(self.q[js],[desired[j] for j in js],marks[js])
            desired=dict(zip(js,map(int,target)))
        available={};prices={};records={}
        for j,target in desired.items():
            delta=target-int(self.q[j])
            if not delta:
                continue
            side=int(np.sign(delta)); q=quotes.get(j) if quotes is not None else None
            good=bool(np.isfinite(marks[j]) and marks[j]>0)
            age=None; quote_stamp=None; depth=np.inf
            if quotes is not None:
                good &= q is not None
                if q is not None:
                    bid,ask,bs,az,quote_stamp=q; quote_stamp=int(quote_stamp);age=int(stamp)-quote_stamp
                    good &= bool(np.isfinite([bid,ask,bs,az]).all() and 0<bid<=ask and bs>0 and az>0 and 0<=age<=self.settings.quote_max_age_seconds*1e9)
                    depth=max(0.,(az if side>0 else bs)-self.used_depth.get((j,quote_stamp,side),0))
            cap=max(0,int(np.floor(volume[j]*self.settings.participation))) if np.isfinite(volume[j]) else 0
            cap=max(0,cap-self.used_volume.get((minute,j),0))
            if stress=='hedge_unavailable' and j!=next(iter(desired)) and reason=='entry':
                depth=0
            px=((q[1] if side>0 else q[0]) if quotes is not None and good else marks[j])
            px*=1+side*self.settings.adverse_bps/10000
            available[j]=min(abs(delta),cap,int(depth) if np.isfinite(depth) else cap) if good else 0
            prices[j]=float(px)
            records[j]=dict(minute=int(minute),arrival_ns=int(stamp),stock=j,parent=str(parent)+':'+str(j)+':'+str(side),reason=reason,
                intended_shares=delta,filled_shares=0,benchmark=float(marks[j]),quote_valid=bool(good) if quotes is not None else None,
                quote_age_ns=age,quote_timestamp_ns=quote_stamp,query_complete=quotes is None or j in quotes,
                intended_notional=float(abs(delta*marks[j])),fees=0.)
        # Greedy nearest-neutral next child; adverse variants exhaust one side
        # first, stopping at the same risk cap instead of assuming a hedge.
        while any(available.values()):
            before=exposures(self.q,marks,self.cash,self.sectors)
            choices=[]
            for j,count in available.items():
                if count<=0:continue
                side=int(np.sign(desired[j]-self.q[j]));px=prices[j]
                key=records[j]['parent'];oldqty,oldnot=self.parents.get(key,(0,0.))
                fees=cumulative_charge_delta(oldqty,oldnot,1,px,side,self.settings)
                proposed=self.q.copy();proposed[j]+=side
                post=exposures(proposed,marks,self.cash-side*px-sum(fees),self.sectors)
                reducing=reason!='entry' and before['error']>1e-10 and post['error']<before['error']-1e-12
                if post['error']<=1e-10 or reducing:
                    priority=-side if stress=='long_first' else side if stress=='short_first' else abs(post['net'])
                    choices.append((priority,j,side,fees,post))
            if not choices:break
            _,j,side,fees,post=min(choices,key=lambda x:(x[0],x[1]))
            px=prices[j];key=records[j]['parent'];oldqty,oldnot=self.parents.get(key,(0,0.))
            self.parents[key]=(oldqty+1,oldnot+px)
            self.cash-=side*px+sum(fees);self.q[j]+=side;available[j]-=1
            self.used_volume[(minute,j)]=self.used_volume.get((minute,j),0)+1
            if quotes is not None:
                dk=(j,records[j]['quote_timestamp_ns'],side)
                self.used_depth[dk]=self.used_depth.get(dk,0)+1
            records[j]['filled_shares']+=side;records[j]['fees']+=sum(fees)
            self.fills.append(dict(minute=int(minute),sequence=len(self.fills),stock=j,parent=key,shares=side,price=px,
                commission=float(fees[0]),exchange=float(fees[1]),clearing=float(fees[2]),regulatory=float(fees[3]),
                cash=float(self.cash),post_nav=post['nav'],post_net=post['net'],post_error=post['error'],reason=reason))
        self.attempts.extend(records.values())
        return records
