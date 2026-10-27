"""Export the complete ledger and cohort cost curves without hiding failed arms."""
import base64
import html
from pathlib import Path
import numpy as np
from experiments.strategy_search.storage import read,atomic,sha
from .config import CATALOG,SCOPES


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);summary=read(output/'summary.json');rows=summary['rows']
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    colors=plt.get_cmap('tab20')
    for col,scope in enumerate(SCOPES):
        for n,c in enumerate(CATALOG):
            arms=[r for r in rows if r['scope']==scope and r['config']['id']==c['id'] and r['mode']=='minute_proxy']
            arms.sort(key=lambda r:r['binding']['settings']['adverse_bps'])
            x=[r['binding']['settings']['adverse_bps'] for r in arms]
            for ax,metric,multiplier in [(axes[0,col],'total_return',100),(axes[1,col],'Sharpe',1)]:
                y=[r['metrics'][metric]*multiplier if r['metrics'][metric] is not None else np.nan for r in arms]
                ax.plot(x,y,color=colors(n),marker='.',label=c['id'],alpha=.85)
        for ax in axes[:,col]:ax.axhline(0,color='black',linewidth=.8);ax.grid(alpha=.2);ax.set_xlabel('Flat cost per side (bp)')
        axes[0,col].set_title(scope)
    axes[0,0].set_ylabel('Total return (%)');axes[1,0].set_ylabel('Daily Sharpe')
    axes[1,2].legend(fontsize=7,ncol=2,loc='best')
    fig.suptitle('Longer-horizon sparse strategies — minute-open diagnostics, not executable-price validation')
    chart=output/'cost_curves.png';fig.savefig(chart,dpi=150);plt.close(fig)
    def fmt(v,p=False):return '—' if v is None else f'{v*100:.3f}%' if p else f'{v:.3f}'
    def table(items,auction=False):
        result=[]
        for r in items:
            m=r['metrics'];strategy=r['strategy'] if auction else r['config']['id']
            scenario=f'{r["cost_bps"]:g}bp' if auction else r['scenario']
            fields=[strategy,r['scope'],scenario,fmt(m['total_return'],True),fmt(m['Sharpe']),
                fmt(m.get('quote_coverage'),True),m['accounting_valid'],m['active_sessions']]
            result.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in fields)+'</tr>')
        return '<table><tr>'+''.join('<th>'+x+'</th>' for x in ['Strategy','Cohort','Case','Return','Sharpe','Quote coverage','Valid','Active days'])+'</tr>'+''.join(result)+'</table>'
    image=base64.b64encode(chart.read_bytes()).decode();auction=read(output/'auction_benchmarks.json') if (output/'auction_benchmarks.json').exists() else None
    doc=f'''<!doctype html><html><meta charset="utf-8"><title>Longer-horizon strategy round</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#172b3a}}p{{line-height:1.5;max-width:1150px}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #d7dfe5;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}</style>
<h1>Longer-horizon sparse strategy round</h1><p><b>Qualified candidates: {html.escape(', '.join(summary['qualified']) or 'none')}.</b> {summary['trials']} main trials. Fourteen policies: direct lead–lag, pure two-hop, industry and sector peer signals, self-momentum, self-reversal, and a fixed shuffled-network null; each held 60 or 120 minutes. All 250 stocks can enter through causal eligibility. Historical data was already research-exposed; none is a fresh confirmatory test. Reserved future sessions remain sealed.</p>
<p>First 60 earlier sessions fit feature SD and the reused lag-five graph; their returns are excluded from reported evaluation. Five earlier/initial development blocks at 1bp determine the frozen two-candidate shortlist before later replay. Later data never chooses a candidate. The self policy is a control, not evidence of network value. A shuffled network is a null, not causal evidence.</p>
<img alt="Return and Sharpe against per-side cost" src="data:image/png;base64,{image}">
<h2>Quote replay for the frozen diagnostic shortlist</h2>{table([r for r in rows if r['mode']=='quotes'])}
<h2>Assumptions and constraints</h2><p>One sparse paired event per session, chosen from completed five-minute residual shocks scanned every 15 minutes from clock60 to clock210. Graph shock magnitude must reach two training SDs and the follower must have moved less than one quarter in the leader direction. At most three names per side, inverse trailing60-minute volatility, 4% operating stock/sector net caps, whole shares, $100,000 capital. Mandatory scheduled exits retry until filled; last-five-minute forced flattening remains. Hard gross 100%, stock 5%, sector net 5%, net-dollar band ±0.25% NAV. Prior20-session10%-annual-vol scaling, capped at one; first20 sessions cash in every cohort.</p>
<p>Minute-open cases assume flat per-side costs of 0, 0.5, 1, 2 or 5bp and retain 3% annual short borrow. They assume no spread beyond that proxy. Quote cases cross as-of SIP bid/ask, retain the frozen fee schedule, displayed depth and 1% of previous completed minute volume. Entry is two minutes after the signal bar starts, after that bar completes; delay-five is a separate case. Commission-free retains other charges. Age60 is sensitivity only; primary requires quotes at most five seconds old and 99% intended-notional coverage. Every intended request in both unscaled and scaled executions of all five cases was explicitly queried, including negative responses. Reused snapshots and new per-symbol snapshots are hashed locally. Neither quotes nor bar proxies establish real queue priority, short availability, full impact or future profitability.</p>
<h2>Closing-price benchmarks</h2><p>Two fixed baskets use first-hour relative momentum or the last60-minute relative momentum at 30 minutes before the close. Entry is a minute-open proxy two minutes later; exit assumes full fills at the largest reported condition6 market-center closing trade. Auction prints are terminal outcomes only. There is no historical imbalance signal, routing/depth/queue model or execution qualification. All selected missing auction outcomes remain invalid rather than being dropped. Price availability can never select a stock. The basket is held without interim risk rebalancing; any hard-limit drift is flagged invalid. Closing-print identity is a benchmark convention, not proof of a primary-listing auction fill. See <a href="https://docs.alpaca.markets/us/docs/market-data-faq">Alpaca's condition definitions</a>.</p>
{table(auction['rows'],True) if auction else '<p>Auction benchmarks not yet computed.</p>'}
<h2>Reproduction</h2><p>protocol.json binds source, dependencies, fees, partitions and protected-date manifest. fit.json binds train-only fitting. decisions/ records explicit symbols and weights. trials/ contains every minute NAV, actual/intended whole-share orders, daily returns, scales and unscaled order tapes. trial_ledger.csv covers every main case. quote_closure/ proves complete query acquisition; quotes/ and auctions/ preserve reusable raw responses. selection.json records the development choice including diagnostic fallback status. auction_benchmarks/ stores its separate daily data and trades. Cash, fee and NAV histories for both unscaled and scaled main trials were independently reconciled; reconciliation alone does not imply constraint or execution validity.</p></html>'''
    (output/'report.html').write_text(doc,encoding='utf-8')
    atomic(output/'report_receipt.json',dict(report_sha256=sha(output/'report.html'),chart_sha256=sha(chart)))
