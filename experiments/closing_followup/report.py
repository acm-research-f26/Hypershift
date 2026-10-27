"""Standalone report and scientific plots from the complete trial ledger."""
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
    fig,axes=plt.subplots(1,3,figsize=(18,8),sharey=True,layout='constrained')
    ids=[c['id'] for c in CATALOG];y=np.arange(len(ids));colors=['#244f75','#bc6136','#6b8c42']
    for ax,scope,color in zip(axes,SCOPES,colors):
        primary=[next(r for r in rows if r['config']['id']==s and r['scope']==scope and r['scenario']=='primary') for s in ids]
        proxy=[next(r for r in rows if r['config']['id']==s and r['scope']==scope and r['scenario']=='zero_execution_cost_proxy') for s in ids]
        ax.barh(y-.18,[r['metrics']['total_return']*100 for r in primary],height=.34,color=color,label='Bid/ask + fees')
        ax.barh(y+.18,[r['metrics']['total_return']*100 for r in proxy],height=.34,color='#b5bec5',label='Zero execution-cost bar proxy')
        for bar,r in zip(ax.containers[0],primary):
            if not(r['metrics']['accounting_valid'] and r['metrics']['coverage_passed']):bar.set_hatch('///')
        ax.axvline(0,color='black',linewidth=.7);ax.set_title(scope);ax.set_xlabel('Total return (%)');ax.grid(axis='x',alpha=.2)
        ax.set_yticks(y,ids);ax.invert_yaxis();ax.legend(fontsize=8)
    fig.suptitle('Closing-momentum follow-up — hatched quote results fail accounting or coverage gates')
    chart=output/'comparison.png';fig.savefig(chart,dpi=150);plt.close(fig)
    def fmt(v,percent=False):
        if v is None:return '—'
        return f'{v*100:.3f}%' if percent else f'{v:.3f}'
    table=[]
    for r in rows:
        if r['scenario']!='primary':continue
        m=r['metrics'];table.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['config']['id'],r['scope'],fmt(m['total_return'],True),fmt(m['Sharpe']),fmt(m['quote_coverage'],True),m['accounting_valid'],m['active_sessions'],fmt(m['net_bps_per_traded_dollar'])])+'</tr>')
    image=base64.b64encode(chart.read_bytes()).decode()
    doc=f'''<!doctype html><html><meta charset="utf-8"><title>Closing momentum follow-up</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#172b3a}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #d7dfe5;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}p{{max-width:1100px;line-height:1.5}}</style>
<h1>Closing momentum follow-up</h1><p><b>Qualified candidates: {html.escape(', '.join(summary['qualified']) or 'none')}.</b> {summary['trials']} predetermined trials. All cash/NAV histories and recorded fills were independently recomputed. Execution validity and quote coverage are separate requirements.</p>
<p>First 60 earlier sessions fit normalizers, liquidity and the closing-return regression; excluded from performance selection. Five subsequent earlier/initial blocks determine the frozen shortlist. Later replay has prior research exposure and is exploratory. Reserved future sessions remain sealed.</p>
<img alt="Cohort return comparison" src="data:image/png;base64,{image}">
<h2>Primary quote execution</h2><table><tr><th>Strategy</th><th>Cohort</th><th>Return</th><th>Daily Sharpe</th><th>Quote coverage</th><th>Accounting valid</th><th>Active days</th><th>Net bp/traded dollar</th></tr>{''.join(table)}</table>
<h2>Implementation and assumptions</h2><p>All policies return explicit stock or ETF weights. Concentrated and participation variants make one decision in the final 30 minutes. Concentrated variants select three or five names per side from original S10 signals and a training-only liquid-stock filter. The cost hurdle uses a training-only expected closing-return regression against the observed spread, estimated round-trip fees and 1bp buffer. Participation requires first-hour relative volume of 1.25 and absolute early relative score of 1; the confirmed version additionally requires 65% peer sign persistence. ETFs use dollar-neutral SPY/QQQ or two strongest/weakest sector pairs.</p>
<p>Capital $100,000; whole shares; stock operating cap 4%, sector net 4%, gross 90%, hard limits 5%/5%/100%. Net-dollar band ±0.25% NAV with tighter target rounding. Prior 20-session volatility targeting starts with 20 cash sessions. At most three entry attempts and five closing attempts. Orders cross valid as-of bid/ask no more than five seconds old, bounded by displayed shares and 1% of previous completed minute volume. Paired batches reduce the leg that would create excessive net exposure, including liquidation. Missing execution is retained as a failure, never replaced with an assumed close fill.</p>
<p>Costs retain the previous frozen October 26, 2026 counterfactual schedule, including broker minimums, an assumed taker route and regulatory charges. Shorts assume availability and 3% annual borrowing. Commission-free retains other costs; zero-explicit-fee quotes retain spread and borrow; zero-execution-cost bar proxies retain borrow but assume no spread or fees. None models full order-book impact or guarantees broker fills. ETF sleeves share the same 5% single-instrument cap; SPY/QQQ are one hedge group, while sector ETFs have individual groups. Raw ETF sessions with unusually large opening discontinuities are excluded, a precaution rather than a point-in-time corporate-action archive.</p>
<p>The original S10 control uses the same signals with this round's balanced execution and entry-expiry policy. It is an implementation comparison. New signals and the execution correction cannot be combined into a claim that the original strategy was validated.</p>
<h2>Reproduction</h2><p>protocol.json freezes all choices and source hashes. fit.json records training-only fitting. decisions/ records what to buy and intended weights. trials/ contains minute NAV, orders, daily returns, volatility scales, and shadow daily returns. trial_ledger.csv includes all fees and proxy cases. ETF bars and quotes are reusable in etf_cache/; reuse/ binds the prior immutable stock quote cache. selection.json is frozen before later-cohort replay.</p></html>'''
    (output/'report.html').write_text(doc,encoding='utf-8')
    atomic(output/'report_receipt.json',dict(report_sha256=sha(output/'report.html'),chart_sha256=sha(chart)))
