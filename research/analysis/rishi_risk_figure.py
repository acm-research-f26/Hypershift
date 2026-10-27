"""Render the fixed risk-account comparison; no new strategy calculations."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def drawdown_episode():
    root=Path('docs/rishi_mechanism_20261027/execution_batch8_risk');target=root/'2022_DRAWDOWN.png'
    if target.exists():raise FileExistsError(target)
    with np.load('results/RISHI_MECHANISM_RISK_20261027/PATHS_PORTABLE.npz',allow_pickle=False) as data:
        dates=pd.DatetimeIndex(data['dates']);use=dates.year==2022
        ids=data['ids'];returns=data['returns']
    fig,axes=plt.subplots(2,1,figsize=(11.5,7),sharex=True);fig.subplots_adjust(left=.09,right=.98,top=.85,bottom=.12,hspace=.12)
    for policy,label,color in [('full','Full exposure','#7b8490'),('fixed','Calibrated fixed','#b79147'),('vol20','20% volatility rule','#087e83')]:
        index=np.flatnonzero(ids==f'E/ten_seed/equal5/{policy}/modeled1')[0]
        wealth=np.cumprod(1+returns[use,index]);peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
        axes[0].plot(dates[use],100*wealth,label=label,color=color,linewidth=1.8)
        axes[1].plot(dates[use],wealth/peak-1,color=color,linewidth=1.8)
    axes[0].axhline(100,color='#555555',alpha=.5,linewidth=.7);axes[0].set_ylabel('Wealth from $100 initial NAV')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_ylabel('Drawdown from initial/prior peak')
    for ax in axes:
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,ncol=3,loc='lower left',fontsize=10)
    fig.suptitle('The prespecified 2022 drawdown episode',x=.09,y=.96,ha='left',fontsize=18,weight='bold')
    fig.text(.09,.895,'Ten-seed blend · original top five · modeled base fees · selected development history',fontsize=10,color='#444444')
    fig.text(.09,.035,'Reduced historical loss does not imply a hard drawdown bound; source execution and zero-interest cash assumptions retained.',fontsize=9,color='#444444')
    fig.savefig(target,dpi=170,facecolor='white');plt.close(fig)
    print(target.resolve())


def main():
    root=Path('docs/rishi_mechanism_20261027/execution_batch8_risk')
    target=root/'RISK_COMPARISON.png'
    if target.exists():raise FileExistsError(target)
    metrics=pd.read_csv(root/'ACCOUNT_METRICS.csv');annual=pd.read_csv(root/'CALENDAR_ATTRIBUTION.csv')
    colors={'full':'#7b8490','fixed':'#b79147','vol20':'#087e83'}
    labels={'full':'Full exposure','fixed':'Calibrated fixed','vol20':'20% volatility rule'}
    candidates=['E','B','D','REV5','Hlabel','EW','QQQ'];names=['Blend','Parent','Dynamic','Reversal','ID mean','EW','QQQ']
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.3));fig.subplots_adjust(left=.065,right=.98,bottom=.21,top=.79,wspace=.27)
    for offset,policy in zip([-.25,0,.25],['full','fixed','vol20']):
        values=[]
        for candidate in candidates:
            group='benchmark' if candidate in ['EW','QQQ'] else 'ten_seed'
            allocation={'EW':'eligible_weekly','QQQ':'one_etf'}.get(candidate,'equal5')
            row=metrics[(metrics.candidate==candidate)&(metrics.group==group)&(metrics.allocation==allocation)&(metrics.policy==policy)&(metrics.cost=='modeled1')].iloc[0]
            values.append(row.sharpe)
        axes[0].bar(np.arange(7)+offset,values,width=.24,color=colors[policy],label=labels[policy])
        key=f'E/ten_seed/equal5/{policy}/modeled1';rows=annual[annual.id==key].sort_values('year')
        axes[1].bar(np.arange(4)+offset,rows.calendar_return,width=.24,color=colors[policy])
    axes[0].set_xticks(np.arange(7),names,rotation=15);axes[0].set_ylabel('Net Sharpe (3% annual hurdle)')
    axes[0].set_title('Same exposure rule, different selection books',loc='left',fontsize=12,pad=12)
    axes[1].set_xticks(np.arange(4),['2022','2023','2024','2025']);axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel('Exact calendar-year net return');axes[1].set_title('Blend: less 2022 loss, less later upside',loc='left',fontsize=12,pad=12)
    for ax in axes:
        ax.axhline(0,color='#444444',linewidth=.7);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Exposure management on the saved forecasts',x=.065,y=.98,ha='left',fontsize=19,weight='bold')
    fig.text(.065,.91,'2022–2025 exploratory history · ten seeds / top five stocks; EW and QQQ are matched benchmark accounts',fontsize=10,color='#444444')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.06,.045),ncol=3,frameon=False,fontsize=10)
    fig.text(.065,.025,'All 14 primary-block paired Sharpe intervals include zero. Fixed exposure uses training-return calibration, with disclosed fallbacks.',fontsize=9,color='#444444')
    fig.savefig(target,dpi=170,facecolor='white');plt.close(fig)
    print(target.resolve())


if __name__=='__main__':main()
