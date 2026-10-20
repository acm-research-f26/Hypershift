"""Explain matched-loss similarities and uncertainty without training new models."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_engine import performance
from run_robust_research import bootstrap_difference
from analyze_ndx_loss_similarity import RUN, digest

OUT=Path('docs/nasdaq_monthly_pit/loss_similarity')


def table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])
def pct(x): return f'{100*x:.1f}%'


def build_section():
    if not (RUN/'AUDIT.json').exists(): return ''
    audit=json.loads((RUN/'AUDIT.json').read_text())
    assert audit['status']=='passed' and audit['matched_checkpoint_pairs']==20
    for source,expected in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items(): assert digest(source)==expected,source
    OUT.mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
    files=['AUDIT.json','INPUT_CODE_LOCK.json','agreement_by_period.csv','agreement_summary.csv',
        'validation_residuals.csv','marginal_seed_metrics.csv','marginal_seed_curves.csv',
        'gradient_examples.csv','ensemble_agreement_by_year.csv']
    for name in files: shutil.copyfile(RUN/name,OUT/name)
    fits=pd.read_csv(RUN/'matched_fits.csv')
    # The source field counts state tensors, including buffers, not parameters.
    fits.rename(columns={'parameter_count':'state_elements_including_buffers'}).to_csv(OUT/'matched_fits.csv',index=False)
    agreement=pd.read_csv(RUN/'agreement_summary.csv',dtype={'seed':str}).set_index('seed')
    seed_metrics=pd.read_csv(RUN/'marginal_seed_metrics.csv',dtype={'seed':str})
    paired=seed_metrics.pivot(index='seed',columns='model',values='annualized_return')
    paired['gap']=paired.think_mix-paired.think_mix_pseudohuber
    curve=pd.read_csv(RUN/'marginal_seed_curves.csv',dtype={'seed':str})
    a=curve[(curve.model=='think_mix')&(curve.seed=='ensemble')].net.to_numpy()
    b=curve[(curve.model=='think_mix_pseudohuber')&(curve.seed=='ensemble')].net.to_numpy()
    sharpe=bootstrap_difference(a,b)
    rng=np.random.default_rng(20261004); n=len(a); gaps=[]
    for _ in range(1000):
        starts=rng.integers(0,n,int(np.ceil(n/20)))
        idx=((starts[:,None]+np.arange(20))%n).ravel()[:n]
        gaps.append(float(np.expm1(np.log1p(a[idx]).sum()*252/n)-np.expm1(np.log1p(b[idx]).sum()*252/n)))
    cagr=dict(observed=float(paired.loc['ensemble','gap']),low=float(np.quantile(gaps,.025)),
        high=float(np.quantile(gaps,.975)),bootstrap_fraction_positive=float(np.mean(np.asarray(gaps)>0)))
    uncertainty=dict(MSE_minus_PH_CAGR=cagr,MSE_minus_PH_Sharpe=sharpe,
        conditional_on_fixed_selected_ensemble=True,accounts_for_training_seed_variation=False,
        multiple_comparisons_adjusted=False,random_chance_share_estimated=False)
    (OUT/'conditional_bootstrap.json').write_text(json.dumps(uncertainty,indent=2),encoding='utf-8')
    individuals=paired.drop(index='ensemble')
    residuals=pd.read_csv(RUN/'validation_residuals.csv')
    fraction=residuals[residuals.model=='think_mix'].fraction_abs_error_above_1pp.mean()
    same_epoch=int((fits.MSE_epoch==fits.PH_epoch).sum()); ens=agreement.loc['ensemble']
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    error=np.linspace(0,10,301)
    axes[0].plot(error,2*error,color='#0072b2',lw=2,label='MSE: influence grows with error')
    axes[0].plot(error,2*error/np.sqrt(1+error**2),color='#d55e00',lw=2,label='Pseudo-Huber: influence approaches a cap')
    axes[0].axvline(1,color='#777777',lw=.8,ls='--')
    axes[0].set(title='Delta = 1 percentage point is a material change',xlabel='Absolute forecast error (percentage points)',
        ylabel='Magnitude of per-observation loss derivative')
    axes[0].legend(fontsize=9,frameon=False); axes[0].grid(alpha=.2)
    order=['7','19','42','61','97','ensemble']; x=np.arange(6)
    axes[1].bar(x-.18,paired.loc[order,'think_mix']*100,width=.34,color='#0072b2',label='THINK + mixer: MSE')
    axes[1].bar(x+.18,paired.loc[order,'think_mix_pseudohuber']*100,width=.34,color='#d55e00',label='THINK + mixer: pseudo-Huber')
    axes[1].set_xticks(x,['7','19','42','61','97','Ensemble']); axes[1].set_xlabel('Training seed; all five ranks combined at right')
    axes[1].set_ylabel('Ranks-6–10 net CAGR (%)'); axes[1].set_title('The ensemble gap is not typical of matched seeds')
    axes[1].axhline(0,color='#444444',lw=.8); axes[1].grid(axis='y',alpha=.2); axes[1].legend(fontsize=9,frameon=False)
    fig.subplots_adjust(left=.07,right=.97,bottom=.21,top=.87,wspace=.25)
    fig.text(.07,.085,'Left: loss derivatives before batching, AdamW and global gradient clipping; these are not literal parameter-update sizes.',fontsize=10)
    fig.text(.07,.04,'Right: frozen 2022–2025 forecasts, same prices/orders/costs; no retraining. Ensembles form new portfolios rather than averaging seed returns.',fontsize=10)
    fig.savefig(OUT/'figures'/'loss_and_seed_sensitivity.png',dpi=165,bbox_inches='tight',facecolor='white'); plt.close(fig)
    gradient=pd.read_csv(RUN/'gradient_examples.csv')
    gradient_rows=[[f'{row.abs_error_pp:g} pp',f'{row.MSE_gradient:.2f}',f'{row.PH_gradient:.2f}',f'{100*row.gradient_ratio:.1f}%'] for row in gradient.itertuples()]
    seed_rows=[[f'Seed {seed}',pct(paired.loc[seed,'think_mix']),pct(paired.loc[seed,'think_mix_pseudohuber']),
        f'{100*paired.loc[seed,"gap"]:+.2f} pp'] for seed in ['7','19','42','61','97']]
    seed_rows.append(['Five-seed rank ensemble',pct(paired.loc['ensemble','think_mix']),pct(paired.loc['ensemble','think_mix_pseudohuber']),
        f'{100*paired.loc["ensemble","gap"]:+.2f} pp'])
    overlap_rows=[[policy,f'{100*ens[field]:.1f}%',f'{count*ens[field]:.2f} of {count} equal-weight positions']
        for policy,field,count in [('Top 5','top5_common_weight',5),('Top 10','top10_common_weight',10),('Ranks 6–10','ranks6_10_common_weight',5)]]
    year=pd.read_csv(RUN/'ensemble_agreement_by_year.csv')
    year_rows=[[int(row.year),f'{row.rank_correlation:.3f}',f'{100*row.ranks6_10_common_weight:.1f}%'] for row in year.itertuples()]
    example=fits[(fits.year==2025)&(fits.seed==7)].iloc[0]
    text=f'''### Why MSE and pseudo-Huber can produce very different portfolios

**Matched inspection: 20 same-year/same-seed checkpoint pairs; no new training.**
Both use exactly the same THINK+mixer architecture, inputs, target units,
training/validation/test splits, training-derived normalization and correlation
groups, seed rule, optimizer settings and deterministic GPU runtime. Recreating
their initialization gives identical starting parameters for all 20 pairs;
this checks the code/seed rule, since no initial-state checkpoint was saved.
All 20 final parameter sets differ, and {20-same_epoch}/20 select different
checkpoint epochs under the same prior-validation MSE rule. For 2025 seed 7,
MSE selected epoch {int(example.MSE_epoch)} and pseudo-Huber epoch {int(example.PH_epoch)}.
Their validation MSEs were close ({example.MSE_validation_MSE:.3f} and
{example.PH_validation_MSE:.3f}), which does not guarantee identical ordering at
a trading cutoff. The corrected selected-checkpoint validation exports were
used; the previously disclosed stale final-epoch validation arrays were not.

**Same architecture does not imply the same learned function.** The intended
loss change changes the optimization path and the resulting weights. Delta was
fixed at **1 percentage point**, not an extremely large value that would make
pseudo-Huber approximately MSE for most errors. At the selected checkpoints,
an average {100*fraction:.1f}% of MSE validation errors exceed that scale.
This describes validation residuals, not the residuals at every training step.

| Absolute prediction error | MSE loss derivative | Pseudo-Huber derivative | Relative influence |
| --- | --- | --- | --- |
'''+'\n'.join('| '+' | '.join(row)+' |' for row in gradient_rows)+f'''

These are derivatives of our implemented per-observation losses, before
averaging, AdamW and global norm clipping. Pseudo-Huber's derivative saturates
while MSE's grows with the error ([SciPy pseudo-Huber definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pseudo_huber.html)).
Thus the formula change substantially changes the relative treatment of large
errors, even though both losses have matching curvature near zero.

### Similar broad rankings can still mean different five-stock baskets

The two ensembles have mean same-date rank correlation {ens.rank_correlation:.3f}
across eligible stocks. However, their selected portfolios overlap much less:

{table(['Portfolio','Average common target weight','Equivalent common position count'],overlap_rows)}

Common weight sums the minimum allocation to each security in both portfolios,
then averages over the 201 signal dates. It handles fractional ties; the position
count is an equal-weight equivalent, not a claim that exactly two names always
overlap. Both models rank the same stock universe, but much of the small selected
basket differs. At roughly 20% per position, moving a few names across rank five
or rank ten can materially change portfolio returns.

{table(['Test year','Cross-model rank correlation','Ranks-6–10 common target weight'],year_rows)}

### Does the return difference mostly boil down to random chance?

**The results show sensitivity, but do not identify what fraction is chance.**
The large ensemble CAGR gap is not representative of matched individual seeds:

{table(['Matched fit','MSE ranks-6–10 CAGR','Pseudo-Huber CAGR','MSE minus pseudo-Huber'],seed_rows)}

MSE wins only {int((individuals.gap>0).sum())}/5 matched seed portfolios. Their median
paired difference is {100*individuals.gap.median():+.2f} percentage points, essentially
zero, compared with {100*paired.loc['ensemble','gap']:.2f} points for the ensemble.
Every pseudo-Huber individual-seed ranks-six-to-ten account has a positive CAGR,
although its rank ensemble is negative. The ensemble sorts average ranks and
creates a different portfolio, so constituent CAGRs cannot be averaged to
reconstruct it. Training-seed and cutoff sensitivity are evident; a consistent
MSE advantage has not been demonstrated.

![Loss influence and matched-seed portfolio differences](loss_similarity/figures/loss_and_seed_sensitivity.png)

For completeness, the **fixed ensemble** return difference survives an
exploratory paired 20-session block bootstrap: CAGR-gap interval
[{100*cagr['low']:+.1f}, {100*cagr['high']:+.1f}] percentage points and Sharpe-gap
interval [{sharpe['low']:+.3f}, {sharpe['high']:+.3f}]. Both exclude zero. It would be
incorrect to claim these particular intervals prove the gap is merely sampling
noise. However, they condition on already selected ensembles, omit training-seed
uncertainty, and are not adjusted for the many inspected models, losses, sizes
and rank buckets. They do not establish that MSE will beat pseudo-Huber on
unseen dates or across alternative training seeds.

The supported interpretation is: **a material loss change produced different
fitted weights and stock selections; a highly selective, seed-sensitive portfolio
rule amplified that difference into a large historical CAGR gap.** Market luck,
selection effects and concentration may influence its magnitude, but their
separate causal shares cannot be estimated from these experiments. The evidence
does not justify treating either loss as reliably superior or calling the two
trained models functionally identical.

Twelve individual/ensemble marginal-bucket accounts matched the frozen
production engine. Source checkpoint and forecast hashes were checked, and no
optimization steps or retraining occurred. This diagnostic does not independently
eliminate every data limitation documented in the parent study.

[Matched fits](loss_similarity/matched_fits.csv) · [Selection overlap](loss_similarity/agreement_summary.csv) ·
[Every signal date](loss_similarity/agreement_by_period.csv) · [Validation residual summaries](loss_similarity/validation_residuals.csv) ·
[Individual/ensemble marginal portfolios](loss_similarity/marginal_seed_metrics.csv) ·
[Conditional ensemble intervals](loss_similarity/conditional_bootstrap.json) · [Inspection audit](loss_similarity/AUDIT.json).
'''
    return text


def verify(text):
    audit=json.loads((RUN/'AUDIT.json').read_text())
    assert audit['status']=='passed' and audit['matched_checkpoint_pairs']==20 and audit['new_model_fits']==0
    for source,expected in json.loads((RUN/'INPUT_CODE_LOCK.json').read_text()).items(): assert digest(source)==expected,source
    verified=[]
    for name in ['AUDIT.json','INPUT_CODE_LOCK.json','agreement_by_period.csv','agreement_summary.csv',
        'validation_residuals.csv','marginal_seed_metrics.csv','marginal_seed_curves.csv',
        'gradient_examples.csv','ensemble_agreement_by_year.csv']:
        assert (RUN/name).read_bytes()==(OUT/name).read_bytes(),name
        verified.append(OUT/name)
    expected=pd.read_csv(RUN/'matched_fits.csv').rename(columns={'parameter_count':'state_elements_including_buffers'})
    pd.testing.assert_frame_equal(expected,pd.read_csv(OUT/'matched_fits.csv'))
    verified.append(OUT/'matched_fits.csv')
    metrics=pd.read_csv(OUT/'marginal_seed_metrics.csv',dtype={'seed':str}).set_index(['model','seed'])
    curves=pd.read_csv(OUT/'marginal_seed_curves.csv',dtype={'seed':str})
    assert len(metrics)==12
    for key,sub in curves.groupby(['model','seed']):
        assert len(sub)==1003 and sub.date.is_unique
        result=performance(sub.net)
        for field in ['annualized_return','sharpe','maximum_drawdown','cumulative_return']:
            np.testing.assert_allclose(result[field],metrics.loc[key,field],atol=1e-10)
    conditional=json.loads((OUT/'conditional_bootstrap.json').read_text())
    assert conditional['accounts_for_training_seed_variation'] is False
    assert conditional['multiple_comparisons_adjusted'] is False
    verified.append(OUT/'conditional_bootstrap.json')
    image=OUT/'figures'/'loss_and_seed_sensitivity.png'
    assert image.exists() and 'loss_similarity/figures/loss_and_seed_sensitivity.png' in text
    assert 'The results show sensitivity, but do not identify what fraction is chance.' in text
    verified.append(image)
    print('Verified matched-loss inspection: 20 checkpoint pairs, 12 marginal accounts and selection overlap')
    return verified
