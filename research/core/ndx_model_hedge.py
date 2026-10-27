"""Fixed market weakness signals from raw stock return forecasts (pp units)."""
import numpy as np

SIGNALS = ['raw_mean','raw_basket','raw_breadth60','raw_agreement',
           'cal_mean','cal_basket','cal_breadth60','cal_agreement']


def validation_bias(predicted, realized, labelmask):
    """Equal-date mean forecast error, fitted on the prior-year fold only."""
    p,y,m = np.asarray(predicted),np.asarray(realized),np.asarray(labelmask,bool)
    if p.shape != y.shape or p.shape != m.shape or not np.all(m.sum(1)>0):
        raise ValueError('Invalid calibration shapes/coverage')
    if not np.isfinite(p[m]).all() or not np.isfinite(y[m]).all():
        raise ValueError('Nonfinite calibration observation')
    residual=np.where(m,p-y,0.)
    return float(np.mean(residual.sum(1)/m.sum(1)))


def forecast_signals(families, mask, basket, biases):
    """Mean/basket <0, >=60% negative breadth, or both family means <0.

    Each family first averages its fixed seeds' unranked percentage-point
    forecasts. Families then receive equal weight. No outcomes enter this call.
    Calibration subtracts one prior-year mean error per family/seed group/year.
    """
    mask=np.asarray(mask,bool);basket=np.asarray(basket,float)
    if len(families)!=2 or len(biases)!=2 or basket.shape!=mask.shape or np.any(mask.sum(1)==0):
        raise ValueError('Invalid signal inputs')
    if np.any(basket[~mask]!=0) or np.max(np.abs(basket.sum(1)-1))>1e-10:
        raise ValueError('Invalid proposed basket')
    raw=np.stack(families).astype(float)
    if raw.shape[1:] != mask.shape or not np.isfinite(raw[:,mask]).all():
        raise ValueError('Nonfinite eligible forecasts')
    flags, diagnostics={},{}
    for prefix,values in [('raw',raw),('cal',raw-np.asarray(biases)[:,None,None])]:
        combined=values.mean(0)
        family_means=np.where(mask[None],values,0.).sum(2)/mask.sum(1)[None]
        mean=family_means.mean(0)
        expected=np.where(basket>0,combined*basket,0.).sum(1)
        breadth=((combined<0)&mask).sum(1)/mask.sum(1)
        flags[prefix+'_mean']=mean<0
        flags[prefix+'_basket']=expected<0
        flags[prefix+'_breadth60']=breadth>=.60
        flags[prefix+'_agreement']=(family_means<0).all(0)
        diagnostics[prefix+'_mean_pp']=mean
        diagnostics[prefix+'_basket_pp']=expected
        diagnostics[prefix+'_negative_breadth']=breadth
        diagnostics[prefix+'_think_mean_pp']=family_means[0]
        diagnostics[prefix+'_dynamic_mean_pp']=family_means[1]
    return flags,diagnostics
