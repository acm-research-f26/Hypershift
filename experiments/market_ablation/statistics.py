"""Dependent-session inference on saved forecasts; seeds are not market samples."""
import numpy as np
from scipy.stats import norm, chi2


def hac_covariance(values, bandwidth=None):
    x = np.asarray(values, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    if n < 2 or not np.isfinite(x).all():
        return None
    lag = min(n-1, max(1, int(np.floor(4*(n/100)**(2/9))))) if bandwidth is None else min(n-1, max(0, int(bandwidth)))
    centered = x-x.mean(axis=0)
    omega = centered.T@centered/n
    for k in range(1, lag+1):
        gamma = centered[k:].T@centered[:-k]/n
        omega += (1-k/(lag+1))*(gamma+gamma.T)
    return {"omega": omega, "mean_covariance": omega/n, "bandwidth": lag, "sessions": n}


def paired_inference(loss_difference, bootstrap_replicates=2000, seed=1003):
    difference = np.asarray(loss_difference, np.float64)
    difference = difference[np.isfinite(difference)]
    n = len(difference)
    if n < 5:
        return {"status": "undefined", "reason": "fewer_than_five_sessions"}
    base = hac_covariance(difference)
    l = base["bandwidth"]
    results = {}
    for lag in sorted(set([max(0, l//2), l, min(n-1, l*2)])):
        cov = hac_covariance(difference, lag)
        se = float(np.sqrt(max(cov["mean_covariance"][0, 0], 0)))
        z = float(difference.mean()/se) if se > 0 else None
        results[str(lag)] = {"mean": float(difference.mean()), "standard_error": se,
                             "z": z, "two_sided_p": float(2*norm.sf(abs(z))) if z is not None else None,
                             "ci95": [float(difference.mean()-1.96*se), float(difference.mean()+1.96*se)]}
    rng = np.random.default_rng(seed)
    bootstrap = {}
    for length in (5,10,20):
        draws = stationary_means(difference, bootstrap_replicates, length, rng)
        centered = draws-difference.mean()
        p = float((1+np.sum(np.abs(centered) >= abs(difference.mean())))/(bootstrap_replicates+1))
        bootstrap[str(length)] = {"ci95": np.quantile(draws, [.025,.975]).tolist(), "centered_two_sided_p": p,
                                   "replications": bootstrap_replicates, "block_length": length}
    return {"status": "ok", "sessions": n, "mean_model_minus_reference_loss": float(difference.mean()),
            "HAC": results, "primary_bandwidth": l, "bootstrap": bootstrap,
            "primary_p": results[str(l)]["two_sided_p"], "interpretation": "conditional_on_saved_training_runs",
            "dependence_diagnostics":dependence_diagnostics(difference)}


def dependence_diagnostics(values):
    x=np.asarray(values,float)
    centered=x-x.mean()
    variance=float(np.mean(centered**2))
    omega=float(hac_covariance(x)["omega"][0,0])
    return {"ACF":{str(k):float(np.mean(centered[k:]*centered[:-k])/variance) if variance>0 else None
                   for k in range(1,min(20,len(x)-1)+1)},
            "HAC_to_IID_variance_ratio":omega/variance if variance>0 else None,
            "effective_session_count_from_mean_variance":len(x)*variance/omega if omega>0 else None,
            "chronological_thirds":[{"sessions":len(block),"mean":float(block.mean()),"std":float(block.std())}
                                    for block in np.array_split(x,3)],
            "scope":"descriptive dependence/regime checks; stationarity is not established by these diagnostics"}


def skill_inference(model_loss,reference_loss,replicates=2000,seed=1003):
    """Delta method and paired stationary-bootstrap MSE skill on session means."""
    x=np.c_[model_loss,reference_loss].astype(float)
    x=x[np.isfinite(x).all(axis=1)]
    if len(x)<5 or x[:,1].mean()<=0:
        return {"status":"undefined","reason":"insufficient_support_or_zero_reference_loss"}
    mean=x.mean(axis=0)
    estimate=float(1-mean[0]/mean[1])
    jacobian=np.array([-1/mean[1],mean[0]/mean[1]**2])
    covariance=hac_covariance(x)
    se=float(np.sqrt(max(jacobian@covariance["mean_covariance"]@jacobian,0)))
    bootstrap={}
    for length in (5,10,20):
        draws=stationary_means(x,replicates,length,np.random.default_rng(seed+length))
        valid=draws[:,1]>0
        skills=1-draws[valid,0]/draws[valid,1]
        bootstrap[str(length)]={"ci95":np.quantile(skills,[.025,.975]).tolist() if len(skills) else None,
                                "valid_replications":int(valid.sum()),"replications":replicates}
    return {"status":"ok","MSE_skill":estimate,"delta_method_standard_error":se,
            "delta_method_ci95":[estimate-1.96*se,estimate+1.96*se],"Jacobian":jacobian.tolist(),
            "session_mean_loss_covariance":covariance["mean_covariance"].tolist(),"bandwidth":covariance["bandwidth"],
            "paired_stationary_bootstrap":bootstrap,"sessions":len(x),
            "estimand":"1 - mean(equal-weight session model MSE)/mean(equal-weight session reference MSE)"}


def gaussian_error_diagnostics(session_moments):
    """Golden-style low-dimensional Gaussian residual QML diagnostics.

    Input contains per-session counts and sums of e, e², e³, e⁴. The row OPG
    discrepancy is descriptive: financial rows are dependent and heterogeneous.
    Parameter uncertainty uses the HAC of complete-market session score sums.
    These are post-hoc residual parameters, never neural weight standard errors.
    """
    m=np.asarray(session_moments,float)
    m=m[m[:,0]>0]
    if len(m)<5 or not np.isfinite(m).all():
        return {"status":"undefined","reason":"insufficient_finite_session_moments"}
    total=m.sum(axis=0); n=total[0]; mu=total[1]/n
    variance=total[2]/n-mu**2
    if variance<=0:
        return {"status":"undefined","reason":"constant_forecast_error"}
    sigma=np.sqrt(variance)
    central=np.c_[m[:,0],m[:,1]-mu*m[:,0],m[:,2]-2*mu*m[:,1]+mu**2*m[:,0],
                  m[:,3]-3*mu*m[:,2]+3*mu**2*m[:,1]-mu**3*m[:,0],
                  m[:,4]-4*mu*m[:,3]+6*mu**2*m[:,2]-4*mu**3*m[:,1]+mu**4*m[:,0]]
    central[:,1:]/=sigma**np.arange(1,5)
    z=central.sum(axis=0)/n
    hessian=np.array([[1,z[1]],[z[1],.5*z[2]]])
    opg=np.array([[z[2],.5*(z[3]-z[1])],[.5*(z[3]-z[1]),.25*(1-2*z[2]+z[4])]])
    score=np.c_[-central[:,1],.5*(central[:,0]-central[:,2])]
    hac=hac_covariance(score)
    inverse=np.linalg.inv(hessian)
    normalized_covariance=inverse@(len(m)*hac["omega"])@inverse/n**2
    transform=np.diag([sigma,1.])
    covariance=transform@normalized_covariance@transform
    se=np.sqrt(np.maximum(np.diag(covariance),0))
    eigen=np.linalg.eigvalsh(opg)
    discrepancy={"status":"undefined_singular_OPG"}
    if np.all(eigen>np.finfo(float).eps*max(eigen.max(),1)):
        discrepancy={"status":"descriptive_only","trace_A_inverse_B":float(np.trace(np.linalg.solve(hessian,opg))),
                     "trace_B_inverse_A":float(np.trace(np.linalg.solve(opg,hessian))),
                     "log_determinant_A_inverse_B":float(np.linalg.slogdet(opg)[1]-np.linalg.slogdet(hessian)[1]),
                     "dimension":2,"IID_Gaussian_reference":{"trace_AB":2,"trace_BA":2,"logdet_AB":0}}
    return {"status":"ok","mean_error":float(mu),"error_variance":float(variance),
            "residual_skewness":float(z[3]),"residual_excess_kurtosis":float(z[4]-3),
            "parameters":["mean_error","log_error_variance"],"session_HAC_covariance":covariance.tolist(),
            "standard_errors":se.tolist(),"mean_error_ci95":[float(mu-1.96*se[0]),float(mu+1.96*se[0])],
            "log_variance_ci95":[float(np.log(variance)-1.96*se[1]),float(np.log(variance)+1.96*se[1])],
            "mean_zero_Wald_p":float(2*norm.sf(abs(mu/se[0]))) if se[0]>0 else None,
            "standardized_Hessian":hessian.tolist(),"standardized_row_OPG":opg.tolist(),
            "Hessian_condition_number":float(np.linalg.cond(hessian)),
            "OPG_condition_number":float(np.linalg.cond(opg)) if discrepancy["status"]=="descriptive_only" else None,
            "gradient_infinity_norm":float(np.max(np.abs(score.sum(axis=0)/n))),
            "information_matrix_discrepancy":discrepancy,"sessions":len(m),"observations":int(n),
            "bandwidth":hac["bandwidth"],"scope":"post_hoc_pooled_error_distribution; conditional_on_fixed_forecasts",
            "assumptions":"HAC requires a suitable weakly dependent session process; row information equality is an IID Gaussian descriptive reference"}


def stationary_means(values, replicates, length, rng):
    values = np.asarray(values)
    positions = rng.integers(len(values), size=replicates)
    total = np.zeros((replicates,)+values.shape[1:], np.float64)
    for _ in range(len(values)):
        total += values[positions]
        restart = rng.random(replicates) < 1/length
        positions = np.where(restart, rng.integers(len(values), size=replicates), (positions+1) % len(values))
    return total/len(values)


def holm(p_values):
    values = np.asarray(p_values, np.float64)
    result = np.full(len(values), np.nan)
    positions = np.flatnonzero(np.isfinite(values))
    order = positions[np.argsort(values[positions])]
    adjusted = np.maximum.accumulate(values[order]*(len(order)-np.arange(len(order))))
    result[order] = np.minimum(adjusted, 1)
    return result


def calibration(session_moments):
    """Two-dimensional calibration with HAC covariance of session score sums.

    Columns: count, sum p, sum p², sum y, sum py. The persistence predictor
    is identically zero, so it is never inserted as a singular regressor.
    """
    a = np.asarray(session_moments, np.float64)
    a = a[a[:, 0] > 0]
    if len(a) < 5:
        return {"status": "undefined", "reason": "insufficient_sessions"}
    total = a.sum(axis=0)
    xtx = np.array([[total[0],total[1]],[total[1],total[2]]])
    if np.linalg.matrix_rank(xtx) < 2:
        return {"status": "undefined", "reason": "constant_forecast"}
    inverse = np.linalg.inv(xtx)
    beta = inverse@total[3:5]
    g = np.c_[a[:,3]-beta[0]*a[:,0]-beta[1]*a[:,1],
              a[:,4]-beta[0]*a[:,1]-beta[1]*a[:,2]]
    hac = hac_covariance(g)
    covariance = inverse@(len(g)*hac["omega"])@inverse
    null = beta-np.array([0.,1.])
    statistic = float(null@np.linalg.pinv(covariance)@null)
    return {"status": "ok", "intercept": float(beta[0]), "slope": float(beta[1]),
            "covariance": covariance.tolist(), "standard_errors": np.sqrt(np.maximum(np.diag(covariance), 0)).tolist(),
            "wald_intercept_zero_slope_one": statistic, "wald_p": float(chi2.sf(statistic, 2)),
            "bandwidth": hac["bandwidth"], "scope": "low_dimensional_calibration_of_fixed_forecasts"}


def multiple_model_tests(persistence_loss, candidate_losses, replicates=2000, seed=1003):
    baseline, candidates = np.asarray(persistence_loss, float), np.asarray(candidate_losses, float)
    valid = np.isfinite(baseline) & np.isfinite(candidates).all(axis=1)
    baseline, candidates = baseline[valid], candidates[valid]
    if len(baseline) < 20 or not candidates.shape[1]:
        return {"status": "undefined", "reason": "insufficient_common_session_support"}
    improvement = baseline[:,None]-candidates
    # Studentized SPA requires nonzero sampling variance of loss differences.
    variable = np.var(improvement,axis=0) > np.finfo(float).eps*np.maximum(np.mean(improvement**2,axis=0),1e-30)
    spa_values = None
    if variable.any():
        from arch.bootstrap import SPA
        spa = SPA(baseline, candidates[:,variable], block_size=10, reps=replicates, bootstrap="stationary", seed=seed)
        spa.compute()
        spa_values = {key:float(value) if np.isfinite(value) else None for key,value in spa.pvalues.items()}
    statistic = float(max(0, improvement.mean(axis=0).max()))
    draws = stationary_means(improvement-improvement.mean(axis=0), replicates, 10, np.random.default_rng(seed))
    simulated = np.maximum(0, draws.max(axis=1))
    reality = float((1+np.sum(simulated >= statistic))/(replicates+1))
    return {"status": "ok", "sessions": len(baseline), "models": candidates.shape[1], "SPA_pvalues": spa_values,
            "SPA_variable_loss_models":np.flatnonzero(variable).tolist(),
            "SPA_excluded_constant_loss_models":np.flatnonzero(~variable).tolist(),
            "SPA_status":"ok" if spa_values is not None else "undefined_zero_loss_difference_variance",
            "Reality_Check_p": reality, "Reality_Check_statistic": statistic, "replications": replicates,
            "seed": seed, "block_length": 10}


def encompassing(session_xtx, session_xty):
    """Incremental model information beyond a matching temporal-only forecast."""
    xx, xy = np.asarray(session_xtx,float), np.asarray(session_xty,float)
    keep = xx[:,0,0]>0
    xx,xy = xx[keep],xy[keep]
    if len(xx)<5:
        return {"status":"undefined","reason":"insufficient_sessions"}
    total, response = xx.sum(axis=0),xy.sum(axis=0)
    scale = np.sqrt(np.maximum(np.diag(total)/total[0,0],1e-30))
    xx = xx/(scale[None,:,None]*scale[None,None,:])
    xy = xy/scale[None,:]
    total = xx.sum(axis=0)
    if np.linalg.matrix_rank(total)<3:
        return {"status":"undefined","reason":"collinear_fixed_forecasts"}
    inverse = np.linalg.inv(total)
    beta = inverse@xy.sum(axis=0)
    score = xy-np.einsum("sij,j->si",xx,beta)
    hac = hac_covariance(score)
    covariance = inverse@(len(score)*hac["omega"])@inverse
    se = np.sqrt(np.maximum(np.diag(covariance),0))
    statistic = float(beta[2]**2/covariance[2,2]) if covariance[2,2]>0 else None
    return {"status":"ok","coefficients":(beta/scale).tolist(),"standard_errors":(se/scale).tolist(),
            "incremental_model_coefficient_zero_Wald":statistic,
            "incremental_model_coefficient_zero_p":float(chi2.sf(statistic,1)) if statistic is not None else None,
            "reference":"matching_seed_temporal_only","columns":["intercept","temporal_forecast","model_forecast"],
            "sessions":len(score),"bandwidth":hac["bandwidth"],"scope":"fixed_forecasts_low_dimensional_encompassing_regression"}
