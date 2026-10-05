"""Streaming saved-prediction evaluation, inference and scientific figures."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import numpy as np
import pandas as pd
from .config import manifest, folds
from .dataset import MarketDataset
from .portfolio import portfolios
from .statistics import (paired_inference, holm, calibration, multiple_model_tests, hac_covariance,
                         encompassing, skill_inference, gaussian_error_diagnostics)
from .storage import read_json, atomic_json, atomic_npz, event, digest_file,archive_execution_sources


def final_geometry(config,job):
    """Diagnose the selected learned memberships; reuse identical incidences."""
    from hashlib import sha256
    from hyperedges.common.storage import load_hyperedge_snapshot
    from hyperbolicity.diagnostics import summarize_structural_geometry
    path=config.root/"analysis"/job["id"]/"final_geometry.json"
    saved=read_json(path)
    if saved:
        return saved
    snapshot=load_hyperedge_snapshot(config.root/"fits"/job["id"]/"frozen_snapshot.json")
    incidence=np.concatenate([f.incidence.to_numpy(bool) for f in snapshot.families],axis=1) if snapshot.families else np.zeros((250,0),bool)
    unique,counts=np.unique(incidence,axis=1,return_counts=True)
    geometry_root=Path(__file__).resolve().parents[2]/"hyperbolicity"
    source="".join(digest_file(geometry_root/name) for name in ("core.py","diagnostics.py"))
    identity=sha256(unique.tobytes()+counts.tobytes()+source.encode()).hexdigest()
    cache=config.root/"geometry/final-cache"/f"{identity}.json"
    if not cache.exists():
        value=summarize_structural_geometry(snapshot,seed=config.constructor_seed,include_bipartite=False)
        atomic_json(cache,value)
    value=read_json(cache)
    row={"interval":job["interval"],"fold":job["fold"],"variant":job["variant"]["name"],"seed":job["seed"],
         "cache":str(cache.relative_to(config.root)),"incidence_identity":identity,
         "edges":value["edges"],"coverage":value["covered_fraction"],"duplicate_edges":value["duplicate_edges"],
         "membership_stage":"chosen forecasting checkpoint; learned groups frozen after training",
         "distance_definition":"unweighted stock-node s-walk, s distinct shared edges, deduplicated incidence",
         "projections":{}}
    for s,graph in value["projections"].items():
        largest=max(graph["components"],key=lambda c:c["points"])
        row["projections"][s]={"delta_hg_global":graph["global_delta"],"global_status":graph["status"],
            "delta_hg_largest_component":largest["delta"],"normalized_delta_hg_largest_component":largest["relative_delta"],
            "largest_component_diameter":largest["diameter"],"largest_component_fraction":graph["largest_component_fraction"],
            "connected_pair_coverage":graph["connected_pair_coverage"],"isolates":graph["isolates"],
            "max_exact_component_delta":max(c["delta"] for c in graph["components"]),
            "components":len(graph["components"]),"exact":True}
    atomic_json(path,row)
    return row


def statistical_catalog(config):
    book=Path(__file__).resolve().parents[2]/"local-notes/golden_rm_statistical_machine_learning_a_unified_framework (1) (1).pdf"
    atomic_json(config.root/"report/statistical_catalog.json",{
        "source":{"author":"Richard M. Golden","title":"Statistical Machine Learning: A Unified Framework",
                  "local_file":str(book),"sha256":digest_file(book) if book.exists() else None},
        "implemented":[{"reference":"Chapter 14, sections 14.2.1-14.2.2","analysis":"2000 paired bootstrap replications",
                        "adaptation":"stationary blocks of complete market sessions; no IID resampling of stock-minute rows"},
                       {"reference":"Chapter 15, Theorem 15.2.1 / Recipe Box 15.1","analysis":"sandwich covariance for residual/calibration/encompassing parameters",
                        "adaptation":"HAC of session score sums; finite-dimensional post-hoc fixed-forecast models"},
                       {"reference":"Chapter 15, Theorem 15.2.2 / Recipe Box 15.2","analysis":"delta-method MSE skill confidence interval and matrix/gradient diagnostics"},
                       {"reference":"Chapter 15, sections 15.3-15.4","analysis":"confidence intervals and supported low-dimensional Wald tests"},
                       {"reference":"Chapter 16, section 16.3.2","analysis":"trace/trace-inverse/logdet Hessian-OPG discrepancies for a pooled Gaussian residual working model",
                        "adaptation":"descriptive IID-Gaussian reference; no formal neural likelihood misspecification p-value"}],
        "scope":"conditional on saved forecast fits; seed dispersion separated from market-history uncertainty",
        "assumptions":"weakly dependent session process for HAC/block bootstrap; chronological thirds and ACF diagnostics do not prove stationarity",
        "neural_parameter_inference":"unsupported: nonsmooth memberships, regularization and nonidentifiability prevent applying the textbook IID Hessian theorem to all network weights"})


def load_predictions(config, dataset, job):
    path = config.root / "fits" / job["id"]
    values, positions = [], []
    for file in sorted((path / "predictions").glob("*.npz")):
        with np.load(file) as saved:
            positions.append(saved["origins"])
            values.append(saved["prediction"])
    if not values:
        raise ValueError(f"No archived forecasts for {job['id']}")
    return np.concatenate(positions), np.concatenate(values)


def analysis_identity(config,dataset,job):
    """Reuse analysis only while its numerical recipe and saved inputs agree."""
    from hashlib import sha256
    import inspect
    module=Path(__file__).resolve().parent
    source=inspect.getsource(forecast_statistics)+inspect.getsource(analysis_identity)
    recipe={name:digest_file(module/name) for name in ("statistics.py","portfolio.py","dataset.py")}
    files=[config.root/"fits"/job["id"]/"result.json",config.root/"normalization"/dataset.interval/f"fold-{job['fold']}.npz",
           dataset.root/"bars.npy",dataset.root/"starts.npy",dataset.root/"sessions.npy",dataset.root/"expected.npy",
           dataset.root/"session_dates.npy",config.root/"dataset/metadata.json"]
    directories=[config.root/"fits"/job["id"]/"predictions",
                 config.root/"shared"/dataset.interval/f"fold-{job['fold']}"/"test",
                 config.root/"fits"/dataset.interval/f"fold-{job['fold']}"/"T"/f"seed-{job['seed']}"/"predictions"]
    for directory in directories:
        files.extend(sorted(directory.glob("*.npz")))
    inputs={str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns] if p.exists() else None for p in files}
    economics=[Path(config.gics),config.root/"corporate_actions/records.json",
               *sorted((config.root/"corporate_actions/recipient_prices").glob("*.json"))]
    economic_hashes={str(p.resolve()):digest_file(p) if p.exists() else None for p in economics}
    value={"configuration":config.identity,"job":job["id"],"sources":recipe,"inputs":inputs,"economic_inputs":economic_hashes}
    return sha256(source.encode()+json.dumps(value,sort_keys=True).encode()).hexdigest()


def forecast_statistics(config, dataset, job):
    directory = config.root / "analysis" / job["id"]
    identity=analysis_identity(config,dataset,job)
    saved = read_json(directory / "forecast_metrics.json")
    if saved and saved.get("analysis_identity")==identity:
        return saved
    fit_directory = config.root / "fits" / job["id"]
    scale = dataset.moments(folds()[job["fold"]])["train_price_naive_MAE"]
    stock = np.zeros((250, 11), np.float64)
    session = np.zeros((len(dataset.session_dates), 13), np.float64)
    session_xx=np.zeros((len(session),3,3),float)
    session_xy=np.zeros((len(session),3),float)
    error_moments=np.zeros((len(session),5),float)
    persistence_direction_count=0
    ic_sum, ic_count, ic_rank_sum = 0., 0, 0.
    overlay_column=dataset.symbols.index("MSFT") if "MSFT" in dataset.symbols else 0
    hist_actual, hist_pred, hist_previous = [], [], []
    origins_all, predictions_all, active_all = [], [], []
    for path in sorted((fit_directory / "predictions").glob("*.npz")):
        with np.load(path) as predictions:
            origin = predictions["origins"]
            p = predictions["prediction"].astype(np.float64)
        common_path = config.root / "shared" / dataset.interval / f"fold-{job['fold']}" / "test" / path.name
        with np.load(common_path) as labels:
            if not np.array_equal(origin, labels["origins"]):
                raise ValueError("Model and persistence supports differ")
            target, valid = labels["target"].astype(np.float64), labels["mask"]
            current, actual = labels["current_price"].astype(np.float64), labels["actual_price"].astype(np.float64)
            active = labels["active"]
        predicted_price = current*np.exp(np.clip(p,-30,30))
        error, price_error = p-target, predicted_price-actual
        naive_error = current-actual
        predicted_simple, actual_simple = np.expm1(np.clip(p,-30,30)), np.expm1(target)
        common = lambda a: np.where(valid, a, 0)
        persistence_direction_count += int(np.sum(valid & (target==0)))
        values = [valid, common(error**2), common(np.abs(error)), common(error), common(target**2),
                  common(price_error**2), common(np.abs(price_error)), common(price_error), common(naive_error**2),
                  common(np.abs(naive_error)), common((np.sign(p)==np.sign(target)).astype(float))]
        stock += np.stack([v.sum(axis=0) for v in values],axis=1)
        sid = dataset.sessions[origin]
        error_powers=[valid,common(error),common(error**2),common(error**3),common(error**4)]
        for column,values in enumerate(error_powers):
            error_moments[:,column]+=np.bincount(sid,weights=values.sum(axis=1),minlength=len(session))
        if job["variant"]["name"] != "T":
            reference_path=config.root/"fits"/dataset.interval/f"fold-{job['fold']}"/"T"/f"seed-{job['seed']}"/"predictions"/path.name
            with np.load(reference_path) as reference:
                if not np.array_equal(reference["origins"],origin):
                    raise ValueError("Encompassing reference support changed")
                temporal=reference["prediction"]
            design=[np.ones_like(p),temporal,p]
            for i in range(3):
                session_xy[:,i]+=np.bincount(sid,weights=common(design[i]*target).sum(axis=1),minlength=len(session))
                for j in range(3):
                    session_xx[:,i,j]+=np.bincount(sid,weights=common(design[i]*design[j]).sum(axis=1),minlength=len(session))
        moments = [valid.sum(axis=1),common(p).sum(axis=1),common(p*p).sum(axis=1),
                   common(target).sum(axis=1),common(p*target).sum(axis=1),common(error**2).sum(axis=1),
                   common(target**2).sum(axis=1),common(price_error**2).sum(axis=1),common(naive_error**2).sum(axis=1),
                   common(np.abs(error)).sum(axis=1),common(np.abs(target)).sum(axis=1),
                   common((predicted_simple-actual_simple)**2).sum(axis=1),common(actual_simple**2).sum(axis=1)]
        for column, value in enumerate(moments):
            session[:,column] += np.bincount(sid,weights=value,minlength=len(session))
        # Cross-sectional Pearson IC and rank IC, computed on the same observed stocks.
        counts = valid.sum(axis=1)
        mp = common(p).sum(axis=1)/np.maximum(counts,1)
        my = common(target).sum(axis=1)/np.maximum(counts,1)
        cp, cy = common(p-mp[:,None]), common(target-my[:,None])
        den = np.sqrt((cp*cp).sum(axis=1)*(cy*cy).sum(axis=1))
        eligible = (counts>=3)&(den>0)
        ic_sum += float(np.divide((cp*cy).sum(axis=1),den,out=np.zeros(len(den)),where=eligible).sum())
        ic_count += int(eligible.sum())
        from scipy.stats import rankdata
        rp = rankdata(np.where(valid,p,np.nan),axis=1,nan_policy="omit")
        ry = rankdata(np.where(valid,target,np.nan),axis=1,nan_policy="omit")
        rp = np.where(valid,rp-(counts[:,None]+1)/2,0)
        ry = np.where(valid,ry-(counts[:,None]+1)/2,0)
        denr = np.sqrt(np.nansum(rp*rp,axis=1)*np.nansum(ry*ry,axis=1))
        ic_rank_sum += float(np.divide(np.nansum(rp*ry,axis=1),denr,out=np.zeros(len(denr)),where=eligible&(denr>0)).sum())
        # Small overlay archive is a view of exact saved values, not a resampled forecast.
        observed=np.flatnonzero(valid[:,overlay_column])
        chosen=observed[::max(1,len(observed)//300)]
        hist_actual.append(actual[chosen,overlay_column]); hist_pred.append(predicted_price[chosen,overlay_column]); hist_previous.append(current[chosen,overlay_column])
        origins_all.append(origin); predictions_all.append(p.astype(np.float32)); active_all.append(active)
    total = stock.sum(axis=0)
    count = total[0]
    rows = []
    for symbol, row in zip(dataset.symbols, stock):
        n = row[0]
        rows.append({"symbol":symbol,"count":int(n),"log_return_MSE": row[1]/n if n else None,
                     "log_return_MAE":row[2]/n if n else None,"log_return_bias":row[3]/n if n else None,
                     "persistence_log_return_MSE":row[4]/n if n else None,
                     "log_return_skill":1-row[1]/row[4] if row[4]>0 else None,
                     "price_MSE":row[5]/n if n else None,"price_MAE":row[6]/n if n else None,
                     "price_bias":row[7]/n if n else None,"price_skill":1-row[5]/row[8] if row[8]>0 else None,
                     "price_MASE":row[6]/n/scale[dataset.symbols.index(symbol)] if n and scale[dataset.symbols.index(symbol)]>0 else None,
                     "relative_price_MAE":row[6]/row[9] if row[9]>0 else None,"direction_accuracy":row[10]/n if n else None})
    frame = pd.DataFrame(rows)
    directory.mkdir(parents=True,exist_ok=True)
    frame.to_csv(directory / "by_stock.csv",index=False)
    classifications = pd.read_csv(config.gics,dtype=str).set_index("node_id")
    for level in ("sector","industry_group"):
        grouped = frame.join(classifications[[level]],on="symbol").groupby(level,dropna=False)
        summary = grouped[["log_return_MSE","log_return_skill","price_MSE","price_skill","direction_accuracy"]].mean()
        summary.to_csv(directory / f"by_{level}.csv")
    support = session[:,0]>0
    calibration_result = calibration(session[support,:5])
    result = {"job":job["id"],"observations":int(count),"sessions":int(support.sum()),
              "price_MSE":float(total[5]/count),"price_RMSE":float(np.sqrt(total[5]/count)),
              "price_MAE":float(total[6]/count),"price_bias":float(total[7]/count),
              "persistence_price_MSE":float(total[8]/count),"persistence_price_MAE":float(total[9]/count),
              "price_skill":float(1-total[5]/total[8]) if total[8]>0 else None,
              "price_MASE":float(frame.price_MASE.dropna().mean()) if frame.price_MASE.notna().any() else None,
              "relative_price_MAE":float(total[6]/total[9]) if total[9]>0 else None,
              "log_return_MSE":float(total[1]/count),"log_return_MAE":float(total[2]/count),
              "log_return_bias":float(total[3]/count),"persistence_log_return_MSE":float(total[4]/count),
              "log_return_skill":float(1-total[1]/total[4]) if total[4]>0 else None,
              "out_of_sample_R2_vs_persistence":float(1-total[5]/total[8]) if total[8]>0 else None,
              "log_return_out_of_sample_R2_vs_persistence":float(1-total[1]/total[4]) if total[4]>0 else None,
              "simple_return_MSE":float(session[:,11].sum()/count),
              "simple_return_skill":float(1-session[:,11].sum()/session[:,12].sum()) if session[:,12].sum()>0 else None,
              "direction_accuracy":float(total[10]/count),"neutral_forecast_rule":"zero has sign zero",
              "persistence_direction_accuracy":float(persistence_direction_count/count),
              "persistence_IC":None,"persistence_IC_reason":"constant_zero_return_forecast",
              "IC":ic_sum/ic_count if ic_count else None,"rank_IC":ic_rank_sum/ic_count if ic_count else None,
              "calibration":calibration_result,"encompassing":encompassing(session_xx,session_xy) if job["variant"]["name"]!="T" else {"status":"reference_model"},
              "Gaussian_error_diagnostics":{"error_scale":"log_return",**gaussian_error_diagnostics(error_moments)},
              "market_uncertainty_unit":"whole_market_session",
              "training_seed":job["seed"],"scope":job["scope"],
              "forecast_coverage":float(count/(sum(len(x) for x in origins_all)*250)),
              "price_exp_reconstruction_clip":30,"analysis_identity":identity}
    atomic_npz(directory / "session_moments.npz", moments=session, session_dates=dataset.session_dates)
    atomic_npz(directory / "encompassing_contributions.npz",xtx=session_xx,xty=session_xy)
    atomic_npz(directory / "error_distribution_contributions.npz",moments=error_moments)
    atomic_npz(directory / "overlay.npz", actual=np.concatenate(hist_actual), model=np.concatenate(hist_pred), persistence=np.concatenate(hist_previous),
               symbol=np.array(dataset.symbols[overlay_column]))
    portfolios(config,dataset,np.concatenate(origins_all),np.concatenate(predictions_all),np.concatenate(active_all),directory / "portfolios")
    atomic_json(directory / "forecast_metrics.json",result)
    return result


def comparisons(config, jobs):
    grouped = defaultdict(list)
    for job in jobs:
        grouped[(job["interval"],job["fold"],job["variant"]["name"])].append(job)
    result = []
    for (interval,fold,name), group in grouped.items():
        seeds = [j["seed"] for j in group]
        matrices = [np.load(config.root / "analysis" / j["id"] / "session_moments.npz")["moments"] for j in group]
        common = np.all([m[:,0]>0 for m in matrices],axis=0)
        model = np.mean([m[:,7]/np.maximum(m[:,0],1) for m in matrices],axis=0)
        baseline = matrices[0][:,8]/np.maximum(matrices[0][:,0],1)
        model_return=np.mean([m[:,5]/np.maximum(m[:,0],1) for m in matrices],axis=0)
        baseline_return=matrices[0][:,6]/np.maximum(matrices[0][:,0],1)
        inference = paired_inference((model-baseline)[common],config.bootstrap_replicates,config.constructor_seed)
        record = {"interval":interval,"fold":fold,"variant":name,"seeds":seeds,"scope":group[0]["scope"],
                  "primary_metric":"raw_price_MSE","persistence_comparison":inference,
                  "log_return_persistence_comparison":paired_inference((model_return-baseline_return)[common],config.bootstrap_replicates,config.constructor_seed),
                  "seed_MSEs":[float((m[:,7].sum()/m[:,0].sum())) for m in matrices],
                  "seed_log_return_MSEs":[float((m[:,5].sum()/m[:,0].sum())) for m in matrices]}
        record["persistence_skill_inference"]=skill_inference(model[common],baseline[common],config.bootstrap_replicates,config.constructor_seed)
        record["log_return_skill_inference"]=skill_inference(model_return[common],baseline_return[common],config.bootstrap_replicates,config.constructor_seed)
        record["training_seed_variability"]={"MSE_std":float(np.std(record["seed_MSEs"],ddof=1)) if len(seeds)>1 else None,
                                             "MSE_range":[min(record["seed_MSEs"]),max(record["seed_MSEs"])],
                                             "interpretation":"training randomness on the same market history; distinct from market-session sampling uncertainty"}
        if group[0]["scope"] == "latest_fold_contrast":
            variant = group[0]["variant"]
            if variant["control"] in ("frozen","rewire","uniform"):
                reference = "A" if variant["control"]=="frozen" else "frozen-reference"
            elif variant["control"] in ("window",):
                reference = "A"
            elif variant["control"] in ("cover_graph","no_family_norm"):
                reference = name.split("-",1)[0]
            else:
                reference = name.split("-",1)[0]
            reference_jobs = [j for j in grouped[(interval,fold,reference)] if j["seed"] in seeds]
            if sorted(j["seed"] for j in reference_jobs) != sorted(seeds):
                raise ValueError("Contrast reference seeds do not match")
            reference_matrices = [np.load(config.root / "analysis" / j["id"] / "session_moments.npz")["moments"] for j in reference_jobs]
            matching = common & np.all([m[:,0]>0 for m in reference_matrices],axis=0)
            ref = np.mean([m[:,7]/np.maximum(m[:,0],1) for m in reference_matrices],axis=0)
            record.update(reference=reference,matched_seed_comparison=paired_inference((model-ref)[matching],config.bootstrap_replicates,config.constructor_seed))
            ref_return=np.mean([m[:,5]/np.maximum(m[:,0],1) for m in reference_matrices],axis=0)
            record["matched_seed_log_return_comparison"]=paired_inference((model_return-ref_return)[matching],config.bootstrap_replicates,config.constructor_seed)
        result.append(record)
    for interval in dict.fromkeys(j["interval"] for j in jobs):
        primary = [r for r in result if r["interval"]==interval and r["scope"]=="core"]
        adjusted = holm([r["persistence_comparison"].get("primary_p",np.nan) for r in primary])
        for record,p in zip(primary,adjusted):
            record["Holm_p_core_family"] = float(p) if np.isfinite(p) else None
    atomic_json(config.root / "report/comparisons.json",result)
    spa_results = []
    for interval in dict.fromkeys(j["interval"] for j in jobs):
        for fold in range(3):
            names = sorted({j["variant"]["name"] for j in jobs if j["interval"]==interval and j["fold"]==fold and j["scope"]=="core"})
            losses = []
            common = None
            for name in names:
                matrices = [np.load(config.root / "analysis" / j["id"] / "session_moments.npz")["moments"] for j in grouped[(interval,fold,name)]]
                valid = np.all([m[:,0]>0 for m in matrices],axis=0)
                common = valid if common is None else common&valid
                losses.append(np.mean([m[:,7]/np.maximum(m[:,0],1) for m in matrices],axis=0))
                baseline = matrices[0][:,8]/np.maximum(matrices[0][:,0],1)
            if losses:
                tests = multiple_model_tests(baseline[common],np.stack(losses,axis=1)[common],config.bootstrap_replicates,config.constructor_seed)
                spa_results.append({"interval":interval,"fold":fold,"variants":names,"primary_metric":"raw_price_MSE",**tests})
    atomic_json(config.root / "report/multiple_model_tests.json",spa_results)


def figures(config, jobs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = config.root / "report/figures"
    directory.mkdir(parents=True,exist_ok=True)
    table = pd.read_csv(config.root / "report/metrics.csv")
    for interval in table.interval.unique():
        t = table[(table.interval==interval)&(table.scope=="core")]
        for metric,label,suffix in (("price_skill","Raw-price MSE","price-skill"),("log_return_skill","Log-return MSE","skill")):
            summary = t.groupby("variant")[metric].agg(["mean","std"]).sort_values("mean")
            fig,ax = plt.subplots(figsize=(10,10))
            ax.barh(summary.index,summary["mean"],xerr=summary["std"].fillna(0),color="#247b83")
            ax.axvline(0,color="black",linewidth=1)
            ax.set(xlabel=f"{label} skill versus persistence; dispersion across folds/seeds",title=f"{interval}: fixed-setting core ablations")
            fig.tight_layout(); fig.savefig(directory/f"{interval}-{suffix}.png",dpi=160); plt.close(fig)
        contrasts = table[(table.interval==interval)&(table.scope=="latest_fold_contrast")]
        if len(contrasts):
            summary=contrasts.groupby("variant").price_skill.agg(["mean","std"]).sort_values("mean")
            fig,ax=plt.subplots(figsize=(10,14))
            ax.barh(summary.index,summary["mean"],xerr=summary["std"].fillna(0),color="#247b83")
            ax.axvline(0,color="black",linewidth=1)
            ax.set(title=f"{interval}: latest-fold contrasts, two matching seeds",
                   xlabel="Raw-price MSE skill versus persistence; training-seed dispersion")
            fig.tight_layout(); fig.savefig(directory/f"{interval}-contrasts.png",dpi=160); plt.close(fig)
        structural_path=config.root/"report/final_structural_diagnostics.csv"
        if structural_path.exists():
            structural=pd.read_csv(structural_path)
            structural=structural[(structural.interval==interval)&structural.variant.isin(t.variant.unique())]
            if len(structural):
                fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
                for ax,column,label in zip(axes,["delta_hg_largest_component","normalized_delta_hg_largest_component"],["Largest-component δ_hg","Largest-component 2δ_hg / diameter"]):
                    structural.groupby(["variant","s"])[column].mean().unstack("s").plot.bar(ax=ax)
                    ax.set(ylabel=label,xlabel="Constructor configuration")
                    ax.set_ylim(bottom=0)
                axes[0].set_title(f"{interval}: selected-checkpoint structure, averaged across folds/seeds")
                fig.tight_layout(); fig.savefig(directory/f"{interval}-structural-diagnostics.png",dpi=160); plt.close(fig)
        # Representative fixed seed, latest fold; figures include actual raw price.
        job = next(j for j in jobs if j["interval"]==interval and j["fold"]==2 and j["variant"]["name"]=="A" and j["seed"]==config.seeds[0])
        with np.load(config.root / "analysis" / job["id"] / "overlay.npz") as a:
            fig,ax=plt.subplots(figsize=(12,4))
            for key in ("actual","model","persistence"):
                ax.plot(a[key],label=key,linewidth=1,alpha=.8)
            symbol=str(a["symbol"]) if "symbol" in a else "MSFT"
            ax.legend(); ax.set(title=f"{interval}: {symbol} actual/model/persistence sampled display",xlabel="Displayed chronological observations",ylabel="Raw price ($)")
            fig.tight_layout(); fig.savefig(directory/f"{interval}-price-overlay.png",dpi=160); plt.close(fig)
        for kind in ("long_only","long_short"):
            with np.load(config.root / "analysis" / job["id"] / "portfolios" / f"{kind}.npz") as a:
                fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
                axes[0].plot(a["wealth"]); axes[0].set(ylabel="Wealth",title=f"{interval} {kind}: 5 bps, 3% borrowing")
                axes[1].fill_between(np.arange(len(a["drawdown"])),a["drawdown"],0); axes[1].set(ylabel="Drawdown",xlabel="Execution marks")
                fig.tight_layout(); fig.savefig(directory/f"{interval}-{kind}-wealth.png",dpi=160); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,3),sharey=True)
        for fold in range(3):
            for variant in ("T","GKL","A"):
                curve=read_json(config.root/"fits"/interval/f"fold-{fold}"/variant/f"seed-{config.seeds[0]}"/"learning_curve.json",[])
                if curve:
                    axes[fold].plot([x["epoch"] for x in curve],[x["validation_persistence_relative_mse"] for x in curve],label=variant)
            axes[fold].axhline(1,color="black",linestyle="--",linewidth=.6); axes[fold].set(title=f"fold {fold}",xlabel="Epoch")
        axes[0].set_ylabel("Validation MSE / persistence MSE"); axes[2].legend()
        fig.tight_layout(); fig.savefig(directory/f"{interval}-learning-curves.png",dpi=160); plt.close(fig)
        geometry=[]
        for p in (config.root/"geometry"/interval/"monthly").glob("*.json"):
            g=read_json(p); geometry.append((g["month"],g.get("relative_delta")))
        if geometry:
            geometry.sort()
            fig,ax=plt.subplots(figsize=(12,3)); ax.plot([x[0] for x in geometry],[x[1] for x in geometry],marker=".")
            ax.set(title=f"{interval}: trailing stock-feature geometry",ylabel="2δ / diameter")
            ax.tick_params(axis="x",rotation=75,labelsize=6)
            fig.tight_layout(); fig.savefig(directory/f"{interval}-geometry-timeline.png",dpi=160); plt.close(fig)
        assessment=read_json(config.root/"feature_assessment"/interval/"fold-2.json")
        if assessment:
            fig,ax=plt.subplots(figsize=(10,3)); ax.bar(assessment["feature_names"],assessment["coverage"])
            ax.set(title=f"{interval}: training-only feature assessment",ylabel="Observed fraction"); ax.tick_params(axis="x",rotation=40)
            fig.tight_layout(); fig.savefig(directory/f"{interval}-feature-coverage.png",dpi=160); plt.close(fig)
        from hyperedges.common.storage import load_hyperedge_snapshot
        cover_path=config.root/"geometry"/interval/"fold-2/C.snapshot.json"
        if cover_path.exists():
            family=next(f for f in load_hyperedge_snapshot(cover_path).families if f.instance_id=="C")
            state=family.state
            descriptors=state["descriptors"]
            centered=descriptors.to_numpy()-descriptors.to_numpy().mean(axis=0)
            u,s,v=np.linalg.svd(centered,full_matrices=False); points=u[:,:2]*s[:2]
            adjacency=state["graph_adjacency"]
            membership=state["cover"].T
            from matplotlib.collections import LineCollection
            a,b=np.where(np.triu(adjacency,1)>.05)
            fig,axes=plt.subplots(1,2,figsize=(12,5))
            axes[0].add_collection(LineCollection(np.stack([points[a],points[b]],axis=1),alpha=.15,linewidths=.5))
            axes[0].scatter(points[:,0],points[:,1],c=membership.argmax(axis=1),cmap="tab20",s=18)
            axes[0].set(title="Supplied weighted graph / strongest learned cover")
            axes[1].imshow(membership,aspect="auto",vmin=0,vmax=1,cmap="viridis")
            axes[1].set(title="Stock-by-cover memberships",xlabel="Cover slot",ylabel="Eligible stock")
            fig.tight_layout(); fig.savefig(directory/f"{interval}-cover-graph.png",dpi=160); plt.close(fig)
            fig,ax=plt.subplots(figsize=(9,3))
            for name,curve in zip(["measure","geometry","topology","regularization","total"],state["loss_history"]):
                ax.plot(curve[:,0],curve[:,1],label=name)
            ax.legend(); ax.set(title=f"{interval}: published Cover Learning objective",xlabel="Optimizer iteration",ylabel="Loss")
            fig.tight_layout(); fig.savefig(directory/f"{interval}-cover-objective.png",dpi=160); plt.close(fig)
    for kind in ("circle","sphere","torus"):
        path=config.root/"validation/ph-v2"/f"{kind}-0-missing0.json"
        if path.exists():
            record=read_json(path); run=record["state"]["runs"][0]
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for dim,diagram in run["diagrams"].items():
                d=np.asarray(diagram).reshape(-1,2)
                if len(d):
                    axes[0].scatter(d[:,0],d[:,1],label=f"H{dim}",s=20)
            axes[0].plot([0,2],[0,2],color="gray",linestyle="--"); axes[0].legend()
            axes[0].set(title=f"Planted {kind}: persistence",xlabel="Birth",ylabel="Death")
            expected_dimension=1 if kind=="circle" else 2
            recovered=max((e for e in record["edges"] if e["dimension"]==expected_dimension),key=lambda e:e["persistence"],default=None)
            if recovered:
                colors=["#237f7c" if i in record["truth"] else "#b5b5b5" for i in range(len(recovered["influence"]))]
                axes[1].bar(np.arange(len(colors)),recovered["influence"],color=colors)
                axes[1].set(title=f"Recovered H{expected_dimension} stocks: {recovered['coordinates']}",xlabel="Stock coordinate",ylabel="Coordinate-ablation influence")
            fig.tight_layout(); fig.savefig(directory/f"PH-{kind}-recovery.png",dpi=160); plt.close(fig)


def analysis_worker_count(available_gib=None):
    """Leave 4 GiB for the host and budget 7 GiB per independent analysis."""
    if available_gib is None:
        try:
            import psutil
            available_gib = psutil.virtual_memory().available / 2**30
        except ImportError:
            return 1
    return 4 if available_gib >= 32 else 2 if available_gib >= 18 else 1


def _analyze_job(config, dataset, job):
    if not (config.root / "fits" / job["id"] / "result.json").exists():
        raise RuntimeError(f"Cannot analyze an incomplete forecast fit: {job['id']}")
    values = forecast_statistics(config, dataset, job)
    metric = {"interval":job["interval"],"fold":job["fold"],"variant":job["variant"]["name"],"seed":job["seed"],
              **{k:v for k,v in values.items() if not isinstance(v,(dict,list))}}
    geometry = final_geometry(config, job)
    structures = [{k:v for k,v in geometry.items() if k!="projections"}|{"s":int(s),**projection}
                  for s,projection in geometry["projections"].items()]
    return metric, structures


def _initialize_analysis_worker(config):
    global _ANALYSIS_CONFIG, _ANALYSIS_DATASETS, _ANALYSIS_THREAD_LIMITS
    _ANALYSIS_CONFIG, _ANALYSIS_DATASETS = config, {}
    try:
        from threadpoolctl import threadpool_limits
        _ANALYSIS_THREAD_LIMITS = threadpool_limits(limits=1)
    except ImportError:
        _ANALYSIS_THREAD_LIMITS = None


def _analysis_worker(job):
    if job["interval"] not in _ANALYSIS_DATASETS:
        _ANALYSIS_DATASETS[job["interval"]] = MarketDataset(_ANALYSIS_CONFIG, job["interval"])
    return _analyze_job(_ANALYSIS_CONFIG, _ANALYSIS_DATASETS[job["interval"]], job)


def analyze_jobs(config, jobs, workers=1):
    """Yield completed analyses with manifest indices; cached jobs resume safely."""
    if workers not in (1, 2, 3, 4):
        raise ValueError("Analysis workers must be between one and four")
    if workers == 1:
        datasets = {}
        for index, job in enumerate(jobs):
            if job["interval"] not in datasets:
                datasets[job["interval"]] = MarketDataset(config, job["interval"])
            yield index, *_analyze_job(config, datasets[job["interval"]], job)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize_analysis_worker, initargs=(config,)) as pool:
        pending = {pool.submit(_analysis_worker, job): index for index, job in enumerate(jobs)}
        try:
            for future in as_completed(pending):
                index = pending[future]
                try:
                    yield index, *future.result()
                except Exception as exception:
                    raise RuntimeError(f"Analysis failed for {jobs[index]['id']}") from exception
        except BaseException:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def report(config):
    # Keep reporting-only imports here: live statistics workers inspect the
    # source of the earlier numerical functions at their loaded line positions.
    from .archive_validation import verify_portfolio_archive
    from .forecast_archive_validation import verify_shared_archive, verify_fit_artifacts
    from .relative_price_inference import relative_price_report
    from .feature_pair_inference import feature_pair_report, feature_pair_figures
    jobs = manifest(config)["jobs"]
    missing = [j["id"] for j in jobs if not (config.root / "fits" / j["id"] / "result.json").exists()]
    if missing:
        raise RuntimeError(f"Report requires all {len(jobs)} final fits; {len(missing)} incomplete")
    actions=read_json(config.root/"corporate_actions/manifest.json",{})
    records=read_json(config.root/"corporate_actions/records.json",[])
    if not actions.get("complete") or len(records)!=actions.get("records"):
        raise RuntimeError("Economic reporting requires the complete archived corporate-action records")
    directory = config.root / "report"
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory / "completion.json", {"complete": False, "phase": "reporting", "manifest_identity": config.identity})
    workers = analysis_worker_count()
    atomic_json(directory / "analysis_runtime.json", {"workers":workers,"maximum_workers":4,
                "memory_policy":"4 GiB host reserve plus 7 GiB per analysis worker; serial below 18 GiB available",
                "order":"outputs retain manifest order regardless of worker completion order",
                "resume":"verified forecast-statistics and geometry caches are reused"})
    metrics, structure_batches = [None] * len(jobs), [None] * len(jobs)
    for completed, (index, metric, structure) in enumerate(analyze_jobs(config, jobs, workers), 1):
        metrics[index], structure_batches[index] = metric, structure
        if completed == 1 or completed % 10 == 0 or completed == len(jobs):
            event(config,"statistics",completed=completed,total=len(jobs),workers=workers)
    structures = [row for batch in structure_batches for row in batch]
    pd.DataFrame(metrics).to_csv(directory / "metrics.csv",index=False)
    pd.DataFrame(structures).to_csv(directory/"final_structural_diagnostics.csv",index=False)
    statistical_catalog(config)
    comparisons(config,jobs)
    primary_relative_comparisons = relative_price_report(config, jobs)
    feature_pairs = feature_pair_report(config, jobs)
    figures(config,jobs)
    feature_pair_plots = feature_pair_figures(feature_pairs, directory / "figures")
    atomic_json(directory / "feature_context_pair_figures.json", {"figures": feature_pair_plots,
                "inference_sha256": digest_file(directory / "feature_context_pair_inference.json")})
    verification, portfolio_verification, recovery_verification = [], [], []
    datasets = {i:MarketDataset(config,i) for i in config.intervals}
    shared_verification = [verify_shared_archive(config, datasets[interval], fold)
                           for interval in config.intervals for fold in folds()]
    for job in jobs:
        dataset=datasets[job["interval"]]
        recovery_verification.append(verify_fit_artifacts(config, dataset, folds()[job["fold"]], job))
        for section,folder in (("test","predictions"),("validation","validation_predictions")):
            expected=dataset.split(folds()[job["fold"]])[section]
            positions=[]
            for file in sorted((config.root/"fits"/job["id"]/folder).glob("*.npz")):
                common=config.root/"shared"/job["interval"]/f"fold-{job['fold']}"/section/file.name
                with np.load(file) as saved,np.load(common) as labels:
                    if (saved["prediction"].dtype!=np.float32 or saved["prediction"].shape!=labels["mask"].shape
                            or not np.array_equal(saved["origins"],labels["origins"])
                            or not np.isfinite(saved["prediction"][labels["mask"]]).all()):
                        raise ValueError(f"Invalid lossless prediction archive: {file}")
                    positions.append(saved["origins"])
                verification.append({"path":str(file.relative_to(config.root)),"bytes":file.stat().st_size,"sha256":digest_file(file)})
            if not positions or not np.array_equal(np.concatenate(positions),expected):
                raise ValueError(f"Incomplete {section} forecast support: {job['id']}")
        portfolio = verify_portfolio_archive(config.root / "analysis" / job["id"] / "portfolios", dataset,
                                             dataset.split(folds()[job["fold"]])["test"])
        for file in portfolio["files"]:
            file["path"] = str(Path(file["path"]).relative_to(config.root))
        portfolio_verification.append({"job": job["id"], **portfolio})
    atomic_json(directory / "archive_verification.json",{"files":verification,"lossless_dtype":"float32","compression":"npz_deflate"})
    atomic_json(directory / "portfolio_archive_verification.json", {"jobs": portfolio_verification,
                "checks": "All numeric path values finite; specified delayed execution and session support; wealth and drawdown reconstruction; primary/sensitivity equality; explicit ruin and incomplete-accounting flags; SHA256 of every portfolio path and metrics file."})
    atomic_json(directory / "recovery_archive_verification.json", {"shared_cohorts": shared_verification,
                "jobs": recovery_verification, "manifest_identity": config.identity,
                "checks": "All shared target/mask/price/time values reconstructed from prepared history; full model/optimizer/RNG checkpoints, selected epochs, feature/PH dimensions, complete learning curves, frozen snapshot integrity/stock axes/causality, causal PH scaling and session loss contributions; hashes of all checked files."})
    archive_execution_sources(config)
    atomic_json(directory / "completion.json",{"complete":True,"final_fits":len(jobs),"prediction_files":len(verification),
                "manifest_identity":config.identity,"source_archive_identity":read_json(config.root / "execution_sources/latest.json")["identity"],
                "portfolio_path_files":sum(len(job["files"]) for job in portfolio_verification),
                "recovery_artifact_files":sum(len(job["files"]) for job in recovery_verification),
                "shared_archive_files":sum(len(cohort["files"]) for cohort in shared_verification),
                "seed_variability":"reported separately from session-history uncertainty",
                "primary_relative_price_comparison_groups":len(primary_relative_comparisons),
                "direct_feature_comparison_groups":feature_pairs["groups_count"],
                "direct_feature_comparison_figures":len(feature_pair_plots),
                "metadata_limitations":["retrospective fixed universe","retrospective static GICS"],
                "additional_statistics":"recoverable from saved predictions, common masks and session contributions"})
    event(config,"report_complete",fits=len(jobs),prediction_files=len(verification))
