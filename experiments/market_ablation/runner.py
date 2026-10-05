"""Measured preflight followed by the complete fixed-setting manifest."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
import os
import json
import numpy as np
import pandas as pd
import torch
from hyperbolicity.diagnostics import summarize_feature_geometry, summarize_structural_geometry
from hyperedges.common.storage import save_hyperedge_snapshot
from .config import SweepConfig, core_variants, contrast_variants, folds, manifest, Variant
from .dataset import MarketDataset, prepare
from .constructors import cached_components, ph_context
from .training import create_model, device_for, tensors, fit, tune,training_error,evaluate
from .storage import atomic_json, atomic_npz, event, read_json, reserve,archive_execution_sources


def archive_shared(config, dataset, fold):
    from .config import FEATURE_PACKS
    r = next(v for v in core_variants() if v.name == "T")
    split, moments = dataset.split(fold), dataset.moments(fold)
    for section in ("validation", "test"):
        origins = split[section]
        labels = pd.to_datetime(dataset.times[origins+1], utc=True).strftime("%Y-%m").to_numpy()
        for month in dict.fromkeys(labels):
            path = config.root / "shared" / dataset.interval / f"fold-{fold['id']}" / section / f"{month}.npz"
            if path.exists():
                continue
            positions = origins[labels == month]
            mask = np.empty((len(positions), 250), bool)
            active = np.empty_like(mask)
            target = np.empty((len(positions), 250), np.float32)
            for begin in range(0, len(positions), 256):
                batch = positions[begin:begin+256]
                _, a, y, m = dataset.batch(batch, r, moments)
                active[begin:begin+len(batch)] = a
                mask[begin:begin+len(batch)], target[begin:begin+len(batch)] = m, y*moments["target_scale"]
            reserve(config, target.nbytes*3)
            atomic_npz(path, origins=positions, target=target, mask=mask, active=active,
                       current_price=np.array(dataset.bars[positions, :, 3]), actual_price=np.array(dataset.bars[positions+1, :, 3]),
                       origin_times=dataset.times[positions], target_times=dataset.times[positions+1], sessions=dataset.sessions[positions])


def geometry(config, dataset, fold, variant, seed=None):
    path = config.root / "geometry" / dataset.interval / f"fold-{fold['id']}" / f"{variant.name}.json"
    if path.exists() and path.with_suffix(".snapshot.json").exists():
        return
    snapshot, learned = cached_components(config, dataset, fold, variant, config.seeds[0] if seed is None else seed)
    for family in snapshot.families:
        if family.instance_id == "C" and len(family.edge_ids)<2:
            raise RuntimeError("Cover Learning collapsed to fewer than two useful groups; repair the validated recipe before forecasting")
    frozen = tuple(__import__("hyperedges.learned_membership.constructor", fromlist=["freeze_learned_memberships"]).freeze_learned_memberships(m) for m in learned.values())
    snapshot = replace(snapshot, families=snapshot.families+frozen, learned_references={})
    structural = summarize_structural_geometry(snapshot, seed=config.constructor_seed)
    atomic_json(path, structural)
    save_hyperedge_snapshot(snapshot, path.with_suffix(".snapshot.json"))
    event(config, "geometry", interval=dataset.interval, fold=fold["id"], variant=variant.name,
          coverage=structural["covered_fraction"], edges=structural["edges"])


def monthly_geometry(config, dataset):
    periods = pd.to_datetime(dataset.times, utc=True).strftime("%Y-%m").to_numpy()
    for month in dict.fromkeys(periods):
        path = config.root / "geometry" / dataset.interval / "monthly" / f"{month}.json"
        if path.exists():
            continue
        positions = np.flatnonzero(periods == month)
        positions = positions[-min(64, len(positions)):]
        x = np.array(dataset.features[positions, :, 0:1])
        m = dataset.feature_mask[positions, :, 0:1]
        # A trailing-only pooled scale defines this diagnostic metric.
        scale = np.sqrt(np.where(m, x*x, 0).sum()/max(m.sum(), 1)) or 1
        result = summarize_feature_geometry(x/scale, m, node_ids=dataset.symbols, seed=config.constructor_seed)
        result.update(month=month, interval=dataset.interval, scale=scale,
                      available_at=pd.Timestamp(dataset.times[positions[-1]], unit="ns", tz="UTC").isoformat())
        atomic_json(path, result)


def assess_features(config,dataset,fold):
    path=config.root/"feature_assessment"/dataset.interval/f"fold-{fold['id']}.json"
    if path.exists():
        return
    from .config import FEATURE_NAMES
    split=dataset.split(fold)
    positions=dataset.tuning_origins(split["train"])
    count,sums,squares=[np.zeros(10,float) for _ in range(3)]
    covariance=np.zeros((10,10),float)
    complete_count=0
    for start in range(0,len(positions),512):
        selected=positions[start:start+512]
        x=np.asarray(dataset.features[selected],float).reshape(-1,10)
        m=dataset.feature_mask[selected].reshape(-1,10)
        count+=m.sum(axis=0); sums+=np.where(m,x,0).sum(axis=0); squares+=np.where(m,x*x,0).sum(axis=0)
        complete=x[m.all(axis=1)]
        if len(complete):
            # Raw second moments allow later covariance diagnostics without predictors being refitted.
            covariance+=complete.T@complete
            complete_count+=len(complete)
    mean=sums/np.maximum(count,1)
    std=np.sqrt(np.maximum(squares/np.maximum(count,1)-mean*mean,0))
    result={"interval":dataset.interval,"fold":fold["id"],"fit_cutoff":fold["fit_cutoff"],"sampled_origins":len(positions),
            "feature_names":FEATURE_NAMES,"coverage":(count/max(len(positions)*250,1)).tolist(),
            "mean":mean.tolist(),"std":std.tolist(),"constant_features":[name for name,s in zip(FEATURE_NAMES,std) if s<1e-10],
            "complete_case_second_moments":(covariance/max(complete_count,1)).tolist(),"complete_case_count":complete_count,
            "feature_packs":["R","RV","F"],"assessment_data":"training_only_stratified_origins"}
    atomic_json(path,result)


def pilot(config, dataset, frozen_settings=None):
    fold = folds()[0]
    moments, split = dataset.moments(fold), dataset.split(fold)
    sample = dataset.tuning_origins(split["train"])
    variants = [next(v for v in core_variants() if v.name == label) for label in ("T", "A")]
    measured = []
    for variant in variants:
        batch_sizes = config.batch_sizes if frozen_settings is None else (frozen_settings["batch_size"],)
        for batch_size in batch_sizes:
            settings = {"hidden": 128, "learning_rate": .001, "weight_decay": .0001, "batch_size": batch_size} if frozen_settings is None else dict(frozen_settings)
            model, snapshot = create_model(config, dataset, fold, variant, config.seeds[0], settings)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
            if device_for(config).type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                for step in range(35):
                    origins = sample[(np.arange(batch_size)+step*batch_size) % len(sample)]
                    x, active, y, mask, extra = tensors(dataset, origins, variant, moments, device_for(config))
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(x, snapshot, active).squeeze(-1)
                    error=training_error(config,dataset,origins,prediction,y,moments)
                    loss = (error.square()*mask).sum()/mask.sum().clamp_min(1) + model.membership_penalty()
                    loss.backward()
                    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        raise FloatingPointError("Pilot CUDA gradient check failed")
                    optimizer.step()
                    if step == 4:
                        if device_for(config).type == "cuda":
                            torch.cuda.synchronize()
                        start = perf_counter()
                if device_for(config).type == "cuda":
                    torch.cuda.synchronize()
                training = 30*batch_size/(perf_counter()-start)
                model.eval()
                with torch.no_grad():
                    start = perf_counter()
                    origins=sample[np.arange(30*batch_size)%len(sample)]
                    evaluate(model,snapshot,dataset,origins,variant,moments,batch_size)
                    if device_for(config).type == "cuda":
                        torch.cuda.synchronize()
                    evaluation = 30*batch_size/(perf_counter()-start)
                measured.append({"variant": variant.name, "batch_size": batch_size, "hidden": settings["hidden"],
                                 "training_origins_per_second": training, "evaluation_origins_per_second": evaluation,
                                 "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device_for(config).type == "cuda" else 0, "safe": True})
            except torch.OutOfMemoryError:
                measured.append({"variant": variant.name, "batch_size": batch_size, "safe": False, "reason": "CUDA out of memory"})
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    candidates = [m for m in measured if m["variant"] == "A" and m["safe"]]
    if not candidates:
        raise RuntimeError("Neither requested pilot batch size fits safely")
    batch_size = max(candidates, key=lambda m: m["training_origins_per_second"])["batch_size"]
    selected = {m["variant"]: m for m in measured if m["batch_size"] == batch_size and m["safe"]}
    jobs = [j for j in manifest(config)["jobs"] if j["interval"] == dataset.interval]
    training_seconds, evaluation_seconds, prediction_bytes = 0., 0., 0
    for job in jobs:
        support = dataset.split(folds()[job["fold"]])
        throughput = selected["T" if not job["variant"]["families"] else "A"]
        feature_factor = 1.3 if job["variant"]["features"] == "F" else (1.1 if job["variant"]["features"] == "RV" else 1)
        stride = config.minute_training_stride if dataset.interval == "1m" else 1
        training_seconds += config.epochs*len(support["train"])/stride/throughput["training_origins_per_second"]*feature_factor
        evaluation_seconds += (config.epochs*min(config.tuning_origin_cap, len(support["validation"])) + len(support["validation"]) + len(support["test"]))/throughput["evaluation_origins_per_second"]*feature_factor
        prediction_bytes += (len(support["test"])+len(support["validation"]))*250*4
    tuning_seconds = 0 if frozen_settings is not None else config.tuning_trials*3*config.tuning_epochs*min(config.tuning_origin_cap, len(split["train"]))/selected["A"]["training_origins_per_second"]
    result = {"measurements": measured, "batch_size": batch_size, "device": str(device_for(config)),
              "training_loss":config.training_loss,
              "evaluation_measurement":"actual batched evaluation with device transfers and raw-price/session loss reductions",
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "projected_training_hours_at_cap": (training_seconds+tuning_seconds)/3600,
              "projected_evaluation_hours": evaluation_seconds/3600, "prediction_upper_bound_gib": prediction_bytes/2**30,
              "model_width_projection": "128 (largest tuning candidate)" if frozen_settings is None else f"{frozen_settings['hidden']} (frozen selected settings)",
              "frozen_settings": frozen_settings, "tuning_included": frozen_settings is None,
              "projection_excludes": ["constructor_computation", "PH_context", "statistics", "plots", "compression_I/O"]}
    pilot_directory="pilot" if fold["id"]==0 else "validation/smoke-pilot"
    result["pilot_fold"]=fold
    filename = f"{dataset.interval}.json" if frozen_settings is None else f"frozen-{dataset.interval}.json"
    atomic_json(config.root / pilot_directory / filename, result)
    event(config, "pilot_complete", interval=dataset.interval, **{k:v for k,v in result.items() if k != "measurements"})
    return result


def run(config, phase="run"):
    torch.set_num_threads(4)
    # Prevent accidental concurrent GPU trainers on the same resumable manifest.
    lock_path = config.root / "runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        old = read_json(lock_path)
        import psutil
        if psutil.pid_exists(old["pid"]):
            raise RuntimeError(f"Runner already active with PID {old['pid']}")
    atomic_json(lock_path, {"pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()})
    try:
        if phase == "report":
            from .reporting import report
            return report(config)
        event(config,"runner_started",pid=os.getpid(),manifest_fits=manifest(config)["final_fits"])
        archive_execution_sources(config)
        preparation = read_json(config.root / "dataset/preparation.json", {})
        if not preparation.get("complete"):
            # A preparation worker may already be running; wait without touching its memmaps.
            prep_pid_path = config.root / "prepare.pid"
            if prep_pid_path.exists():
                import psutil
                prep_pid = int(prep_pid_path.read_text().strip())
                while psutil.pid_exists(prep_pid) and not read_json(config.root / "dataset/preparation.json", {}).get("complete"):
                    sleep(30)
            if not read_json(config.root / "dataset/preparation.json", {}).get("complete"):
                prepare(config)
        from .validation import validate
        validate(config)
        from .actions import fetch_actions,fetch_recipient_prices
        fetch_actions(config)
        fetch_recipient_prices(config,read_json(config.root/"corporate_actions/records.json"))
        datasets = {interval: MarketDataset(config, interval) for interval in config.intervals}
        pilots = {}
        for interval, dataset in datasets.items():
            from .preflight import constructor_validation,validate_control_recipes
            constructor_validation(config,dataset,folds()[0])
            validate_control_recipes(config,dataset,folds()[2])
            # Geometry precedes all tuning/forecasting in each interval.
            monthly_geometry(config, dataset)
            for fold in folds():
                assess_features(config,dataset,fold)
                for variant in core_variants():
                    if len(variant.families) == 1 or variant.name == "A":
                        geometry(config, dataset, fold, variant)
                archive_shared(config, dataset, fold)
            saved_pilot=read_json(config.root / "pilot" / f"{interval}.json")
            pilots[interval] = saved_pilot if saved_pilot and saved_pilot.get("pilot_fold")==folds()[0] and saved_pilot.get("training_loss")==config.training_loss else pilot(config,dataset)
            ph_context(config, dataset)
        largest_candidate_projected = sum(p["projected_training_hours_at_cap"]+p["projected_evaluation_hours"] for p in pilots.values()) + 24
        selected_pilots = {}
        for interval, dataset in datasets.items():
            settings = read_json(config.root / "settings" / f"{interval}.json")
            if settings and settings.get("training_loss") == config.training_loss:
                measured = read_json(config.root / "pilot" / f"frozen-{interval}.json")
                selected_pilots[interval] = measured if measured and measured.get("frozen_settings") == settings else pilot(config, dataset, settings)
        effective_pilots = selected_pilots if len(selected_pilots) == len(datasets) else pilots
        projected = sum(p["projected_training_hours_at_cap"]+p["projected_evaluation_hours"] for p in effective_pilots.values()) + 24
        atomic_json(config.root / "preflight.json", {"passed": True, "projected_hours": projected, "target_hours": config.target_hours,
                   "exceeds_target": projected > config.target_hours, "action": "execute_full_authorized_manifest_without_silent_cuts",
                   "free_gib": reserve(config)/2**30, "pilot": effective_pilots,
                   "largest_candidate_projected_hours": largest_candidate_projected,
                   "stage_allowance_hours": 24, "projection_uses_frozen_settings": effective_pilots is selected_pilots})
        event(config, "preflight_complete", projected_hours=projected, target_hours=config.target_hours, exceeds_target=projected>config.target_hours)
        if phase == "preflight":
            return
        for interval, dataset in datasets.items():
            early=read_json(config.root/"early_tuning.lock")
            if early:
                import psutil
                while psutil.pid_exists(early["pid"]):
                    sleep(30)
            settings = tune(config, dataset, folds()[0], pilots[interval]["batch_size"])
            for fold in folds():
                for variant in core_variants()+ (contrast_variants() if fold["id"] == 2 else []):
                    seeds = config.seeds if variant in core_variants() else config.seeds[:2]
                    for seed in seeds:
                        fit(config, dataset, fold, variant, seed, settings)
        from .reporting import report
        report(config)
        event(config, "experiment_complete", fits=manifest(config)["final_fits"])
    finally:
        lock_path.unlink(missing_ok=True)


def early_tuning(config):
    """Overlap first-fold scientific checks/tuning with later archive preparation."""
    import psutil
    lock=config.root/"early_tuning.lock"
    old=read_json(lock)
    if old and psutil.pid_exists(old["pid"]):
        raise RuntimeError("First-fold tuning is already running")
    atomic_json(lock,{"pid":os.getpid()})
    torch.set_num_threads(4)
    try:
        preparation=read_json(config.root/"dataset/preparation.json",{})
        completed=set(preparation.get("completed",[]))
        symbols=read_json(config.root/"dataset/metadata.json")["symbols"]
        required=[f"{month:%Y-%m}/{symbol}" for month in pd.date_range(pd.Timestamp(config.start).replace(day=1),folds()[0]["validation_end"][:10],freq="MS",inclusive="left") for symbol in symbols]
        if not all(key in completed for key in required):
            raise RuntimeError("First-fold training/validation partitions are not all prepared")
        from .validation import validate
        from .preflight import constructor_validation
        validate(config)
        datasets={i:MarketDataset(config,i) for i in config.intervals}
        pilots={}
        for interval,dataset in datasets.items():
            constructor_validation(config,dataset,folds()[0])
            for variant in core_variants():
                if len(variant.families)==1 or variant.name=="A":
                    geometry(config,dataset,folds()[0],variant)
            saved=read_json(config.root/"pilot"/f"{interval}.json")
            pilots[interval]=saved if saved and saved.get("pilot_fold")==folds()[0] and saved.get("training_loss")==config.training_loss else pilot(config,dataset)
        projected=sum(p["projected_training_hours_at_cap"]+p["projected_evaluation_hours"] for p in pilots.values())+24
        atomic_json(config.root/"early_preflight.json",{"passed":True,"projected_hours":projected,"target_hours":config.target_hours,
             "exceeds_target":projected>config.target_hours,"scope":"first_fold_constructor_checks_and_pilots; later-fold/preparation checks continue",
             "action":"execute_full_authorized_manifest_without_silent_cuts","pilot":pilots})
        event(config,"early_preflight_complete",projected_hours=projected,target_hours=config.target_hours,exceeds_target=projected>config.target_hours)
        archive_execution_sources(config)
        for interval,dataset in datasets.items():
            tune(config,dataset,folds()[0],pilots[interval]["batch_size"])
        event(config,"tuning_complete",intervals=list(datasets))
    finally:
        lock.unlink(missing_ok=True)
