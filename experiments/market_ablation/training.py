"""Batched GPU fits with resumable optimizer/RNG state and full-support archives."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from hashlib import sha256
import json
import os
import random
import numpy as np
import pandas as pd
import torch
from src.hypergraph import HyperedgeConsumer
from hyperedges.common.storage import save_hyperedge_snapshot, load_hyperedge_snapshot
from .config import FEATURE_PACKS
from .feature_gather import feature_batch
from .batch_prefetch import prefetched_batches
from .constructors import cached_components, controlled_snapshot, ph_context
from .storage import atomic_json, atomic_npz, event, read_json, reserve


def device_for(config):
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def atomic_checkpoint(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(data, temporary)
    os.replace(temporary, path)


def create_model(config, dataset, fold, variant, seed, settings):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    snapshot, learned = cached_components(config, dataset, fold, variant, seed)
    if variant.control in ("frozen", "rewire", "uniform"):
        reference = config.root / "fits" / dataset.interval / f"fold-{fold['id']}" / "A" / f"seed-{seed}" / "frozen_snapshot.json"
        frozen = load_hyperedge_snapshot(reference)
        family = next(f for f in frozen.families if f.instance_id == "L")
        snapshot = controlled_snapshot(snapshot, variant, seed, family)
        learned = {}
    identifiers = tuple(f.instance_id for f in snapshot.families) + tuple(learned)
    # Cache misses/imports must never change paired forecasting initialization.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = HyperedgeConsumer(config.lookback * len(FEATURE_PACKS[variant.features]) * 2, settings["hidden"],
        family_ids=identifiers, learned_modules=learned, context_channels=15 if variant.ph_context else 0,
        family_normalization=variant.control != "no_family_norm").to(device_for(config))
    return model, snapshot


def context_batch(dataset, origins, context, normalization):
    values, mask, available = context
    sessions = dataset.sessions[origins]
    m = mask[sessions] & (available[sessions] <= dataset.times[origins])[:, None]
    return np.where(m, (values[sessions]-normalization[0])/normalization[1], 0).astype(np.float32), m


def numpy_batch(dataset, origins, variant, moments, context=None, context_normalization=None):
    x, active, y, mask = (feature_batch(dataset, origins, variant, moments)
                          if variant.features == "F" else dataset.batch(origins, variant, moments))
    extra = {}
    if variant.ph_context:
        q, qm = context_batch(dataset, origins, context, context_normalization)
        extra = {"context_values": q, "context_mask": qm}
    return x, active, y, mask, extra


def move_batch(prepared, device):
    *arrays, extra = prepared
    return *[torch.as_tensor(a, device=device) for a in arrays], {key: torch.as_tensor(a, device=device) for key, a in extra.items()}


def tensors(dataset, origins, variant, moments, device, context=None, context_normalization=None):
    return move_batch(numpy_batch(dataset, origins, variant, moments, context, context_normalization), device)


def session_loss(dataset, origins, squared, masks):
    ids = dataset.sessions[origins]
    count = np.bincount(ids, weights=masks.sum(axis=1), minlength=len(dataset.session_dates))
    total = np.bincount(ids, weights=np.where(masks, squared, 0).sum(axis=1), minlength=len(count))
    return total, count


def training_error(config,dataset,origins,prediction,target,moments):
    if config.training_loss=="log_return_MSE":
        return prediction-target
    current=torch.as_tensor(np.array(dataset.bars[origins,:,3]),device=prediction.device)
    actual=torch.as_tensor(np.array(dataset.bars[np.asarray(origins)+1,:,3]),device=prediction.device)
    # This equals predicted raw price minus actual raw price, while expm1
    # avoids subtracting two nearly equal large floating point prices.
    return ((current-actual)+current*torch.expm1(prediction*float(moments["target_scale"]))) / float(moments["price_target_scale"])


@torch.no_grad()
def evaluate(model, snapshot, dataset, origins, variant, moments, batch_size, context=None, context_normalization=None, directory=None):
    model.eval()
    device = next(model.parameters()).device
    total, baseline, counts = [np.zeros(len(dataset.session_dates), np.float64) for _ in range(3)]
    price_total,price_baseline=[np.zeros_like(total) for _ in range(2)]
    month_labels = pd.to_datetime(dataset.times[origins+1], utc=True).strftime("%Y-%m").to_numpy()
    for month in dict.fromkeys(month_labels):
        selected = origins[month_labels == month]
        path = Path(directory) / f"{month}.npz" if directory else None
        if path and path.exists():
            with np.load(path) as saved:
                prediction = saved["prediction"]
                if not np.array_equal(saved["origins"], selected):
                    raise ValueError("Prediction support changed while resuming")
            _, _, y, mask = dataset.batch(selected, variant, moments)
            target = y*moments["target_scale"]
            sums, number = session_loss(dataset, selected, (prediction-target)**2, mask)
            naive, _ = session_loss(dataset, selected, target**2, mask)
            current=np.array(dataset.bars[selected,:,3],np.float64)
            actual=np.array(dataset.bars[selected+1,:,3],np.float64)
            price_sums,_=session_loss(dataset,selected,((current-actual)+current*np.expm1(prediction.astype(float)))**2,mask)
            price_naive,_=session_loss(dataset,selected,(current-actual)**2,mask)
        else:
            prediction = np.empty((len(selected), len(dataset.symbols)), np.float32) if path else None
            sums, naive, number = [np.zeros_like(total) for _ in range(3)]
            price_sums,price_naive=[np.zeros_like(total) for _ in range(2)]
            for begin in range(0, len(selected), batch_size):
                batch = selected[begin:begin+batch_size]
                x, active, y, mask, extra = tensors(dataset, batch, variant, moments, device, context, context_normalization)
                pred = model(x, snapshot, active, **extra).squeeze(-1)
                if not torch.isfinite(pred[mask]).all():
                    raise FloatingPointError("Nonfinite evaluation prediction")
                p = pred.cpu().numpy()*moments["target_scale"]
                target, valid = y.cpu().numpy()*moments["target_scale"], mask.cpu().numpy()
                s, c = session_loss(dataset, batch, (p-target)**2, valid)
                n, _ = session_loss(dataset, batch, target**2, valid)
                current=np.array(dataset.bars[batch,:,3],np.float64)
                actual=np.array(dataset.bars[batch+1,:,3],np.float64)
                ps,_=session_loss(dataset,batch,((current-actual)+current*np.expm1(p.astype(float)))**2,valid)
                pn,_=session_loss(dataset,batch,(current-actual)**2,valid)
                price_sums+=ps;price_naive+=pn
                sums += s
                naive += n
                number += c
                if prediction is not None:
                    prediction[begin:begin+len(batch)] = p
            if path:
                reserve(dataset.config, prediction.nbytes)
                atomic_npz(path, origins=selected, prediction=prediction)
        price_total+=price_sums;price_baseline+=price_naive
        total += sums
        baseline += naive
        counts += number
    objective,reference=(price_total,price_baseline) if dataset.config.training_loss=="raw_price_MSE" else (total,baseline)
    valid = (counts > 0) & (reference > 0)
    score = float(np.mean(objective[valid]/reference[valid])) if valid.any() else float("inf")
    return score, {"loss_sum": total, "persistence_loss_sum": baseline, "count": counts,
                   "price_loss_sum":price_total,"persistence_price_loss_sum":price_baseline}


def fit(config, dataset, fold, variant, seed, settings, *, tuning=False, trial=None, anchor_index=0):
    directory = config.root / ("tuning_fits" if tuning else "fits") / dataset.interval / f"fold-{fold['id']}" / variant.name / f"seed-{seed}"
    if tuning:
        identity = sha256(json.dumps({"settings":settings,"training_loss":config.training_loss},sort_keys=True).encode()).hexdigest()[:20]
        directory = directory / f"settings-{identity}"
    result_path = directory / "result.json"
    saved_result = read_json(result_path)
    if saved_result:
        if saved_result.get("training_loss")!=config.training_loss:
            raise RuntimeError("Completed fit has a different training objective; preserve it and use a new output directory")
        return saved_result
    reserve(config)
    model, snapshot = create_model(config, dataset, fold, variant, seed, settings)
    directory.mkdir(parents=True, exist_ok=True)
    save_hyperedge_snapshot(snapshot, directory / "initial_snapshot.json")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    moments, split = dataset.moments(fold), dataset.split(fold)
    batch_size = settings["batch_size"]
    context = ph_context(config, dataset) if variant.ph_context else None
    normalization = None
    if context:
        q, qm, qa = context
        valid_sessions = (qa <= pd.Timestamp(fold["fit_cutoff"]).value) & qm.all(axis=1)
        mean, std = q[valid_sessions].mean(axis=0), q[valid_sessions].std(axis=0)
        normalization = (mean, np.where(std > 1e-8, std, 1))
        atomic_npz(directory / "context_scaling.npz", mean=normalization[0], std=normalization[1])
    validation = dataset.tuning_origins(split["validation"])
    max_epochs, patience = (config.tuning_epochs, config.tuning_patience) if tuning else (config.epochs, config.patience)
    epoch, batch_offset, best, stale, curves = 0, 0, float("inf"), 0, []
    elapsed_previous = 0.
    train_sum, train_count = 0., 0
    latest = directory / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location=device_for(config), weights_only=False)
        if saved.get("training_loss")!=config.training_loss or saved["snapshot_id"]!=snapshot.snapshot_id:
            raise RuntimeError("Resume checkpoint objective or constructor identity changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, batch_offset, best, stale, curves = (saved[k] for k in ("epoch", "batch_offset", "best", "stale", "curves"))
        elapsed_previous = saved["seconds"]
        train_sum, train_count = saved.get("train_sum", 0.), saved.get("train_count", 0)
        torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
    start = perf_counter()

    def checkpoint(next_epoch, next_offset):
        atomic_checkpoint(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": next_epoch,
            "batch_offset": next_offset, "best": best, "stale": stale, "curves": curves,
            "train_sum": train_sum, "train_count": train_count,
            "training_loss":config.training_loss,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "seconds": elapsed_previous+perf_counter()-start, "settings": settings, "variant": asdict(variant),
            "seed": seed, "snapshot_id": snapshot.snapshot_id})

    while epoch < max_epochs and stale < patience:
        model.train()
        schedule = dataset.epoch_origins(split["train"], epoch, tuning=tuning).copy()
        np.random.default_rng(seed+2+epoch*1009).shuffle(schedule)
        prepare = lambda origins: numpy_batch(dataset, origins, variant, moments, context, normalization)
        with prefetched_batches(schedule, batch_size, prepare, start=batch_offset) as batches:
            for offset, origins, prepared in batches:
                x, active, y, mask, extra = move_batch(prepared, device_for(config))
                count = mask.sum()
                if not count:
                    continue
                optimizer.zero_grad(set_to_none=True)
                pred = model(x, snapshot, active, **extra).squeeze(-1)
                error=training_error(config,dataset,origins,pred,y,moments)
                mse = (error.square()*mask).sum()/count
                loss = mse + model.membership_penalty()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite CUDA training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                train_sum += float(mse.detach())*int(count)
                train_count += int(count)
                if (offset//batch_size+1) % 100 == 0:
                    reserve(config)
                    checkpoint(epoch, offset+len(origins))
        score, _ = evaluate(model, snapshot, dataset, validation, variant, moments, batch_size, context, normalization)
        improved = score < best
        if improved:
            best, stale = score, 0
            atomic_checkpoint(directory / "best.pt", {"model": model.state_dict(), "settings": settings,
                              "variant": asdict(variant), "seed": seed, "epoch": epoch, "score": score})
        else:
            stale += 1
        curves.append({"epoch": epoch+1, "train_mse_scaled": train_sum/max(train_count, 1),
                       "validation_persistence_relative_mse": score, "improved": improved,
                       "training_origins": len(schedule), "sampling_offset": epoch % (config.minute_training_stride if dataset.interval == "1m" else 1)})
        event(config, "training", interval=dataset.interval, fold=fold["id"], variant=variant.name, seed=seed,
              tuning=tuning, trial=trial.number if trial else None, **curves[-1])
        epoch += 1
        batch_offset, train_sum, train_count = 0, 0., 0
        checkpoint(epoch, 0)
        atomic_json(directory / "learning_curve.json", curves)
        if trial is not None:
            import optuna
            trial.report(score, anchor_index*config.tuning_epochs+epoch-1)
            if epoch >= 3 and trial.should_prune():
                raise optuna.TrialPruned()
    state = torch.load(directory / "best.pt", map_location=device_for(config), weights_only=False)
    model.load_state_dict(state["model"])
    full_validation, validation_contributions = evaluate(model, snapshot, dataset, split["validation"], variant, moments,
                                                        batch_size, context, normalization,
                                                        None if tuning else directory/"validation_predictions")
    atomic_npz(directory / "validation_contributions.npz", **validation_contributions)
    result = {"best_sampled_validation": best, "full_validation": full_validation, "epochs": epoch,
              "still_improving_at_cap": bool(epoch == max_epochs and curves[-1]["improved"]),
              "seconds": elapsed_previous+perf_counter()-start, "settings": settings,
              "seed": seed, "variant": asdict(variant), "tuning": tuning}
    result["training_loss"]=config.training_loss
    if not tuning:
        # Freeze exactly the chosen checkpoint's memberships before archival.
        frozen = model.freeze_memberships()
        from dataclasses import replace
        export = replace(snapshot, families=snapshot.families+frozen, learned_references={})
        save_hyperedge_snapshot(export, directory / "frozen_snapshot.json")
        _, contributions = evaluate(model, snapshot, dataset, split["test"], variant, moments, batch_size,
                                    context, normalization, directory / "predictions")
        atomic_npz(directory / "test_contributions.npz", **contributions)
        comparable = (contributions["count"]>0)&(contributions["persistence_loss_sum"]>0)
        result["test_persistence_relative_mse"] = float(np.mean(contributions["loss_sum"][comparable] /
                                     contributions["persistence_loss_sum"][comparable]))
        price_comparable=(contributions["count"]>0)&(contributions["persistence_price_loss_sum"]>0)
        result["test_price_persistence_relative_mse"]=float(np.mean(contributions["price_loss_sum"][price_comparable]/
                                                        contributions["persistence_price_loss_sum"][price_comparable]))
    result["seconds"] = elapsed_previous+perf_counter()-start
    atomic_json(result_path, result)
    event(config, "fit_complete", interval=dataset.interval, fold=fold["id"], variant=variant.name, seed=seed,
          tuning=tuning, epochs=epoch, seconds=result["seconds"])
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def tune(config, dataset, fold, batch_size):
    import optuna
    from .config import core_variants
    path = config.root / "settings" / f"{dataset.interval}.json"
    saved = read_json(path)
    if saved and saved.get("training_loss")==config.training_loss:
        return saved
    storage = f"sqlite:///{(config.root / 'optuna.sqlite3').as_posix()}"
    study = optuna.create_study(study_name=f"joint-{dataset.interval}-{config.identity[:12]}", storage=storage,
        load_if_exists=True, direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.constructor_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3))
    # Interrupted RUNNING trials have intact fit checkpoints; keep their settings.
    for unfinished in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(unfinished.number, state=optuna.trial.TrialState.FAIL)
        study.enqueue_trial(unfinished.params)
    anchors = [next(v for v in core_variants() if v.name == name) for name in ("T", "GKL", "A")]

    def objective(trial):
        settings = {"hidden": trial.suggest_categorical("hidden", [32,64,128]),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                    "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True), "batch_size": batch_size}
        scores = [fit(config, dataset, fold, variant, config.seeds[0], settings, tuning=True, trial=trial, anchor_index=i)["full_validation"]
                  for i, variant in enumerate(anchors)]
        trial.set_user_attr("anchor_scores", scores)
        return float(np.mean(scores))
    finished = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)))
    study.optimize(objective, n_trials=max(0, config.tuning_trials-finished), gc_after_trial=True)
    best = {**study.best_params, "batch_size": batch_size, "objective": study.best_value,
            "training_loss":config.training_loss,
            "selection": "first_fold_only_equal_anchor_persistence_relative_MSE", "fit_cutoff": fold["fit_cutoff"],
            "validation_end": fold["validation_end"], "trials": config.tuning_trials}
    atomic_json(path, best)
    return best
