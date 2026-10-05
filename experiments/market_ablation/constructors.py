"""Recipe caches shared across forecasting variants and seeds."""
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd
from hyperedges.common.types import ConstructionHistory, ConstructionContext, HyperedgeSnapshot, HyperedgeFamily
from hyperedges.common.pipeline import fit_hyperedge_pipeline, build_hyperedge_snapshot
from hyperedges.common.storage import save_hyperedge_snapshot, load_hyperedge_snapshot, _encode, _decode
from hyperedges.learned_membership.constructor import LearnedMembershipConstructor
from .config import constructor_specs
from .storage import atomic_json, read_json, event, reserve

_BUILDER_ROOT=Path(__file__).resolve().parents[2]/"hyperedges"
BUILDER_SHA256=sha256(b"".join(p.read_bytes() for p in sorted(_BUILDER_ROOT.rglob("*.py")))).hexdigest()


def history_context(dataset, fold, seed, window=None):
    stop = np.searchsorted(dataset.times, pd.Timestamp(fold["fit_cutoff"]).value, side="right")
    begin = 0
    if window is not None and stop:
        first_session = max(0, int(dataset.sessions[stop-1]) - window.value + 1)
        begin = int(np.searchsorted(dataset.sessions, first_session))
    values = np.array(dataset.features[begin:stop, :, 0])
    mask = np.array(dataset.feature_mask[begin:stop, :, 0])
    structural = np.ones(stop-begin, bool)
    if dataset.interval != "1d":
        previous = np.r_[-1, dataset.sessions[:-1]][begin:stop]
        structural = dataset.sessions[begin:stop] == previous
    times = pd.to_datetime(dataset.times[begin:stop], utc=True)
    sessions = pd.to_datetime(dataset.session_dates[dataset.sessions[begin:stop]])
    previous_times = np.r_[np.datetime64("NaT", "ns").astype(np.int64), dataset.times[:-1]][begin:stop]
    cutoff = pd.Timestamp(fold["fit_cutoff"])
    history = ConstructionHistory(pd.DataFrame(values, index=times, columns=dataset.symbols),
        pd.DataFrame(mask, index=times, columns=dataset.symbols), structural, sessions, times, cutoff,
        {"interval": dataset.interval, "source": "raw_minute_archive", "return_protocol": "close_to_close_daily_or_within_session_intraday"},
        pd.to_datetime(previous_times, utc=True))
    return ConstructionContext(dataset.symbols, history, cutoff, seed, fold_id=f"fold-{fold['id']}")


def cached_components(config, dataset, fold, variant, seed):
    snapshots, learned = [], {}
    for spec in constructor_specs(variant, dataset.interval, config.gics):
        recipe_id = sha256(json.dumps({"spec":asdict(spec),"builder_source":BUILDER_SHA256}, sort_keys=True, default=str).encode()).hexdigest()[:20]
        directory = config.root / "constructors" / dataset.interval / f"fold-{fold['id']}"
        if spec.method == "learned_membership":
            path = directory / f"{spec.instance_id}-{recipe_id}-seed-{seed+3}.json"
            saved = read_json(path)
            if saved is None:
                context = history_context(dataset, fold, seed+3)
                module = LearnedMembershipConstructor(spec, seed+3)
                module.fit(context)
                saved = _encode(module.export_state())
                atomic_json(path, saved)
            learned[spec.instance_id] = LearnedMembershipConstructor.from_export_state(_decode(saved))
            continue
        path = directory / f"{spec.instance_id}-{recipe_id}.json"
        if not path.exists():
            reserve(config)
            context = history_context(dataset, fold, config.constructor_seed, spec.history_window)
            event(config, "constructor_start", interval=dataset.interval, fold=fold["id"], family=spec.instance_id, recipe=recipe_id)
            pipeline = fit_hyperedge_pipeline(context, (spec,))
            snapshot = build_hyperedge_snapshot(pipeline)
            snapshot=replace(snapshot,configuration={**snapshot.configuration,"builder_source_sha256":BUILDER_SHA256})
            save_hyperedge_snapshot(snapshot, path)
            event(config, "constructor_complete", interval=dataset.interval, fold=fold["id"], family=spec.instance_id,
                  edges=sum(len(f.edge_ids) for f in snapshot.families), seconds=sum(f.diagnostics["construction_seconds"] for f in snapshot.families))
        snapshots.append(load_hyperedge_snapshot(path))
    cutoff = pd.Timestamp(fold["fit_cutoff"])
    identity = sha256(json.dumps([dataset.interval, fold["id"], variant.name, [s.snapshot_id for s in snapshots]]).encode()).hexdigest()
    combined=[f for s in snapshots for f in s.families]
    event_families=[f for f in combined if f.instance_id.startswith("E-")]
    if event_families:
        merged=HyperedgeFamily("E","event_dowker_combined",pd.concat([f.incidence for f in event_families],axis=1),
            attributes={edge:attribute for f in event_families for edge,attribute in f.attributes.items()},
            diagnostics={"components":[f.instance_id for f in event_families]},
            provenance={"components":{f.instance_id:f.provenance for f in event_families}},
            state={"components":{f.instance_id:f.state for f in event_families}})
        combined=[f for f in combined if not f.instance_id.startswith("E-")]+[merged]
    snapshot = HyperedgeSnapshot(identity, dataset.symbols, tuple(combined), cutoff, cutoff, cutoff,
                                fold_id=f"fold-{fold['id']}", configuration={"variant": asdict(variant)},
                                learned_references={name:module.export_state() for name,module in learned.items()})
    return snapshot, learned


def controlled_snapshot(snapshot, variant, seed, frozen_family=None):
    """Switching preserves each node degree and edge size; uniform controls preserve sizes."""
    if variant.control not in ("frozen", "rewire", "uniform"):
        return snapshot
    if frozen_family is None:
        raise ValueError("Frozen-construction contrasts require the matching full-model seed's learned family")
    rng = np.random.default_rng(seed + 701)
    families = [*snapshot.families, frozen_family]
    result = []
    for family in families:
        selected = variant.control in ("rewire", "uniform") and (variant.parameter == "all" or family.instance_id.split("-")[0] == variant.parameter)
        if not selected:
            result.append(family)
            continue
        h = family.incidence.to_numpy(copy=True)
        sizes, degrees = h.sum(axis=0), h.sum(axis=1)
        switches = 0
        if variant.control == "rewire" and h.sum():
            memberships = np.argwhere(h)
            attempts = int(20 * len(memberships))
            for _ in range(attempts):
                a, b = rng.integers(len(memberships), size=2)
                u, e = memberships[a]
                v, f = memberships[b]
                if u != v and e != f and not h[u, f] and not h[v, e]:
                    h[u, e] = h[v, f] = False
                    h[u, f] = h[v, e] = True
                    memberships[a, 1], memberships[b, 1] = f, e
                    switches += 1
            if not np.array_equal(h.sum(axis=0), sizes) or not np.array_equal(h.sum(axis=1), degrees):
                raise RuntimeError("Rewire margins changed")
        elif variant.control == "uniform":
            h[:] = False
            for edge, size in enumerate(sizes):
                h[rng.choice(len(h), int(size), replace=False), edge] = True
        result.append(replace(family, incidence=pd.DataFrame(h, index=family.incidence.index, columns=family.incidence.columns),
                       attributes={edge:{"members":tuple(family.incidence.index[h[:,i]]),"control":variant.control,
                                          "weight":family.attributes.get(edge,{}).get("weight",1.)}
                                   for i,edge in enumerate(family.edge_ids)},
                       diagnostics={**family.diagnostics, "control": variant.control, "switches": switches,
                                    "changed_fraction": float(np.mean(h != family.incidence.to_numpy()))},
                       state={**family.state, "original_attributes":family.attributes,"control_seed": seed+701}))
    return replace(snapshot, snapshot_id=snapshot.snapshot_id + f"/{variant.control}/{variant.parameter}/{seed}", families=tuple(result), learned_references={})


def ph_context(config, dataset):
    """Completed historical sessions only, exported as a separate causal stream."""
    from hyperedges.ph_localized.constructor import persistence
    from .storage import atomic_npz
    directory = config.root / "ph_context" / dataset.interval
    path = directory / "stream.npz"
    if path.exists():
        with np.load(path) as saved:
            return saved["values"], saved["mask"], saved["available_at"]
    values = np.zeros((len(dataset.session_dates), 15), np.float32)
    mask = np.zeros_like(values, bool)
    availability = np.full(len(values), np.iinfo(np.int64).max, np.int64)
    for session in range(1, len(values)):
        record = directory / "diagrams" / f"session-{session}.npz"
        begin = np.searchsorted(dataset.sessions, max(0, session-(126 if dataset.interval == "1d" else 20)))
        end = np.searchsorted(dataset.sessions, session)
        availability[session] = dataset.starts[end]
        if end-begin < 32:
            continue
        if record.exists():
            with np.load(record) as saved:
                vector = saved["summary"]
        else:
            source = np.linspace(begin, end-1, min(256, end-begin), dtype=int)
            x = np.array(dataset.features[source, :, 0])
            valid = dataset.feature_mask[source, :, 0]
            mean = np.where(valid, x, 0).sum(axis=0)/np.maximum(valid.sum(axis=0), 1)
            scale = np.sqrt(np.where(valid, (x-mean)**2, 0).sum()/max(valid.sum(), 1)) or 1
            cloud = np.c_[np.where(valid, (x-mean)/scale, 0), .25*valid]
            # Farthest point landmarks are chosen within this strictly past cloud.
            from hyperedges.ph_localized.constructor import _landmarks
            rows = _landmarks(cloud, 64, np.random.default_rng(config.constructor_seed))
            diagrams = persistence(cloud[rows])
            vector = []
            for dim in range(3):
                life = np.diff(diagrams[dim], axis=1).ravel()
                probabilities = life/life.sum() if life.sum() > 0 else np.zeros_like(life)
                vector.extend([len(life), float(life.sum()), float(life.max()) if len(life) else 0,
                               float(life.mean()) if len(life) else 0,
                               float(-np.sum(probabilities[probabilities > 0]*np.log(probabilities[probabilities > 0])))])
            atomic_npz(record, summary=np.array(vector, np.float32), source_times=dataset.times[source][rows],
                       **{f"H{dim}": diagram for dim, diagram in diagrams.items()})
        values[session], mask[session] = vector, True
        if session % 50 == 0:
            event(config, "PH_context", interval=dataset.interval, completed_session=session, total=len(values))
    atomic_npz(path, values=values, mask=mask, available_at=availability)
    return values, mask, availability
