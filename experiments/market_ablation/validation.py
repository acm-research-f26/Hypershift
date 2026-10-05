"""Scientific launch gates, including planted stock-coordinate recovery."""
from time import perf_counter
import numpy as np
import pandas as pd
from .storage import atomic_json, atomic_npz, event, read_json


def synthetic_cloud(kind, seed, missing=False, points=64):
    rng = np.random.default_rng(seed)
    if kind == "circle":
        angle = np.linspace(0, 2*np.pi, points, endpoint=False)
        planted = np.c_[np.cos(angle), np.sin(angle)]
        dimension = 1
    elif kind == "sphere":
        z = 1 - 2*(np.arange(points)+.5)/points
        angle = np.arange(points)*np.pi*(3-np.sqrt(5))
        planted = np.c_[np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z]
        dimension = 2
    elif kind == "torus":
        side=int(np.sqrt(points))
        a, b = np.meshgrid(np.arange(side)*2*np.pi/side, np.arange(side)*2*np.pi/side)
        planted = np.c_[np.cos(a.ravel()), np.sin(a.ravel()), np.cos(b.ravel()), np.sin(b.ravel())]
        dimension = 2
    elif kind == "null":
        return rng.normal(size=(points, 8)), set(), None, None
    else:
        raise ValueError(kind)
    planted += rng.normal(0, .008, planted.shape)
    nuisance = rng.normal(0, .025, (len(planted), 8-planted.shape[1]))
    cloud = np.c_[planted, nuisance]
    permutation = rng.permutation(8)
    cloud = cloud[:, permutation]
    true = set(np.flatnonzero(permutation < planted.shape[1]).tolist())
    groups = None
    if missing:
        valid = rng.random(cloud.shape) >= .01
        cloud = np.c_[np.where(valid, cloud, 0), .25*valid]
        groups = [np.array([i, i+8]) for i in range(8)]
    return cloud, true, dimension, groups


def validate_ph(config):
    from hyperedges.ph_localized.constructor import localize_cloud
    directory = config.root / "validation/ph-v2"
    trials = []
    started = perf_counter()
    for kind in ("circle", "sphere", "torus", "null"):
        for seed in range(10):
            for missing in ((False, True) if kind != "null" else (False,)):
                path = directory / f"{kind}-{seed}-missing{int(missing)}.json"
                saved = read_json(path)
                if saved is None:
                    x, truth, dim, groups = synthetic_cloud(kind, seed, missing)
                    edges, state = localize_cloud(x, coordinate_groups=groups, seed=seed + 1709)
                    candidates = [set(e["coordinates"]) for e in edges if e["dimension"] == dim]
                    best = max(candidates, key=lambda s: len(s & truth)/max(1, len(s | truth)), default=set())
                    precision, recall = len(best & truth)/max(1, len(best)), len(best & truth)/max(1, len(truth))
                    saved = {"kind": kind, "seed": seed, "missing": missing, "truth": sorted(truth), "edges": edges,
                             "precision": precision, "recall": recall, "success": precision >= .8 and recall >= .8,
                             "false_positive": bool(edges) if kind == "null" else False, "state": state}
                    atomic_json(path, saved)
                    atomic_npz(directory / f"{kind}-{seed}-missing{int(missing)}.npz", cloud=x)
                    event(config, "validate_PH", kind=kind, seed=seed, missing=missing,
                          precision=precision, recall=recall, edges=len(edges))
                trials.append(saved)
    recovery = {kind: float(np.mean([t["success"] for t in trials if t["kind"] == kind and not t["missing"]]))
                for kind in ("circle", "sphere", "torus")}
    missing_recovery = {kind: float(np.mean([t["success"] for t in trials if t["kind"] == kind and t["missing"]]))
                        for kind in recovery}
    false_positive = float(np.mean([t["false_positive"] for t in trials if t["kind"] == "null"]))
    passed = min((*recovery.values(), *missing_recovery.values())) >= .8 and false_positive <= .1
    result = {"passed": passed, "recovery": recovery, "null_false_positive_rate": false_positive,
              "missing_recovery": missing_recovery, "missingness_gate_required": True, "seconds": perf_counter()-started,
              "limitation": "Geometric localization does not guarantee recovery of arbitrary dependence such as XOR."}
    atomic_json(config.root / "validation/ph_gate.json", result)
    if not passed:
        raise RuntimeError(f"PH recovery gate failed: {result}")
    return result


def validate_cover(config):
    saved = read_json(config.root / "validation/cover_gate.json")
    if saved and saved.get("passed"):
        return saved
    from hyperedges.cover_learning.constructor import build_cover_neighborhood_graph, fit_published_cover, export_cover_memberships, stock_cover_matrix, CoverGraph
    from shapediscover.weighted_graph import WeightedGraph
    rng = np.random.default_rng(19)
    frame = pd.DataFrame(rng.normal(size=(40, 8)), index=[f"stock-{i}" for i in range(40)])
    frame.attrs["seed"] = 19
    recipe = {"neighbors": 5, "algorithm": "umap", "backend_version": "0.1.0"}
    graph = build_cover_neighborhood_graph(frame, recipe)
    first = fit_published_cover(frame, graph, {"name": "shapediscover", "version": "0.1.0"},
                               {"preset": "ShapeDiscover", "parameters": {"n_cover": 8, "loss_weights": [1,10,1,10],
                               "initialization_algorithm": "spectral_clustering", "model": "set_function", "n_max_iter": 20}})
    # Permute graph identity while holding descriptors, optimizer and seed fixed.
    p = rng.permutation(40)
    adjacency = graph.graph.adjacency_matrix().toarray()
    from scipy.sparse import csr_matrix
    changed = CoverGraph(WeightedGraph(csr_matrix(adjacency[p][:, p])), recipe, 19)
    second = fit_published_cover(frame, changed, {"name": "shapediscover", "version": "0.1.0"},
                                {"preset": "ShapeDiscover", "parameters": {"n_cover": 8, "loss_weights": [1,10,1,10],
                                "initialization_algorithm": "spectral_clustering", "model": "set_function", "n_max_iter": 20}})
    rule = {"threshold": .5, "min_size": 2}
    groups = export_cover_memberships(first, rule, node_ids=tuple(frame.index))
    # Membership comparison is invariant to arbitrary cover-slot permutations.
    from scipy.optimize import linear_sum_assignment
    h1, h2 = stock_cover_matrix(first, len(frame)) >= .5, stock_cover_matrix(second, len(frame)) >= .5
    distance = np.logical_xor(h1[:, :, None], h2[:, None, :]).mean(axis=0)
    rows, columns = linear_sum_assignment(distance)
    difference = float(distance[rows, columns].mean())
    result = {"passed": bool(groups) and difference > 0,
              "supplied_graph_consumed": first.graph_ is graph.graph and second.graph_ is changed.graph,
              "graph_sensitivity": difference, "groups": groups, "backend": "shapediscover@7b9aa1631c6893f3f08d3ce9f0310d4d2b934567"}
    atomic_npz(config.root / "validation/cover_graphs.npz", descriptors=frame.to_numpy(), graph=adjacency,
               alternate_graph=adjacency[p][:, p], memberships=first.cover_, alternate_memberships=second.cover_)
    atomic_json(config.root / "validation/cover_gate.json", result)
    if not result["passed"] or not result["supplied_graph_consumed"]:
        raise RuntimeError(f"Cover graph validation failed: {result}")
    return result


def validate(config):
    validate_cover(config)
    validate_ph(config)
    validate_landmarks(config)
    event(config, "constructor_gates_passed")


def validate_landmarks(config):
    """Use a larger source pool so the 64-point landmark sets actually differ."""
    from hyperedges.ph_localized.constructor import localize_cloud
    path=config.root/"validation/landmark_gate.json"
    saved=read_json(path)
    if saved and saved.get("passed"):
        return saved
    trials=[]
    for kind in ("circle","sphere","torus","null"):
        for seed in range(10):
            record_path=config.root/"validation/landmarks"/f"{kind}-{seed}.json"
            record=read_json(record_path)
            if record is None:
                x,truth,dim,groups=synthetic_cloud(kind,seed,points=256 if kind=="torus" else 128)
                edges,state=localize_cloud(x,coordinate_groups=groups,seed=seed+3251)
                candidates=[set(e["coordinates"]) for e in edges if e["dimension"]==dim]
                best=max(candidates,key=lambda s:len(s&truth)/max(1,len(s|truth)),default=set())
                precision,recall=len(best&truth)/max(1,len(best)),len(best&truth)/max(1,len(truth))
                changed=len({tuple(run["rows"]) for run in state["runs"]})>1
                record={"kind":kind,"seed":seed,"precision":precision,"recall":recall,
                        "success":precision>=.8 and recall>=.8 and changed,"landmark_sets_differ":changed,
                        "false_positive":bool(edges) if kind=="null" else False,"state":state,"edges":edges}
                atomic_json(record_path,record)
                event(config,"validate_landmarks",kind=kind,seed=seed,precision=precision,recall=recall,sets_differ=changed)
            trials.append(record)
    recovery={kind:float(np.mean([t["success"] for t in trials if t["kind"]==kind])) for kind in ("circle","sphere","torus")}
    null=float(np.mean([t["false_positive"] for t in trials if t["kind"]=="null"]))
    result={"passed":min(recovery.values())>=.8 and null<=.1,"recovery":recovery,"null_false_positive_rate":null,
            "max_landmarks":64,"source_pools":{"circle":128,"sphere":128,"torus":256,"null":128}}
    atomic_json(path,result)
    if not result["passed"]:
        raise RuntimeError(f"Landmark stability gate failed: {result}")
    return result
