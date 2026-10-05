"""Reproducibility and actual-constructor controls before expensive forecasting."""
from dataclasses import replace
from time import perf_counter
import numpy as np
from hyperedges.common.pipeline import fit_hyperedge_pipeline, build_hyperedge_snapshot
from hyperedges.common.storage import save_hyperedge_snapshot
from .config import constructor_specs, core_variants, contrast_variants
from .constructors import history_context,cached_components
from .storage import read_json,atomic_json,event


def constructor_validation(config,dataset,fold):
    path=config.root/"validation/constructors"/dataset.interval/f"fold-{fold['id']}.json"
    if path.exists():
        return
    full=next(v for v in core_variants() if v.name=="A")
    reference,_=cached_components(config,dataset,fold,full,config.seeds[0])
    checks=[]
    for spec in constructor_specs(full,dataset.interval,config.gics):
        if spec.instance_id=="L":
            continue
        context=history_context(dataset,fold,config.constructor_seed,spec.history_window)
        start=perf_counter()
        repeated=build_hyperedge_snapshot(fit_hyperedge_pipeline(context,(spec,)))
        family=repeated.families[0]
        if spec.instance_id.startswith("E-"):
            original=next(f for f in reference.families if f.instance_id=="E")
            common=original.incidence.loc[:,family.edge_ids]
        else:
            original=next(f for f in reference.families if f.instance_id==spec.instance_id)
            common=original.incidence
        passed=family.incidence.equals(common)
        if not passed:
            raise RuntimeError(f"Constructor {spec.instance_id} is not reproducible at the declared seed")
        checks.append({"family":spec.instance_id,"passed":passed,"edges":len(family.edge_ids),
                       "sizes":family.incidence.sum(axis=0).tolist(),"coverage":float(family.incidence.any(axis=1).mean()),
                       "seconds":perf_counter()-start})
    cover=next(f for f in reference.families if f.instance_id=="C")
    knn=next(f for f in reference.families if f.instance_id=="K")
    hc,hk=cover.incidence.to_numpy(int),knn.incidence.to_numpy(int)
    intersection=hc.T@hk
    union=hc.sum(axis=0)[:,None]+hk.sum(axis=0)[None,:]-intersection
    similarity=intersection/np.maximum(union,1)
    best=similarity.max(axis=1) if similarity.shape[1] else np.zeros(similarity.shape[0])
    cover_sets={tuple(np.flatnonzero(hc[:,i])) for i in range(hc.shape[1])}
    knn_sets={tuple(np.flatnonzero(hk[:,i])) for i in range(hk.shape[1])}
    if cover_sets and cover_sets==knn_sets:
        raise RuntimeError("Cover Learning exactly reproduced the KNN groups")
    result={"passed":True,"checks":checks,"C_best_KNN_edge_Jaccard_mean":float(best.mean()) if len(best) else None,
            "C_best_KNN_edge_Jaccard_quantiles":np.quantile(best,[0,.5,1]).tolist() if len(best) else [],
            "PH_financial_empty":not len(next(f for f in reference.families if f.instance_id=="P").edge_ids),
            "reproducibility_unit":"constructor recipe on permitted first-fold history"}
    atomic_json(path,result)
    event(config,"constructor_validation_complete",interval=dataset.interval,fold=fold["id"])


def validate_control_recipes(config,dataset,fold):
    path=config.root/"validation/controls"/dataset.interval/f"fold-{fold['id']}.json"
    if path.exists():
        return
    variants=[v for v in contrast_variants() if v.control in ("cover_graph","window")]
    records=[]
    for variant in variants:
        snapshot,_=cached_components(config,dataset,fold,variant,config.seeds[0])
        save_hyperedge_snapshot(snapshot,path.parent/f"{variant.name}.snapshot.json")
        records.append({"variant":variant.name,"families":{f.instance_id:{"edges":len(f.edge_ids),"coverage":float(f.incidence.any(axis=1).mean())} for f in snapshot.families}})
    base,_=cached_components(config,dataset,fold,next(v for v in core_variants() if v.name=="C"),config.seeds[0])
    alternative,_=cached_components(config,dataset,fold,next(v for v in variants if v.name=="C-cover-return-pca"),config.seeds[0])
    x,y=base.families[0],alternative.families[0]
    if x.state["graph_node_ids"]!=y.state["graph_node_ids"]:
        raise RuntimeError("Alternative cover graph changed stock IDs")
    difference=float(np.mean(np.abs(x.state["graph_adjacency"]-y.state["graph_adjacency"])))
    if not difference>0:
        raise RuntimeError("Alternative cover representation did not change its supplied graph")
    from scipy.optimize import linear_sum_assignment
    memberships_x,memberships_y=x.state["cover"].T,y.state["cover"].T
    costs=np.mean(np.abs(memberships_x[:,:,None]-memberships_y[:,None,:]),axis=0)
    a,b=linear_sum_assignment(costs)
    membership_change=float(costs[a,b].mean())
    if membership_change<=1e-6:
        raise RuntimeError("Alternative supplied graph did not change the learned cover after slot matching")
    atomic_json(path,{"passed":True,"controls":records,"C_alternative_graph_mean_absolute_change":difference,
                     "C_slot_matched_membership_mean_absolute_change":membership_change})
