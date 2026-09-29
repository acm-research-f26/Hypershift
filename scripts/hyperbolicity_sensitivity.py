"""Sensitivity of hyperbolicity (delta_hg, delta_rel) to graph variant, s, sampling and feature choice. NYSE. Run from repo root."""
import json, numpy as np, time
from pathlib import Path
from hypershift.data.hypergraph import build_rsr_hypergraph, Hypergraph, canonical, industry_hyperedges
from hypershift.data.rsr import load_rsr, read_ticker_file
from hypershift.geometry.hyperbolicity import s_distance_matrix, largest_component, gromov_delta, sampled_delta, feature_hyperbolicity
R=Path("data/raw/rsr/data")
def hg_stats(name, hg, s=1, bases=(0,)):
    D=s_distance_matrix(hg,s); c=largest_component(D); D=D[np.ix_(c,c)]
    vals=[gromov_delta(D,base=int(b)%len(c)) for b in bases]
    fin=D[np.isfinite(D)]
    print(f"{name:42s} s={s} LCC={len(c):4d} diam={fin.max():.0f} meanDist={fin[fin>0].mean():.2f} delta(exact, {len(bases)} base pts)={max(vals):.1f} per-base={vals}", flush=True)
rng=np.random.default_rng(0)
for m in ["NYSE"]:
    hg=build_rsr_hypergraph(R,m); n=hg.num_nodes
    bases=tuple(int(b) for b in rng.choice(1600,4,replace=False))
    hg_stats(f"{m} industry+wiki (ours)",hg,1,bases)
    rel=np.load(R/"relation/sector_industry"/f"{m}_industry_relation.npy")
    tick=read_ticker_file(R/f"{m}_tickers_qualify_dr-0.98_min-5_smooth.csv")
    j=json.load(open(R/"relation/sector_industry"/f"{m}_industry_ticker.json")); pos={t:i for i,t in enumerate(tick)}
    na=set(pos[t] for t in j.get("n/a",[]) if t in pos)
    no_na=Hypergraph(n, canonical([e for e in hg.edges if not (len(e)>=400 and set(e)<=na)]))
    hg_stats(f"{m} without n/a bucket",no_na,1,bases)
    ind=Hypergraph(n, canonical(industry_hyperedges(rel)))
    hg_stats(f"{m} industry only",ind,1,bases)
    hg_stats(f"{m} industry+wiki, s=2",hg,2,bases[:2])
    # sampling variability of the sampled estimator used in our table
    print("  sampled(1000 nodes) over 5 seeds:",[sampled_delta(s_distance_matrix(hg,1)[np.ix_(*(2*[largest_component(s_distance_matrix(hg,1))]))],1000,1,seed=k)["delta_max"] for k in range(5)])
    d=load_rsr(R,m,"train"); vi=d.valid_index
    feats={"train daily returns (ours)":np.where(d.mask[:,1:vi]>0,d.gt[:,1:vi],0),
           "train normalized close series":d.features[:,:vi,-1],
           "train 5-feature series flattened":d.features[:,:vi,:].reshape(n,-1),
           "last-16-day window features (model input)":d.features[:,vi-16:vi,:].reshape(n,-1)}
    for k,X in feats.items():
        r=[feature_hyperbolicity(X,1000,1,seed=s)["delta_rel"] for s in range(3)]
        print(f"  delta_rel {k:45s} {np.mean(r):.3f} (seeds {np.round(r,3).tolist()})",flush=True)
    D=s_distance_matrix(hg,1); c=largest_component(D); D=D[np.ix_(c,c)]
    for k in (30,50,100,200,500):
        v=[sampled_delta(D,k,1,seed=i)["delta_max"] for i in range(20)]
        print(f"  sampled {k:4d} nodes x20: mean delta {np.mean(v):.2f}, share <=0.5: {np.mean(np.array(v)<=0.5):.2f}")
