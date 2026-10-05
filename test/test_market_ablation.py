"""Launch-critical causal, masking, geometry and CUDA integration contracts."""
from dataclasses import replace
from itertools import combinations
import numpy as np
import pandas as pd
import pytest
import torch
from experiments.market_ablation.config import SweepConfig, core_variants, contrast_variants, manifest
from experiments.market_ablation.dataset import MarketDataset, aggregate
from experiments.market_ablation.statistics import hac_covariance, calibration, holm
from experiments.market_ablation.portfolio import desired_weights, risk_metrics, execution_path
from hyperbolicity.core import exact_delta_details, basepoint_bounds, four_point_delta
from hyperedges.common.types import HyperedgeSnapshot, HyperedgeFamily, ConstructorSpec
from hyperedges.learned_membership.constructor import LearnedMembershipConstructor
from src.hypergraph import HyperedgeConsumer
from test.test_hyperedge_pipeline import context


def test_manifest_preserves_design_and_matched_seeds():
    jobs = manifest(SweepConfig())["jobs"]
    assert len(core_variants())==35 and len(contrast_variants())==48
    assert sum(j["scope"]=="core" for j in jobs)==1260
    assert sum(j["scope"]=="latest_fold_contrast" for j in jobs)==384
    assert all(j["fold"]==2 and j["seed"] in (1001,2001) for j in jobs if j["scope"]!="core")


def test_ph_launch_rejects_failed_missingness_recovery(tmp_path, monkeypatch):
    from experiments.market_ablation import validation
    published = {}
    def cached(path):
        missing = path.stem.endswith("missing1")
        return {"kind":path.stem.split("-")[0], "missing":missing,
                "success":not missing, "false_positive":False}
    monkeypatch.setattr(validation,"read_json",cached)
    monkeypatch.setattr(validation,"atomic_json",lambda path,value:published.update(value))
    with pytest.raises(RuntimeError,match="PH recovery gate failed"):
        validation.validate_ph(replace(SweepConfig(),output=str(tmp_path)))
    assert all(value==1. for value in published["recovery"].values())
    assert all(value==0. for value in published["missing_recovery"].values())
    assert published["missingness_gate_required"] and not published["passed"]


def test_rotating_origins_preserve_minute_horizon_and_full_history():
    d = object.__new__(MarketDataset)
    d.config, d.interval = SweepConfig(), "1m"
    t,n = 100,250
    d.features = np.zeros((t,n,10),np.float32)
    d.feature_mask = np.ones_like(d.features,bool)
    d.valid = np.ones((t,n),bool)
    d.bars = np.ones((t,n,7),np.float32)
    d.bars[:,:,3] = np.arange(1,t+1)[:,None]
    d.features[:,:,0] = np.arange(t)[:,None]
    train = np.arange(19,90)
    schedules = [d.epoch_origins(train,e) for e in range(5)]
    assert np.array_equal(np.sort(np.concatenate(schedules)),train)
    moments = {"mean":np.zeros((n,10)),"std":np.ones((n,10)),"target_scale":np.array(1.)}
    x,active,y,mask = d.batch([30],core_variants()[0],moments)
    assert np.array_equal(x[0,0,0::2],np.arange(11,31))
    assert y[0,0]==pytest.approx(np.log(32/31))
    # An unavailable optional feature changes its feature mask, never target support.
    d.feature_mask[:,:,[1,2,3]] = False
    rich = next(v for v in contrast_variants() if v.features=="F")
    assert np.array_equal(d.batch([30],rich,moments)[3],mask)


def test_batched_consumer_matches_individual_origins_and_gradients(context):
    nodes = context.node_ids
    h = pd.DataFrame(np.array([[i in (0,1,2),i in (2,3,4)] for i in range(len(nodes))]),index=nodes,columns=["a","b"])
    family = HyperedgeFamily("K","fixture",h)
    snapshot = HyperedgeSnapshot("batch-test",nodes,(family,),context.cutoff,context.cutoff,context.cutoff)
    module = LearnedMembershipConstructor(ConstructorSpec("L","learned_membership",{"slots":3}),123)
    module.fit(context)
    snapshot=replace(snapshot,learned_references={"L":module.export_state()})
    model = HyperedgeConsumer(4,8,family_ids=("K","L"),learned_modules={"L":module})
    x = torch.randn(3,len(nodes),4)
    active = torch.ones(3,len(nodes),dtype=torch.bool)
    active[1,0]=False
    model.eval()
    batch = model(x,snapshot,active)
    individual = torch.stack([model(x[i],snapshot,active[i]) for i in range(3)])
    torch.testing.assert_close(batch,individual,rtol=1e-5,atol=1e-6)
    model.train()
    model(x,snapshot,active).square().mean().backward()
    assert module.logits.grad is not None and torch.isfinite(module.logits.grad).all()
    assert module.logits.grad.abs().sum()>0
    if torch.cuda.is_available():
        model.cuda(); model.zero_grad()
        model(x.cuda(),snapshot,active.cuda()).square().mean().backward()
        assert torch.isfinite(module.logits.grad).all()


def test_certified_bounds_contain_exact_and_chunked_matches_enumeration():
    rng = np.random.default_rng(3)
    for n in (4,8,15):
        x=rng.normal(size=(n,3))
        d=np.linalg.norm(x[:,None]-x[None,:],axis=2)
        exact = exact_delta_details(d)["delta"]
        expected=max(four_point_delta(d[np.ix_(q,q)]) for q in combinations(range(n),4))
        assert exact==pytest.approx(expected)
        bounds=basepoint_bounds(d)
        assert bounds["lower_bound"]<=exact+1e-10<=bounds["upper_bound"]+1e-10


def test_persistence_portfolio_is_cash_and_ratios_are_undefined():
    weights=desired_weights(np.zeros((5,250)),np.ones((5,250),bool),"long_only")
    assert not weights.any()
    assert not desired_weights(np.zeros((5,250)),np.ones((5,250),bool),"long_short").any()
    # Ranking portfolios can have both legs even when all return forecasts are positive.
    ranked=desired_weights(np.arange(1,251)[None,:],np.ones((1,250),bool),"long_short")
    assert ranked.sum()==pytest.approx(0) and np.abs(ranked).sum()==pytest.approx(1)
    risk=risk_metrics(np.zeros(30),np.zeros(30))
    assert risk["MaxDD"]==0 and risk["Sharpe"] is None and risk["IR"] is None and risk["Calmar"] is None


def test_dependence_covariance_scaling_and_calibration_rank():
    x=np.arange(30,dtype=float)
    cov=hac_covariance(x,0)
    assert cov["mean_covariance"][0,0]==pytest.approx(np.var(x)/len(x))
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
    assert calibration(np.c_[np.ones(30),np.zeros((30,4))])["reason"]=="constant_forecast"


def test_geometry_freezes_learned_family_without_duplicate_references(context, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.market_ablation import runner
    from hyperedges.common.storage import load_hyperedge_snapshot
    module = LearnedMembershipConstructor(ConstructorSpec("L","learned_membership",{"slots":3}),123)
    module.fit(context)
    snapshot = HyperedgeSnapshot("geometry-fixture",context.node_ids,(),context.cutoff,context.cutoff,context.cutoff,
                                learned_references={"L":module.export_state()})
    monkeypatch.setattr(runner,"cached_components",lambda *args:(snapshot,{"L":module}))
    config=replace(SweepConfig(),output=str(tmp_path))
    dataset=SimpleNamespace(interval="1m")
    variant=next(v for v in core_variants() if v.name=="L")
    # A crash after the geometry JSON must still repair the missing snapshot.
    path=tmp_path/"geometry/1m/fold-0/L.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    runner.geometry(config,dataset,{"id":0},variant)
    exported=load_hyperedge_snapshot(path.with_suffix(".snapshot.json"))
    assert not exported.learned_references
    assert [f.instance_id for f in exported.families]==["L"]


def test_portfolio_carries_unquoted_positions_and_is_self_financing():
    desired=np.array([[1.,0.],[0.,1.],[0.,1.]])
    visible=np.array([[True,True],[False,True],[True,True]])
    returns=np.array([[.1,0.],[0.,.2],[0.,0.]])
    sessions=np.zeros(3,np.int64)
    duration=np.zeros(3)
    path=execution_path(desired,visible,returns,sessions,duration,False,0.,0.)
    assert np.allclose(path[0],[.1,0.,0.])
    assert np.allclose(path[-1],[[1.,0.],[1.,0.],[0.,1.]])
    assert path[7][1]==pytest.approx(1.)
    # Invest against wealth remaining after costs: q=1-c*q, not q=1-c.
    single=execution_path(np.ones((1,1)),np.ones((1,1),bool),np.zeros((1,1)),np.zeros(1,np.int64),
                          np.zeros(1),False,100.,0.)
    assert single[0][0]==pytest.approx(1/1.01-1,abs=1e-10)


def test_cash_merger_releases_exposure_for_the_next_investment(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from experiments.market_ablation import portfolio
    bars=np.zeros((3,2,7),float)
    bars[:,:,0]=[[100,100],[0,100],[0,120]]
    data=SimpleNamespace(bars=bars,symbols=("A","B"),sessions=np.arange(3),
                         session_dates=np.array(["2020-01-01","2020-01-02","2020-01-03"]))
    actions=[{"id":"cash-merger","type":"cash_mergers","symbol":"A","effective_date":"2020-01-02","rate":110}]
    monkeypatch.setattr(portfolio,"read_json",lambda path,default=None:actions)
    total,quoted,unsupported,audit,flows=portfolio.daily_economic_returns(
        replace(SweepConfig(),output=str(tmp_path)),data,0,2,include_flows=True)
    assert total[0,0]==pytest.approx(.1) and quoted[0,0] and not unsupported.any()
    assert flows["capital_returns"][0,0]==-1
    cash=np.zeros(2)
    path=execution_path(np.array([[1.,0.],[0.,1.]]),bars[:2,:,0]>0,total,np.arange(2),np.zeros(2),False,0.,0.,
                        flows["capital_returns"],flows["liquidation_yields"],None,cash)
    np.testing.assert_allclose(path[-1],[[1.,0.],[0.,1.]])
    np.testing.assert_allclose(path[0],[.1,.2])
    np.testing.assert_allclose(cash,[1.1,0.])


def test_dividend_cash_reinvestment_pays_rebalancing_cost():
    desired=np.ones((2,1))
    total=np.array([[.1],[0.]])
    cash=np.zeros(2)
    path=execution_path(desired,np.ones((2,1),bool),total,np.arange(2),np.zeros(2),False,100.,0.,
                        np.zeros_like(total),np.zeros_like(total),None,cash)
    # The first dividend leaves 1/1.1 of wealth in stock and the rest in cash.
    expected=(1+.01/1.1)/1.01-1
    assert path[0][1]==pytest.approx(expected,abs=1e-10)
    assert path[2][1]>0 and path[5][1]>0
    assert cash[0]==pytest.approx(.1/1.01,abs=1e-10)


def test_stock_merger_liquidation_costs_and_missing_recipient_quotes(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from experiments.market_ablation import portfolio
    bars=np.zeros((2,2,7),float)
    bars[:,:,0]=[[100,100],[0,100]]
    data=SimpleNamespace(bars=bars,symbols=("A","B"),sessions=np.arange(2),
                         session_dates=np.array(["2020-01-01","2020-01-02"]))
    actions=[{"id":"mixed-merger","type":"stock_and_cash_mergers","acquiree_symbol":"A","acquirer_symbol":"B",
              "effective_date":"2020-01-02","acquirer_rate":.5,"acquiree_rate":1.,"cash_rate":10.}]
    monkeypatch.setattr(portfolio,"read_json",lambda path,default=None:actions)
    config=replace(SweepConfig(),output=str(tmp_path))
    total,quoted,unsupported,audit,flows=portfolio.daily_economic_returns(config,data,0,1,include_flows=True)
    assert not unsupported.any() and flows["liquidation_yields"][0,0]==.5
    corporate=np.zeros(1)
    path=execution_path(np.array([[1.,0.]]),np.ones((1,2),bool),total,np.zeros(1,np.int64),np.zeros(1),False,100.,0.,
                        flows["capital_returns"],flows["liquidation_yields"],corporate)
    assert path[0][0]==pytest.approx((.6-.005)/1.01-1,abs=1e-10)
    assert corporate[0]==pytest.approx(.005/1.01,abs=1e-10)
    bars[1,1,0]=0
    assert portfolio.daily_economic_returns(config,data,0,1)[2][0,0]


def test_multiple_model_inference_records_degenerate_spa_without_nan():
    from experiments.market_ablation.statistics import multiple_model_tests
    baseline=np.ones(30)
    result=multiple_model_tests(baseline,baseline[:,None],replicates=20)
    assert result["SPA_pvalues"] is None
    assert result["SPA_excluded_constant_loss_models"]==[0]
    assert result["Reality_Check_p"]==1.


def test_interrupted_fit_resumes_identical_optimizer_and_membership_rng(context, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.market_ablation import training
    module=LearnedMembershipConstructor(ConstructorSpec("L","learned_membership",{"slots":3}),123)
    module.fit(context)
    initial=module.export_state()
    def components(*args):
        learned=LearnedMembershipConstructor.from_export_state(initial)
        snapshot=HyperedgeSnapshot("resume-test",context.node_ids,(),context.cutoff,context.cutoff,context.cutoff,
                                   learned_references={"L":learned.export_state()})
        return snapshot,{"L":learned}
    monkeypatch.setattr(training,"cached_components",components)
    features=np.random.default_rng(100).normal(size=(260,len(context.node_ids),4)).astype(np.float32)
    targets=np.sin(np.arange(260)[:,None]+np.arange(len(context.node_ids))[None,:]).astype(np.float32)*.01
    class FixtureDataset:
        interval="1m"
        symbols=context.node_ids
        times=pd.date_range("2020-01-01",periods=260,freq="min",tz="UTC").as_unit("ns").asi8
        sessions=np.zeros(260,np.int64)
        session_dates=np.array(["2020-01-01"])
        bars=np.zeros((260,len(context.node_ids),7),np.float32)
        bars[:,:,3]=100*np.exp(np.r_[np.zeros((1,len(context.node_ids))),np.cumsum(targets[:-1],axis=0)])
        def __init__(self,config): self.config=config
        def split(self,fold): return {"train":np.arange(210),"validation":np.arange(210,230),"test":np.arange(230,250)}
        def moments(self,fold): return {"target_scale":np.array(1.,np.float32),"price_target_scale":np.array(1.,np.float32)}
        def batch(self,origins,*args):
            observed=np.ones((len(origins),len(self.symbols)),bool)
            return features[origins],observed,targets[origins],observed
        def tuning_origins(self,origins): return origins
        def epoch_origins(self,origins,*args,**kwargs): return origins
    config=replace(SweepConfig(),output=str(tmp_path/"uninterrupted"),lookback=2,epochs=2,device="cpu")
    fold={"id":0,"fit_cutoff":context.cutoff.isoformat()}
    variant=next(v for v in core_variants() if v.name=="L")
    settings={"hidden":8,"learning_rate":.001,"weight_decay":0.,"batch_size":2}
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training.fit(config,FixtureDataset(config),fold,variant,1001,settings)
        resumed=replace(config,output=str(tmp_path/"resumed"))
        original=training.atomic_checkpoint
        def interrupt_after_saved_batch(path,state):
            original(path,state)
            if path.name=="latest.pt" and state["batch_offset"]>0:
                raise RuntimeError("simulated interruption")
        monkeypatch.setattr(training,"atomic_checkpoint",interrupt_after_saved_batch)
        with pytest.raises(RuntimeError,match="simulated interruption"):
            training.fit(resumed,FixtureDataset(resumed),fold,variant,1001,settings)
        monkeypatch.setattr(training,"atomic_checkpoint",original)
        training.fit(resumed,FixtureDataset(resumed),fold,variant,1001,settings)
        def result_dir(c): return c.root/"fits/1m/fold-0/L/seed-1001"
        first=torch.load(result_dir(config)/"latest.pt",weights_only=False)
        second=torch.load(result_dir(resumed)/"latest.pt",weights_only=False)
        for key,value in first["model"].items():
            torch.testing.assert_close(value,second["model"][key],rtol=0,atol=0)
        assert first["curves"]==second["curves"]
        with np.load(result_dir(config)/"predictions/2020-01.npz") as a, np.load(result_dir(resumed)/"predictions/2020-01.npz") as b:
            assert np.array_equal(a["prediction"],b["prediction"])
    finally:
        torch.set_num_threads(previous_threads)


def test_golden_residual_moment_and_delta_method_reference_cases():
    from experiments.market_ablation.statistics import gaussian_error_diagnostics,skill_inference
    # Six residuals {0,0,0,0,-sqrt(3),sqrt(3)} have Gaussian moments through order four.
    moments=np.tile([6,0,6,0,18.],(30,1))
    result=gaussian_error_diagnostics(moments)
    assert result["gradient_infinity_norm"]==pytest.approx(0,abs=1e-12)
    assert result["residual_skewness"]==0 and result["residual_excess_kurtosis"]==0
    assert result["information_matrix_discrepancy"]["trace_A_inverse_B"]==pytest.approx(2)
    assert result["information_matrix_discrepancy"]["trace_B_inverse_A"]==pytest.approx(2)
    baseline=np.arange(1,31,dtype=float)
    skill=skill_inference(.8*baseline,baseline,replicates=40)
    assert skill["MSE_skill"]==pytest.approx(.2)
    for block in skill["paired_stationary_bootstrap"].values():
        assert block["ci95"]==pytest.approx([.2,.2])


def test_atomic_status_allows_independent_preparation_and_training_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from experiments.market_ablation.storage import atomic_json,read_json
    target=tmp_path/"progress.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda value:atomic_json(target,{"value":value}),range(40)))
    assert read_json(target)["value"] in range(40)
    assert not list(tmp_path.glob("*.partial"))


def test_raw_price_loss_matches_physical_target_and_nominal_price_weighting():
    from types import SimpleNamespace
    from experiments.market_ablation.training import training_error
    bars=np.zeros((2,2,7),np.float32)
    bars[0,:,3]=[100,10]
    bars[1,:,3]=[110,11]
    dataset=SimpleNamespace(bars=bars)
    moments={"target_scale":.01,"price_target_scale":2.}
    p=torch.zeros((1,2),requires_grad=True)
    error=training_error(SweepConfig(),dataset,[0],p,torch.zeros_like(p),moments)
    torch.testing.assert_close(error,torch.tensor([[-5.,-.5]]))
    error.square().sum().backward()
    assert p.grad[0,0]/p.grad[0,1]==pytest.approx(100.)
    perfect=torch.full((1,2),np.log(1.1)/.01)
    assert training_error(SweepConfig(),dataset,[0],perfect,torch.zeros_like(perfect),moments).abs().max()<1e-5


def test_evaluation_and_archive_resume_select_raw_price_mse(tmp_path):
    from experiments.market_ablation.training import evaluate
    class Dataset:
        config=replace(SweepConfig(),output=str(tmp_path),device="cpu")
        symbols=("A","B")
        times=pd.date_range("2020-01-01",periods=2,freq="min",tz="UTC").as_unit("ns").asi8
        sessions=np.zeros(2,np.int64)
        session_dates=np.array(["2020-01-01"])
        bars=np.zeros((2,2,7),np.float32)
        bars[:,:,3]=[[100,10],[110,20]]
        def batch(self,origins,*args):
            observed=np.ones((len(origins),2),bool)
            target=np.tile([np.log(1.1),np.log(2)],(len(origins),1)).astype(np.float32)/.01
            return np.zeros((len(origins),2,4),np.float32),observed,target,observed
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias=torch.nn.Parameter(torch.tensor([np.log(1.1)/.01,0],dtype=torch.float32))
        def forward(self,x,*args,**kwargs): return self.bias[None,:,None].expand(len(x),-1,-1)
    moments={"target_scale":np.array(.01,np.float32)}
    model=Model();dataset=Dataset();variant=core_variants()[0]
    score,contributions=evaluate(model,None,dataset,np.array([0]),variant,moments,1,directory=tmp_path/"predictions")
    assert score==pytest.approx(.5,abs=1e-6)
    assert contributions["loss_sum"][0]/contributions["persistence_loss_sum"][0]>.9
    resumed,_=evaluate(model,None,dataset,np.array([0]),variant,moments,1,directory=tmp_path/"predictions")
    assert resumed==pytest.approx(score)


def test_analysis_cache_identity_tracks_forecasts_and_recipient_valuations(tmp_path):
    from types import SimpleNamespace
    from experiments.market_ablation.reporting import analysis_identity
    from experiments.market_ablation.storage import atomic_json
    config=replace(SweepConfig(),output=str(tmp_path),gics=str(tmp_path/"gics.csv"))
    dataset=SimpleNamespace(root=tmp_path/"dataset/1m",interval="1m")
    job=manifest(config)["jobs"][0]
    first=analysis_identity(config,dataset,job)
    assert analysis_identity(config,dataset,job)==first
    predictions=tmp_path/"fits"/job["id"]/"predictions/2024-01.npz"
    predictions.parent.mkdir(parents=True)
    np.savez(predictions,prediction=np.zeros((1,250),np.float32))
    second=analysis_identity(config,dataset,job)
    assert second!=first
    recipient=tmp_path/"corporate_actions/recipient_prices/B-2024-01-02.json"
    atomic_json(recipient,{"bars":[]})
    third=analysis_identity(config,dataset,job)
    atomic_json(recipient,{"bars":[{"o":10.}]})
    assert analysis_identity(config,dataset,job)!=third
