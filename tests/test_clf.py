from hypershift.config import RunConfig
from hypershift.train.clf import tertile_thresholds, train_clf_run


def test_clf_runs(synthetic_market, synthetic_hypergraph, tmp_path):
    th = tertile_thresholds(synthetic_market, 8)
    assert th[0] < th[1]
    cfg = RunConfig(exp="E11_clf", label="t", seq=8, kernel=2, hidden=8, epochs=2, device="cpu", out_root=str(tmp_path))
    m = train_clf_run(cfg, synthetic_market, synthetic_hypergraph)
    assert 0.0 <= m["test_f1"] <= 1.0
