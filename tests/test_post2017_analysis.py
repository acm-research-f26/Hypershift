import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("post2017_analysis", Path(__file__).resolve().parents[1] / "scripts" / "post2017_analysis.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def test_date_join_is_explicit_not_tail_aligned():
    da = np.array(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"], dtype="datetime64[D]")
    db = np.array(["2020-01-03", "2020-01-06", "2020-01-07"], dtype="datetime64[D]")     # missing 01-02
    ra, rb = np.array([1., 2., 3., 4.]), np.array([20., 30., 40.])
    d, x, y, na, nb = A.date_join(da, ra, db, rb)
    assert d.tolist() == db.tolist() and x.tolist() == [2., 3., 4.] and y.tolist() == [20., 30., 40.] and (na, nb) == (1, 0)
    with pytest.raises(ValueError):
        A.date_join(np.array(["2020-01-02", "2020-01-02"], dtype="datetime64[D]"), [1., 2.], db[:2], [1., 2.])


def test_pooled_sharpe_not_mean_of_annual():
    rng = np.random.default_rng(0)
    r = np.concatenate([rng.normal(0.003, 0.01, 252), rng.normal(-0.002, 0.03, 252)])
    dates = np.concatenate([np.full(252, "2018-06-01"), np.full(252, "2019-06-01")]).astype("datetime64[D]")
    ann = A.annual(dates, r)
    assert A.pooled_sharpe(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert A.pooled_sharpe(r) != pytest.approx(np.mean([ann[2018]["sr"], ann[2019]["sr"]]))
    assert ann[2018]["n"] == 252
    loyo = A.leave_one_year_out(dates, r)
    assert loyo[2018] == pytest.approx(A.pooled_sharpe(r[252:]))


def test_deterministic_boot_and_contrast():
    r = np.random.default_rng(1).normal(0.001, 0.01, 300)
    i1 = A.boot_indices(300, 50, 10, np.random.default_rng(5))
    i2 = A.boot_indices(300, 50, 10, np.random.default_rng(5))
    assert (i1 == i2).all()
    c = A.boot_contrast(r, r, i1)
    assert c["est"] == 0 and c["lo"] == 0 and c["hi"] == 0


def test_costs_reduce_sharpe():
    import hypershift.eval.forensics as F
    r = np.random.default_rng(2).normal(0.002, 0.01, 200)
    to = np.ones(200)
    assert A.pooled_sharpe(F.net(r, to, 10)) < A.pooled_sharpe(r)
    assert np.allclose(F.net(r, to, 0), r)
