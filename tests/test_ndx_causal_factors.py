import unittest
import numpy as np
import pandas as pd
from ndx_causal_factors import factor_scores, eligible_percentiles, factor_blend


class CausalFactorTests(unittest.TestCase):
    def test_unknown_histories_are_neutral_and_ineligible_stocks_zero(self):
        actual=eligible_percentiles(np.array([[1.,2.,np.nan,4.]]),np.array([[1,1,1,0]],bool))
        np.testing.assert_allclose(actual,[[.5,1.,.5,0.]])

    def test_future_prices_cannot_change_prior_factor_scores(self):
        rng=np.random.default_rng(42);price=100*np.exp(np.cumsum(rng.normal(0,.02,(330,6)),axis=0))
        def data(p):
            return dict(marks=p,returns=pd.DataFrame(p).pct_change().to_numpy(),
                adv=np.ones_like(p)*1e7,origin=np.array([280,290,310]),mask=np.ones((3,6),bool))
        first=factor_scores(data(price),np.arange(3));changed=price.copy();changed[300:]*=10
        second=factor_scores(data(changed),np.arange(3))
        for name in first: np.testing.assert_array_equal(first[name][:2],second[name][:2])

    def test_blending_has_declared_weights_and_rejects_bad_inputs(self):
        a=np.array([[.2,.8]]);b=np.array([[.6,.4]])
        np.testing.assert_allclose(factor_blend(a,b,.25),[[.3,.7]])
        with self.assertRaises(ValueError):factor_blend(a,b,2)


if __name__=='__main__':unittest.main()
