import unittest
import numpy as np
from rishi_mechanism_selection import tail_weights,observed_label_return,stratified_draws,active_ir,covariance_terms,asof_style


class SelectionTests(unittest.TestCase):
    def test_fractional_tail_ties_preserve_exact_budget_and_mask(self):
        score=np.array([3.,2.,2.,0.,np.nan]);mask=np.array([1,1,1,1,0],bool)
        np.testing.assert_allclose(tail_weights(score,mask,.375),[2/3,1/6,1/6,0,0])
        np.testing.assert_allclose(tail_weights(np.ones(4),np.ones(4,bool)),.25)

    def test_missing_label_does_not_rerank_selected_stock(self):
        ret,coverage=observed_label_return(np.array([1.,0.]),np.array([999.,.05]),np.array([0,1],bool))
        self.assertTrue(np.isnan(ret));self.assertEqual(coverage,0)

    def test_draws_stay_in_annual_fitted_strata(self):
        years=np.array([2022,2022,2022,2023,2023])
        a=stratified_draws(years,20,4,17);b=stratified_draws(years,20,4,17)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(years[a],np.broadcast_to(years,a.shape))

    def test_active_ir_does_not_subtract_risk_free_rate(self):
        left=np.array([.01,-.01,.02]);right=np.zeros(3)
        self.assertAlmostEqual(float(active_ir(left,right)),left.mean()/left.std(ddof=1)*np.sqrt(252))
        self.assertTrue(np.isnan(active_ir(right,right)))

    def test_shared_target_covariance_and_baseline_invariance(self):
        rng=np.random.default_rng(7);b=rng.normal(size=100);d=rng.normal(size=100);y=rng.normal(size=100)*10;h=rng.normal(size=100)
        result=covariance_terms(b,d,y,h)
        self.assertAlmostEqual(result['error_covariance'],result['forecast_covariance']-result['b_target_covariance']-result['d_target_covariance']+result['target_variance'])
        self.assertLess(result['common_baseline_error_invariance_max'],1e-12)

    def test_style_features_cannot_see_future_data(self):
        rng=np.random.default_rng(19);returns=rng.normal(0,.01,(300,2));marks=100*np.cumprod(1+returns,axis=0)
        data=dict(returns=returns,marks=marks,adv=np.full_like(returns,1e8),tradable=np.ones_like(returns,bool))
        q=rng.normal(0,.01,300);first=asof_style(data,220,q)
        changed={k:v.copy() for k,v in data.items()}
        for name in ['returns','marks','adv']:changed[name][221:]=999.
        altered=q.copy();altered[221:]=999.
        np.testing.assert_array_equal(first,asof_style(changed,220,altered))


if __name__=='__main__':unittest.main()
