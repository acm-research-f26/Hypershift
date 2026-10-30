import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from ndx_portfolio_robustness import band_target, average_targets, simulate_banded
from ndx_weak_market_short import simulate_signed
from ndx_extended_hours import ExtendedModel
from ndx_shared_think import SharedThink


class PortfolioRobustnessTests(unittest.TestCase):
    def test_bands_use_current_positions_and_force_ineligible_exits(self):
        target = np.array([.3, .3, 0., -.3])
        current = np.array([.295, .33, .01, -.29])
        actual = band_target(target, current, [True, True, False, True], .02)
        np.testing.assert_allclose(actual, [.295, .3, 0., -.29])
        np.testing.assert_array_equal(band_target(target, current, [1,1,0,1], 0), target)

    def test_budget_priority_and_forced_sign_change(self):
        result = band_target([.5, -.5], [.51, -.51], [1,1], .02)
        self.assertAlmostEqual(np.abs(result).sum(), 1.)
        np.testing.assert_allclose(band_target([.1, -.1], [-.1,.1], [1,1], .3), [.1,-.1])

    def test_average_targets_preserves_gross_and_does_not_pool_rankings(self):
        w = np.array([[[.6,0,-.4]], [[0,.8,-.2]]])
        np.testing.assert_allclose(average_targets(w), [[.3,.4,-.3]])
        with self.assertRaises(ValueError): average_targets(w*2)

    def test_zero_band_parity_and_causal_order_sizing(self):
        dates = pd.bdate_range('2024-01-01', periods=9)
        marks = np.array([[100+i,200-i,150+2*i] for i in range(9)], float)
        data = dict(panel=SimpleNamespace(dates=dates,tickers=['A','B','QQQ']),
                    marks=marks, adv=np.full_like(marks,1e10), tradable=np.ones_like(marks,bool),
                    origin=np.array([0,3,6]), entry=np.array([1,4,7]), events=[])
        w=np.array([[.4,.3,-.3],[.3,.4,-.3],[.5,.2,-.3]])
        old, path, _=simulate_signed(data,np.arange(3),w,detailed=True)
        new, actual, _, effective, _=simulate_banded(data,np.arange(3),w,np.ones_like(w,bool))
        np.testing.assert_allclose(actual,path,rtol=0,atol=1e-14)
        np.testing.assert_array_equal(effective,w)
        self.assertAlmostEqual(old['terminal_nav'],new['terminal_nav'])
        _,_,_,_,d1=simulate_banded(data,np.arange(3),w,np.ones_like(w,bool),.025)
        changed=dict(data); changed['marks']=marks.copy(); changed['marks'][5:]*=1.2
        _,_,_,_,d2=simulate_banded(changed,np.arange(3),w,np.ones_like(w,bool),.025)
        np.testing.assert_array_equal(d1[1]['target'],d2[1]['target'])
        np.testing.assert_array_equal(d1[1]['current'],d2[1]['current'])

    def test_shared_ablation_retains_matched_parameter_initialization(self):
        torch.manual_seed(7); old=ExtendedModel(4,[[0,1],[2,3]],'after_hours_premarket')
        torch.manual_seed(7); new=SharedThink(4,[[0,1],[2,3]],'after_hours_premarket')
        original=old.state_dict()
        self.assertFalse(hasattr(new.core,'market'))
        for name,value in new.state_dict().items(): torch.testing.assert_close(value,original[name],rtol=0,atol=0)
        # Duplicated stock histories within the same graph role get the same
        # score; no arbitrary output vector assigns them different identities.
        x=torch.randn(2,16,1,5).expand(-1,-1,4,-1).clone()
        extra=torch.randn(2,1,320).expand(-1,4,-1).clone()
        new.eval()
        with torch.no_grad(): score=new(x,extra,torch.ones(2,4))[0]
        torch.testing.assert_close(score,score[:,0,None].expand_as(score),rtol=0,atol=1e-6)


if __name__ == '__main__': unittest.main()
