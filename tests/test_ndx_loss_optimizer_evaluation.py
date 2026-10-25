import unittest
import numpy as np
from unittest.mock import patch
from ndx_loss_optimizer import select_learning_rates
from report_ndx_loss_optimizer import family,error_diversity


class EvaluationTests(unittest.TestCase):
    def test_residual_diversity_uses_raw_return_units_and_observed_labels(self):
        target=np.array([[10.,20.,30.,np.nan]])
        a=[np.array([[11.,18.,33.,1e9]]),np.array([[11.,18.,33.,-1e9]])]
        b=[np.array([[9.,22.,27.,0.]])]
        self.assertAlmostEqual(error_diversity(a,b,target,np.array([[True,True,True,False]])),-1.)

    def test_selection_uses_only_validation_and_lower_lr_tie(self):
        plan={'years':[2025],'pilot_seeds':[7,19],'learning_rates':{'adamw':[.0003,.001,.003]}}
        def fake(loss,opt,lr,year,seed):
            return {'validation_mse':{.0003:1.00000001,.001:1.,.003:2.}[lr], 'test_sharpe':1e9 if lr==.003 else -1e9}
        with patch('ndx_loss_optimizer.checked_record',side_effect=fake):result=select_learning_rates(plan)
        self.assertEqual(result['2025']['adamw']['lr'],.0003)

    def test_fixed_twenty_contrast_family(self):
        for group in ['ten_seed','original_five','new_five']:
            rows=family(group);self.assertEqual(len(rows),20);self.assertEqual(len({r[0] for r in rows}),20)
            self.assertTrue(any(r[0]=='pseudo-Huber minus Huber' for r in rows))


if __name__=='__main__':unittest.main()
