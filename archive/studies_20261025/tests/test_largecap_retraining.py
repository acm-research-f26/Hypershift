import unittest
import numpy as np
from run_robust_research import model_data
from robust_engine import backtest
from research_engine import prepare_panel,fold_indices
from test_research import fixture

class LargecapTests(unittest.TestCase):
    def test_baseline_history_excludes_only_unavailable_scores(self):
        data=prepare_panel(fixture())
        _,validation,_=fold_indices(data,2017)
        data['momentum'][validation,0]=np.nan
        baseline=model_data(data,'momentum126')
        self.assertFalse(baseline['mask'][validation,0].any())
        self.assertFalse(baseline['labelmask'][validation,0].any())
        self.assertTrue(data['mask'][validation,0].all())
        self.assertIs(model_data(data,'think'),data)
        metrics,path,_=backtest(baseline,validation,data['momentum'][validation],5,'confidence')
        self.assertTrue(np.isfinite(path.net).all())
        future_change=dict(data,momentum=data['momentum'].copy())
        future_change['momentum'][validation[-1]+1:]=np.nan
        np.testing.assert_array_equal(baseline['mask'][validation],model_data(future_change,'momentum126')['mask'][validation])

if __name__=='__main__':unittest.main()
