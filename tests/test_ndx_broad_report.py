"""Verify selection isolation, selected-tree prefixes and the QQQ risk control."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import run_ndx_broad as training
from report_ndx_broad import qqq_adapter
from ndx_policy_engine import policy_backtest,policy_replay
from test_pit_research_engine import fixture


class EvaluationTests(unittest.TestCase):
    def test_changing_all_test_predictions_cannot_change_annual_validation_choices(self):
        base=Path(__file__).resolve().parent/'tmp'; base.mkdir(exist_ok=True)
        plan=dict(test_years=[2022],views=['daily','intraday'],targets=['raw','residual'],
                  families=['ridge','boosted_trees','shared_network'],seeds=[7])
        data=dict(dates=pd.date_range('2021-01-01',periods=12),origin=np.arange(12),mask=np.ones((12,5),bool),
                  labelmask=np.ones((12,5),bool),targets={'residual':np.tile(np.arange(5),(12,1))})
        rng=np.random.default_rng(7)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root=Path(temporary).resolve(); self.assertTrue(root.is_relative_to(base.resolve()))
            paths=[]
            for view in plan['views']:
                folder=root/view/'2022'; folder.mkdir(parents=True)
                for target in plan['targets']:
                    for family in plan['families']:
                        path=folder/f'{family}_{target}_7_predictions.npz'; paths.append(path)
                        np.savez_compressed(path,validation=rng.normal(size=(12,5)),test=rng.normal(size=(2,5)))
            with patch.object(training,'RUN',root),patch.object(training,'splits',return_value=(np.array([],int),np.arange(12),np.array([],int))):
                training.lock_selection(data,plan)
                before=json.loads((root/'SELECTION_LOCK.json').read_text())
                (root/'SELECTION_LOCK.json').unlink()
                for path in paths:
                    with np.load(path) as saved:
                        validation=saved['validation'].copy()
                    np.savez_compressed(path,validation=validation,test=rng.normal(size=(2,5))*1e20)
                training.lock_selection(data,plan)
                after=json.loads((root/'SELECTION_LOCK.json').read_text())
            self.assertEqual(before['choices'],after['choices'])
            self.assertTrue(before['created_before_new_financial_metrics'])
            self.assertNotEqual(before['validation_artifact_hashes'],after['validation_artifact_hashes'])

    def test_selected_tree_prefix_matches_independently_fitted_short_model(self):
        rng=np.random.default_rng(42); x=rng.normal(size=(300,4)); y=x[:,0]**2+.3*x[:,1]
        args=dict(learning_rate=.1,max_leaf_nodes=7,min_samples_leaf=5,early_stopping=False,random_state=7)
        with threadpool_limits(limits=2):
            full=HistGradientBoostingRegressor(max_iter=20,**args).fit(x,y)
            short=HistGradientBoostingRegressor(max_iter=5,**args).fit(x,y)
            selected=training.tree_prediction(full,x,5)
            np.testing.assert_allclose(selected,short.predict(x),atol=1e-12,rtol=0)
            self.assertGreater(float(np.abs(selected-full.predict(x)).mean()),.01)

    def test_single_etf_adapter_has_independently_reconciling_risk_account(self):
        data=fixture(); data['manifest']={'root':'unused'}; n=len(data['panel'].dates)
        prices=pd.DataFrame({'QQQ':np.linspace(100,130,n)})
        volumes=pd.DataFrame({'QQQ':np.full(n,1e9)})
        with patch('report_ndx_broad.pd.read_csv',side_effect=lambda path,**kwargs:prices if Path(path).name=='etf_adj_close.csv' else volumes):
            adapted=qqq_adapter(data)
        # The production control has a twenty-session ADV warmup, as in real data.
        adapted['adv']=np.full_like(adapted['adv'],1e9)
        exposure=np.array([1.,.5,1.,.5,1.]); ids=np.arange(5); scores=np.ones((5,1))
        _,daily,_=policy_backtest(adapted,ids,scores,1,2.,exposure=exposure)
        replay=policy_replay(adapted,ids,scores,1,2.,exposure=exposure)
        np.testing.assert_allclose(replay,1e6*np.cumprod(1+daily.net),atol=1e-6,rtol=1e-12)


if __name__=='__main__':
    unittest.main()
