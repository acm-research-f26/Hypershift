import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ndx_corrections import proxy
from report_ndx_corrections import independent_proxy
from run_ndx_corrections import validation_stage,CELLS
from run_ndx_alpaca_training import ROOT


class EvaluationTests(unittest.TestCase):
    def test_independent_proxy_agrees_with_boundary_ties_missing_and_turnover(self):
        rng=np.random.default_rng(7);a=rng.integers(0,5,(30,16)).astype(float);b=rng.integers(0,5,(30,16)).astype(float)
        mask=rng.random((30,16))>.1;observed=mask.copy();observed[4,5]=False
        y=rng.normal(size=(30,16))
        expected=proxy(a,b,mask,observed,y);actual=independent_proxy(a,b,mask,observed,y)
        for key in expected:np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=0)

    def test_h2_can_change_acceptance_but_never_h1_choice(self):
        val=np.arange(48);h1=val[:24];h2=val[24:];mask=np.ones((48,14),bool)
        core=[np.zeros((48,14)) for _ in range(2)]
        adapter=[np.tile(np.arange(14),(48,1)).astype(float) for _ in range(2)]
        adapters={(cell,5):adapter for cell in CELLS}
        base=dict(mask=mask,labelmask=mask.copy(),y=np.tile(np.arange(14)/100,(48,1)))
        plan=dict(test_years=[2022],adapter=dict(checkpoint_epochs=[5]),shrink_factors=[0,1])
        temporary=ROOT/'tmp';temporary.mkdir(exist_ok=True);folder=Path(tempfile.mkdtemp(prefix='correction_gate_',dir=temporary))
        try:
            with patch('run_ndx_corrections.RUN',folder),patch('run_ndx_corrections.partitions',return_value=(np.array([0]),val,np.array([49]),h1,h2)),patch('run_ndx_corrections.validation_predictions',return_value=(core,adapters)):
                choices,gates=validation_stage(dict(base=base),plan)
                self.assertTrue(gates[0]['results']['primary']['accepted'])
                lock=(folder/'H1_SELECTION_LOCK.json').read_bytes()
                base['y'][h2]*=-100
                new_choices,new_gates=validation_stage(dict(base=base),plan,write_files=False)
                self.assertEqual(choices,new_choices)
                self.assertFalse(new_gates[0]['results']['primary']['accepted'])
                self.assertEqual(lock,(folder/'H1_SELECTION_LOCK.json').read_bytes())
                self.assertIsNone(new_gates[0]['results']['primary']['deployed_candidate'])
        finally:
            if folder.resolve().is_relative_to(temporary.resolve()):shutil.rmtree(folder)

    def test_independent_proxy_zero_is_zero_under_extreme_return_inputs(self):
        rng=np.random.default_rng(17);score=rng.normal(size=(30,14));mask=np.ones_like(score,dtype=bool)
        result=independent_proxy(score,score,mask,mask,rng.normal(size=score.shape)*1e6)
        self.assertEqual(result['score'],0.);self.assertEqual(result['wins'],0)


if __name__=='__main__':unittest.main()
