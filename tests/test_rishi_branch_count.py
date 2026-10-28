import unittest
import numpy as np
import torch
from rishi_branch_count import components
from rishi_member_count import toy_input
from rishi_synthetic_gate import make_model


class BranchCountTests(unittest.TestCase):
    def test_adapter_and_dynamic_reconstruction(self):
        torch.set_num_threads(2)
        model=make_model(7).eval()
        raw,mask=toy_input(0,4,'replicated_peer','zero')
        with torch.no_grad():
            result=components(model,torch.tensor(raw),torch.tensor(mask))
            actual=model(torch.tensor(raw),None,torch.tensor(mask))[0]
        self.assertTrue(torch.equal(result['score'],actual))
        self.assertTrue(torch.equal(result['dynamic'],result['reconstructed']))
        np.testing.assert_allclose(result['incidence'].sum(-1).numpy(),np.ones(8),rtol=1e-6)
