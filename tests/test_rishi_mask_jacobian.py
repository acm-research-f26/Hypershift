import unittest
from unittest.mock import patch
import numpy as np
import torch
from rishi_mask_jacobian import individual_jacobian


class JacobianTests(unittest.TestCase):
    def test_individual_derivatives_do_not_cancel(self):
        def fake_forecast(model, x, eligible, arm):
            return torch.stack([x[0, 0, 0], -x[0, 0, 0], x[2, 0, 0] * 0])
        raw = np.zeros((3, 1, 1), dtype='float32')
        mask = np.array([True, True, False])
        with patch('rishi_mask_jacobian.forecast', side_effect=fake_forecast):
            scores, jac = individual_jacobian(None, raw, mask, 'full')
        self.assertEqual(jac.shape, (2, 3, 1, 1))
        self.assertEqual(jac[0, 0, 0, 0], 1)
        self.assertEqual(jac[1, 0, 0, 0], -1)
        self.assertTrue((jac[:, 2] == 0).all())
        self.assertEqual(len(scores), 3)
