import unittest
import numpy as np
import torch
from rishi_synthetic_gate import make_data, score_recovery


class SyntheticGateTests(unittest.TestCase):
    def test_planted_own_and_peer_targets_use_correct_stocks(self):
        data = make_data()
        x = data['train_x']
        np.testing.assert_allclose(data['train_y']['own'], x[:,:,0,0]/.2)
        np.testing.assert_allclose(data['train_y']['peer'], (x[:,:,0,0]/.2)[:,np.arange(8)^1])
        self.assertEqual(x.shape,(128,8,16,25))
        self.assertEqual(data['test_x'].shape,(64,8,16,25))
        np.testing.assert_array_equal(x[:,:,0,0],x[:,:,-1,0])

    def test_identity_has_no_input_tag_and_train_test_are_independent(self):
        data = make_data()
        self.assertFalse(torch.equal(data['train_x'][:64],data['test_x']))
        np.testing.assert_allclose(data['train_y']['identity'][0],data['test_y']['identity'][-1])
        self.assertAlmostEqual(float(data['train_y']['identity'].mean()),0.,places=6)
        other = make_data()
        self.assertTrue(torch.equal(data['train_x'],other['train_x']))

    def test_recovery_is_relative_to_zero_predictor(self):
        y = torch.tensor([[1.,-1.],[2.,-2.]])
        self.assertEqual(score_recovery(y,y)['r2_zero'],1.)
        self.assertEqual(score_recovery(torch.zeros_like(y),y)['r2_zero'],0.)


if __name__=='__main__':
    unittest.main()
