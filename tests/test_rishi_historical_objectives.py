import unittest
import torch
from rishi_historical_objectives import pairwise, portfolio
class ObjectiveTests(unittest.TestCase):
 def test_ranking_ties_masks_and_grad(self):
  s=torch.tensor([[0.,1.,10.]],requires_grad=True);y=torch.tensor([[0.,1.,100.]]);mask=torch.tensor([[1,1,0]],dtype=torch.bool);a=pairwise(s,y,mask);self.assertLess(float(a),float(pairwise(-s,y,mask)));a.backward();self.assertEqual(float(s.grad[0,2]),0.)
  self.assertEqual(float(pairwise(s,torch.ones_like(y),mask)),0.)
 def test_cost_turnover_and_mask(self):
  s=torch.tensor([[1.,0.,10.],[0.,1.,10.]],requires_grad=True);y=torch.zeros_like(s);mask=torch.tensor([[1,1,0],[1,1,0]],dtype=torch.bool);a=portfolio(s,y,mask,0.);b=portfolio(s,y,mask,.02);self.assertGreater(float(b),float(a));b.backward();self.assertEqual(float(s.grad[:,2].abs().sum()),0.);self.assertTrue(torch.isfinite(s.grad).all())
if __name__=='__main__':unittest.main()
