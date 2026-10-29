import unittest
import torch
from rishi_historical_objectives import pairwise, portfolio
class ObjectiveTests(unittest.TestCase):
 def test_ranking_ties_masks_and_grad(self):
  s=torch.tensor([[0.,1.,10.]],requires_grad=True);y=torch.tensor([[0.,1.,100.]]);mask=torch.tensor([[1,1,0]],dtype=torch.bool);a=pairwise(s,y,mask);self.assertLess(float(a),float(pairwise(-s,y,mask)));a.backward();self.assertEqual(float(s.grad[0,2]),0.)
  self.assertEqual(float(pairwise(s,torch.ones_like(y),mask)),0.)
 def test_cost_turnover_and_mask(self):
  s=torch.tensor([[1.,0.,10.],[0.,1.,10.]],requires_grad=True);y=torch.zeros_like(s);mask=torch.tensor([[1,1,0],[1,1,0]],dtype=torch.bool);a=portfolio(s,y,mask,0.);b=portfolio(s,y,mask,.02);self.assertGreater(float(b),float(a));b.backward();self.assertEqual(float(s.grad[:,2].abs().sum()),0.);self.assertTrue(torch.isfinite(s.grad).all())
 def test_future_mutation_does_not_change_training(self):
  torch.manual_seed(4);x=torch.randn(12,4,2);y=torch.randn(12,4)*.01;mask=torch.ones((12,4),dtype=torch.bool)
  def fit(features,labels,mode):
   w=torch.nn.Parameter(torch.zeros(2));opt=torch.optim.Adam([w],lr=.03)
   for _ in range(5):
    scores=features[:8]@w; loss=pairwise(scores,labels[:8],mask[:8]) if mode=='pairwise' else portfolio(scores,labels[:8],mask[:8]);opt.zero_grad();loss.backward();opt.step()
   return w.detach().clone()
  altered_x=x.clone();altered_y=y.clone();altered_x[8:]*=100;altered_y[8:]*=-100
  for mode in ('pairwise','portfolio'):torch.testing.assert_close(fit(x,y,mode),fit(altered_x,altered_y,mode),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
