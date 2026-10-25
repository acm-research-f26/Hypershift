"""Objective-scale and optimizer checks for the controlled training extension."""
import unittest
import numpy as np
import torch
from ndx_loss_optimizer import objective,optimizer_for,method_spec,candidate_specs


class ObjectiveTests(unittest.TestCase):
    def test_mse_parent_arithmetic(self):
        p=torch.tensor([[.2,-.3,5.]],requires_grad=True);y=torch.tensor([[.1,.2,0.]]);m=torch.tensor([[1.,1.,0.]])
        torch.testing.assert_close(objective(p,y,m,'mse'),(((p-y)**2)*m).sum()/m.sum(),rtol=0,atol=0)

    def test_robust_gradients_and_masks(self):
        for loss in ['huber','pseudohuber']:
            p=torch.tensor([[0.,100.,float('nan')]],dtype=torch.float64,requires_grad=True)
            y=torch.tensor([[0.,0.,float('nan')]],dtype=torch.float64);m=torch.tensor([[1.,1.,0.]],dtype=torch.float64)
            value=objective(p,y,m,loss);value.backward()
            self.assertTrue(torch.isfinite(value));self.assertEqual(p.grad[0,2].item(),0.)
            self.assertLessEqual(abs(p.grad[0,1].item()),1.000001)

    def test_small_error_curvature_and_zero(self):
        p=torch.tensor([[0.,1e-4]],dtype=torch.float64);y=torch.zeros_like(p);m=torch.ones_like(p)
        mse=objective(p,y,m,'mse')
        for loss in ['huber','pseudohuber']:
            torch.testing.assert_close(objective(p,y,m,loss),mse,rtol=1e-7,atol=1e-15)

    def test_scalar_robust_formulas(self):
        errors=np.array([-.01,.5,1.,5.]);p=torch.tensor(errors);y=torch.zeros_like(p);m=torch.ones_like(p)
        huber=np.where(abs(errors)<1,errors**2,2*abs(errors)-1).mean()
        pseudo=(2*(np.sqrt(1+errors**2)-1)).mean()
        self.assertAlmostEqual(objective(p,y,m,'huber').item(),huber,12)
        self.assertAlmostEqual(objective(p,y,m,'pseudohuber').item(),pseudo,12)

    def test_optimizer_parameters(self):
        for name in ['adamw','radam','sgd']:
            model=torch.nn.Linear(2,1);opt=optimizer_for(model,name,.003)
            self.assertEqual(opt.defaults['lr'],.003);self.assertEqual(opt.defaults['weight_decay'],.001)
            if name=='radam':self.assertTrue(opt.defaults['decoupled_weight_decay'])
            if name=='sgd':self.assertEqual(opt.defaults['momentum'],.9)
            old=model.weight.detach().clone();model(torch.ones(1,2)).square().sum().backward();opt.step()
            self.assertFalse(torch.equal(old,model.weight.detach()))

    def test_method_specs_and_inventory(self):
        self.assertEqual(method_spec('fixed_adamw_pseudohuber',2025,{}),('pseudohuber','adamw',.001))
        self.assertEqual(method_spec('tuned_radam_mse',2025,{'2025':{'radam':{'lr':.003}}}),('mse','radam',.003))


if __name__=='__main__':unittest.main()
