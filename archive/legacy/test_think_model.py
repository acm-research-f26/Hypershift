import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from think_model import exp0,log0,beta_concat,PoincareFC,ThinkReconstruction,HypergraphAttention
from run_think_reconstruction import rank_loss,metrics,dcg,load_nyse


class ThinkTests(unittest.TestCase):
    def test_exp_log_and_concat(self):
        x=torch.randn(2,3,4,dtype=torch.float64)*.1
        torch.testing.assert_close(log0(exp0(x)),x)
        self.assertEqual(beta_concat(exp0(x)).shape,(2,12))
        self.assertTrue((beta_concat(exp0(x)).norm(dim=-1)<1).all())

    def test_fc_origin_bias_and_gradcheck(self):
        layer=PoincareFC(2,3).double()
        torch.testing.assert_close(layer(torch.zeros(1,2,dtype=torch.float64)),torch.zeros(1,3,dtype=torch.float64))
        x=torch.tensor([[.12,-.08]],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(layer,(x,),atol=1e-5))

    def test_all_variants_backward_and_checkpoint(self):
        for variant in ['think','euclidean_temporal','no_distance','euclidean']:
            torch.manual_seed(7)
            model=ThinkReconstruction(2,4,[{0,1},{1,2},{2,3}],4,variant=variant)
            x=torch.randn(2,4,4,2)*.1
            out=model(x)
            self.assertEqual(out.shape,(2,4))
            out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            other=ThinkReconstruction(2,4,[{0,1},{1,2},{2,3}],4,variant=variant)
            other.load_state_dict(model.state_dict())
            torch.testing.assert_close(out,other(x))

    def test_node_relabeling_equivariance(self):
        torch.manual_seed(1)
        groups=[{0,1,2},{2,3}]; order=torch.tensor([2,0,3,1]); inv=torch.argsort(order)
        changed=[{int(inv[j]) for j in g} for g in groups]
        for variant in ['think','euclidean_temporal','no_distance','euclidean']:
            a=ThinkReconstruction(2,4,groups,4,variant=variant)
            b=ThinkReconstruction(2,4,changed,4,variant=variant)
            state={k:v for k,v in a.state_dict().items() if k not in ['spatial.node_ids','spatial.edge_ids','spatial.edge_degree']}
            b.load_state_dict(state,strict=False)
            x=torch.randn(2,4,4,2)*.1
            torch.testing.assert_close(b(x[:,:,order]),a(x)[:,order],atol=1e-6,rtol=1e-5)

    def test_chunked_loss_matches_dense(self):
        p=torch.randn(2,7,requires_grad=True); y=torch.randn(2,7); m=torch.randint(0,2,(2,7)).float()
        reference=((p-y).square()*m).mean()+torch.relu((p[:,:,None]-p[:,None,:])*(y[:,None,:]-y[:,:,None])).mul(m[:,:,None]*m[:,None,:]).mean()
        actual=rank_loss(p,y,m,chunk=2)
        torch.testing.assert_close(actual,reference)
        torch.testing.assert_close(torch.autograd.grad(actual,p,retain_graph=True)[0],torch.autograd.grad(reference,p)[0])

    def test_ndcg_identity_and_relabeling(self):
        y=np.array([[.03,.01,-.02,.04,.02,.005]])
        p=np.array([[.1,.3,.2,.6,.5,.4]]); m=np.ones_like(p)
        original=metrics(p,y,m)['ndcg_at_5_positive_return']
        order=[4,0,5,2,1,3]
        self.assertAlmostEqual(original,metrics(p[:,order],y[:,order],m[:,order])['ndcg_at_5_positive_return'])
        self.assertAlmostEqual(metrics(y,y,m)['ndcg_at_5_positive_return'],1.)
        gains=np.array([1.,2.,3.])
        self.assertAlmostEqual(dcg(gains,np.zeros(3),2),2*(1+1/np.log2(3)))

    def test_training_boundary_example(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'NYSE').mkdir()
            (root/'NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv').write_text('A\nB\n')
            relation=root/'relations.json'; relation.write_text(json.dumps([['A','B']]))
            raw=np.column_stack([np.arange(1245),np.ones((1245,5))])
            raw[500,-1]=-1234
            for ticker in ['A','B']:
                np.savetxt(root/'NYSE'/f'NYSE_{ticker}_1.csv',raw,delimiter=',')
            first=load_nyse(root,relation)
            # Alter only validation/test observations. Earlier windows/labels must stay unchanged.
            raw[756:,1:]*=17
            for ticker in ['A','B']:
                np.savetxt(root/'NYSE'/f'NYSE_{ticker}_1.csv',raw,delimiter=',')
            second=load_nyse(root,relation)
            tr=first['splits']['train']; va=first['splits']['validation']
            self.assertLess(first['targets'][tr].max(),first['targets'][va].min())
            for key in ['x','y','mask','base']:
                torch.testing.assert_close(first[key][tr],second[key][tr])
            missing_windows=np.isin(first['targets'],np.arange(500,505))
            self.assertTrue((first['mask'][missing_windows]==0).all())


if __name__=='__main__': unittest.main()
