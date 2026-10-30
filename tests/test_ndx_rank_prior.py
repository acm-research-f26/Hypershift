import unittest
import numpy as np
from ndx_rank_prior import training_rank_prior


class RankPriorTests(unittest.TestCase):
    def test_full_history_matches_existing_neutral_prior(self):
        data=dict(y=np.array([[1.,2.,0.],[1.,3.,0.]]),labelmask=np.array([[1,1,0],[1,1,0]],bool),origin=np.array([0,5]))
        score,count=training_rank_prior(data,np.array([0,1]),0)
        np.testing.assert_allclose(score,[(1+10)/22,(2+10)/22,.5])
        np.testing.assert_array_equal(count,[2,2,0])

    def test_only_training_labels_enter_prior_and_decay_is_chronological(self):
        data=dict(y=np.array([[1.,2.],[3.,1.],[100.,-100.]]),labelmask=np.ones((3,2),bool),origin=np.array([0,5,10]))
        a,count=training_rank_prior(data,np.array([0,1]),1)
        data['y'][2]*=-1;b,_=training_rank_prior(data,np.array([0,1]),1)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(count,[1.5,1.5])
        self.assertGreater(a[0],a[1])


if __name__=='__main__':unittest.main()
