import unittest
import numpy as np
from rishi_rank_behavior import rank_anova,adjacent_behavior,selection_categories


class RankBehaviorTests(unittest.TestCase):
    def test_unbalanced_eligibility_weighted_anova_identity(self):
        score=np.array([[0.,2.],[2.,999.],[4.,4.]])
        mask=np.array([[1,1],[1,0],[1,1]],bool)
        x=rank_anova(score,mask)
        self.assertAlmostEqual(x['total_ss'],11.2)
        self.assertAlmostEqual(x['between_id_ss'],1.2)
        self.assertAlmostEqual(x['within_id_ss'],10.)
        self.assertAlmostEqual(x['between_id_fraction'],1.2/11.2)

    def test_constant_scores_do_not_invent_variation(self):
        x=rank_anova(np.ones((3,2)),np.ones((3,2),bool))
        self.assertEqual(x['total_ss'],0)
        self.assertIsNone(x['between_id_fraction'])

    def test_adjacent_common_ids_and_fold_boundaries(self):
        s=np.array([[1.,2.,3.],[3.,2.,1.],[1.,2.,3.]])
        m=np.array([[1,1,1],[1,1,0],[1,1,1]],bool)
        rows=adjacent_behavior(s,m,np.array([2022,2022,2023]),1)
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0]['rank_correlation'],-1)
        self.assertEqual(rows[0]['membership_exits'],1)

    def test_fractional_top_k_ties_and_exclusive_categories(self):
        s=np.ones((2,3));m=np.ones((2,3),bool)
        row=adjacent_behavior(s,m,np.array([1,1]),1)[0]
        self.assertAlmostEqual(row['selected_weight_overlap'],1)
        c=selection_categories(np.array([.5,.5,0,0]),np.array([.5,0,.5,0]))
        np.testing.assert_array_equal(c,np.arange(4))


if __name__=='__main__':unittest.main()
