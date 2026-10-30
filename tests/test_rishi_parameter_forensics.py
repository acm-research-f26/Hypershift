import unittest
import numpy as np
from rishi_parameter_forensics import delta_stats,id_categories,operator_stats,indexed_vector_update


class ParameterForensicsTests(unittest.TestCase):
    def test_vector_rotation_is_not_mistaken_for_no_update(self):
        initial=np.array([[1.0],[0.0]])
        trained=np.array([[0.0],[1.0]])
        self.assertAlmostEqual(indexed_vector_update(trained,initial,axis=0)[0],np.sqrt(2))

    def test_zero_initialization_keeps_relative_change_undefined(self):
        result=delta_stats(np.ones(4),np.zeros(4))
        self.assertEqual(result['initial_l2'],0)
        self.assertEqual(result['update_l2'],2)
        self.assertEqual(result['update_rms'],1)
        self.assertIsNone(result['relative_update'])

    def test_support_groups_are_exclusive_and_use_training_counts(self):
        result=id_categories(np.array([40,20,0,0,12]),np.array([35,19,0,0,0]),np.array([2,1,3,0,1]))
        self.assertEqual(result.tolist(),['supported','sparse','newly_eligible','unsupported','sparse'])

    def test_uniform_filter_does_not_cancel_noninverse_operators(self):
        result=operator_stats(np.full((2,2),.5),np.eye(2)*2)
        self.assertAlmostEqual(result['inverse_defect_fro'],np.sqrt(2))
        self.assertAlmostEqual(result['uniform_filter_offdiag_fro'],np.sqrt(2))
        identity=operator_stats(np.eye(3),np.eye(3))
        self.assertEqual(identity['inverse_defect_fro'],0)
        self.assertEqual(identity['condition'],1)
        self.assertEqual(identity['inverse_rank'],3)
        self.assertEqual(identity['inverse_singular_min'],1)


if __name__=='__main__':unittest.main()
