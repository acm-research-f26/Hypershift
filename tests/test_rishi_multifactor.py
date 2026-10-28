import tempfile
import unittest
from pathlib import Path
import numpy as np
from rishi_multifactor import read_factors
from rishi_factor_nw import fit_hac


class MultiFactorTests(unittest.TestCase):
    def test_factor_percent_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'factor.csv'
            path.write_text('header\n,date\n20220103, 1.50\n20220104,-0.50\n')
            result = read_factors(path,['MOM'])
            np.testing.assert_allclose(result.MOM.to_numpy(),[.015,-.005])
            path.write_text('20220103, 1.50\n20220103, 2.00\n')
            with self.assertRaises(ValueError):
                read_factors(path,['MOM'])

    def test_multivariate_coefficients(self):
        rng = np.random.default_rng(123)
        x = np.column_stack([np.ones(100), rng.normal(size=(100,7))])
        beta = np.arange(8)/10
        result = fit_hac(x@beta,x,5)
        np.testing.assert_allclose(result['beta'],beta,atol=1e-12)
        self.assertEqual(result['n'],100)
