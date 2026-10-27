import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from rishi_mechanism_batch1 import (
    read_leader_paths, validate_path, account_metrics, stationary_indices,
    paired_contrast, initial_capital_pool, holm, reserve_output,
)


class MechanismBatchTests(unittest.TestCase):
    def workbook(self, folder, formula=False):
        path = Path(folder)/'fixture.xlsx'
        root = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
        rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
        cell = '<c r="C7"><f>1/100</f><v>.01</v></c>' if formula else '<c r="C7"><v>.01</v></c>'
        with ZipFile(path, 'w') as z:
            z.writestr('xl/workbook.xml', f'<workbook xmlns="{root}" xmlns:r="{rel}"><sheets><sheet name="Leader paths" sheetId="1" r:id="r1"/></sheets></workbook>')
            z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
            z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{root}"><sheetData>'
                '<row r="5"><c r="C5" t="inlineStr"><is><t>equal5/full/ten_seed</t></is></c></row>'
                '<row r="6"><c r="A6" t="d"><v>2021-12-31T00:00:00Z</v></c><c r="C6"><v>0</v></c></row>'
                '<row r="7"><c r="A7" t="d"><v>2022-01-03T00:00:00Z</v></c>'+cell+'</row>'
                '<row r="8"><c r="A8" t="d"><v>2022-01-04T00:00:00Z</v></c><c r="C8"><v>-.02</v></c></row>'
                '</sheetData></worksheet>')
        return path

    def test_workbook_reads_literals_and_excludes_initial_zero(self):
        with tempfile.TemporaryDirectory(dir='.cache/test_tmp') as folder:
            frame = read_leader_paths(self.workbook(folder))
        self.assertEqual(len(frame), 2)
        self.assertEqual(list(frame.date.dt.strftime('%Y-%m-%d')), ['2022-01-03','2022-01-04'])
        np.testing.assert_allclose(frame.net, [.01,-.02])

    def test_workbook_rejects_formula_even_with_numeric_cache(self):
        with tempfile.TemporaryDirectory(dir='.cache/test_tmp') as folder:
            with self.assertRaisesRegex(ValueError, 'literal'):
                read_leader_paths(self.workbook(folder, True))

    def test_validate_rejects_missing_duplicate_and_out_of_order_dates(self):
        expected = pd.date_range('2022-01-03', periods=3, freq='B')
        for dates in [expected[:2], expected[[0,0,2]], expected[::-1]]:
            with self.subTest(dates=dates):
                with self.assertRaises(ValueError):
                    validate_path(dates, np.zeros(len(dates)), expected)

    def test_validate_rejects_nonfinite_or_bankrupt_returns(self):
        dates = pd.date_range('2022-01-03', periods=2)
        for values in [[0., np.nan], [0., -1.], [np.inf, 0.]]:
            with self.assertRaises(ValueError):
                validate_path(dates, values, dates)

    def test_metrics_include_first_fee_and_initial_drawdown(self):
        r = np.array([-.01,.02,-.03])
        m = account_metrics(r)
        self.assertAlmostEqual(m['terminal_wealth'], .99*1.02*.97)
        self.assertAlmostEqual(m['maximum_drawdown'], -.03)
        self.assertAlmostEqual(account_metrics([-.01,-.02])['maximum_drawdown'], .99*.98-1)

    def test_bootstrap_identical_arms_have_exact_zero_difference(self):
        r = np.tile([.01,-.008,.002], 12)
        ids = stationary_indices(len(r), 150, 5, 42)
        result = paired_contrast(r, r, ids, 252)
        for key in ['mean_difference','mean_low','mean_high','cagr_difference','sr_difference']:
            self.assertEqual(result[key], 0.)
        self.assertEqual(result['mean_p_one_sided'], 1.)

    def test_stationary_blocks_are_common_reproducible_and_wrap(self):
        ids = stationary_indices(7, 20, 1000, 13)
        np.testing.assert_array_equal(ids, stationary_indices(7, 20, 1000, 13))
        self.assertEqual(ids.shape, (20,7))
        self.assertTrue(((ids>=0)&(ids<7)).all())
        self.assertGreater(np.mean((np.diff(ids, axis=1)%7)==1), .9)
        left = np.arange(7, dtype=float)
        np.testing.assert_array_equal((left+1)[ids]-left[ids], np.ones_like(ids))

    def test_initial_capital_pool_is_not_periodic_rebalancing(self):
        # Opposing +100% / -50% components each finish flat. Initial split is flat;
        # resetting to 50:50 every period would compound 25% twice.
        pooled = initial_capital_pool([1.,-.5], [-.5,1.])
        np.testing.assert_allclose(pooled, [.25,-.2])
        self.assertAlmostEqual(np.prod(1+pooled), 1.)

    def test_holm_respects_family_order_and_monotonicity(self):
        np.testing.assert_allclose(holm([.04,.001,.03]), [.06,.003,.06])

    def test_completed_or_partial_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir='.cache/test_tmp') as folder:
            target = Path(folder)/'batch'
            reserve_output(target)
            sentinel = target/'partial.csv'
            sentinel.write_text('original evidence')
            with self.assertRaises(FileExistsError):
                reserve_output(target)
            self.assertEqual(sentinel.read_text(),'original evidence')


if __name__ == '__main__':
    unittest.main()
