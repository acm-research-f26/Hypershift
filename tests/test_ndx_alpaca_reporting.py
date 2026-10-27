"""Paired inference must preserve identical-path and constant-difference cases."""
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import report_ndx_alpaca as reporting


class PairedInferenceTests(unittest.TestCase):
    def test_exposure_recovers_known_beta_and_alpha(self):
        rng = np.random.default_rng(19)
        control = rng.normal(.0004,.01,200)
        hurdle = np.expm1(np.log1p(.03)/252)
        returns = hurdle+.0002+1.2*(control-hurdle)
        base = Path(__file__).resolve().parent/'tmp'
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            if not Path(folder).resolve().is_relative_to(base.resolve()):
                raise RuntimeError('Test output outside workspace')
            with patch.object(reporting,'RUN',Path(folder)):
                reporting.exposure_diagnostics({('intraday_augmented','think'):returns,
                    ('control','QQQ'):control,('control','SPY'):control*.8})
            frame = pd.read_csv(Path(folder)/'exposure_diagnostics.csv')
            row = frame[frame.benchmark=='QQQ'].iloc[0]
            self.assertAlmostEqual(row.beta,1.2)
            self.assertAlmostEqual(row.alpha_annual,.0002*252)
            self.assertAlmostEqual(row.tracking_error_annual,.2*np.std(control,ddof=1)*np.sqrt(252))

    def test_identical_inputs_have_zero_paired_differences(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(.0003,.01,150)
        primary = {('intraday_augmented','think'):returns,
                   ('daily_control','think'):returns.copy(),
                   ('control','SPY'):returns*.8,('control','QQQ'):returns*.9}
        base = Path(__file__).resolve().parent/'tmp'
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            if not Path(folder).resolve().is_relative_to(base.resolve()):
                raise RuntimeError('Test output outside workspace')
            with patch.object(reporting,'RUN',Path(folder)):
                frame = reporting.paired_intervals(primary,dict(models=['think']),repeats=200)
            row = frame[(frame.comparison=='intraday_minus_daily')&(frame.block_sessions==20)].iloc[0]
            self.assertEqual(row.active_mean_annual,0.)
            self.assertEqual(row.active_low,0.)
            self.assertEqual(row.active_high,0.)
            self.assertEqual(row.sharpe_difference,0.)
            self.assertLessEqual(row.simultaneous_low,0.)
            self.assertGreaterEqual(row.simultaneous_high,0.)
            self.assertEqual(row.family_comparisons,3)
            self.assertEqual(set(frame.block_sessions),{5,20,60})


if __name__=='__main__':
    unittest.main()
