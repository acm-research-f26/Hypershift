import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ndx_matched_controls import momentum_scores, momentum_weights, paired_inference, etf_data


class MatchedControlTests(unittest.TestCase):
    def panel(self):
        days = pd.bdate_range('2020-01-01', periods=330)
        t = np.arange(330)[:, None]
        marks = 10*np.exp(t*np.array([.001, .001, .002, .003, .004, .005, .006]))
        return dict(panel=SimpleNamespace(dates=days, tickers=['GOOG', 'GOOGL', 'A', 'B', 'C', 'D', 'E']),
                    marks=marks, origin=np.array([270, 280, 290]),
                    mask=np.ones((3, 7), dtype=bool), tradable=np.ones_like(marks, dtype=bool))

    def test_momentum_uses_only_lagged_endpoints_and_preserves_neural_mask(self):
        data = self.panel(); before = momentum_scores(data, [0, 1, 2])
        expected = data['marks'][249]/data['marks'][18]-1
        np.testing.assert_allclose(before[0], expected)
        data['marks'][250:] *= 100
        np.testing.assert_array_equal(momentum_scores(data, [0])[0], before[0])
        data['tradable'][18, 2] = False
        missing = momentum_scores(data, [0]); self.assertTrue(np.isnan(missing[0, 2]))
        _, support = momentum_weights(data, [0])
        self.assertEqual(support.missing_history.iloc[0], 1)
        self.assertTrue(data['mask'].all())

    def test_issuer_momentum_cannot_double_weight_alphabet(self):
        data = self.panel(); data['marks'][:, :2] = 10*np.exp(np.arange(330)[:, None]*.02)
        weights, _ = momentum_weights(data, [0, 1, 2])
        np.testing.assert_allclose(weights.sum(1), 1)
        np.testing.assert_allclose(weights[:, :2].sum(1), .2)

    def test_etf_calendar_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); dates = pd.bdate_range('2022-01-01', periods=30)
            pd.DataFrame({'QQQ': np.arange(30)+100}, index=dates).to_csv(p/'etf_adj_close.csv')
            pd.DataFrame({'QQQ': np.full(30, 1e9)}, index=dates).to_csv(p/'etf_dollar_volume.csv')
            data = dict(panel=SimpleNamespace(dates=dates[1:]), manifest={'root': directory})
            with self.assertRaisesRegex(ValueError, 'calendars'):
                etf_data(data, 'QQQ')

    def test_joint_inference_changes_sign_when_contrast_is_reversed(self):
        rng = np.random.default_rng(7)
        values = rng.normal([.001, .0002], [.015, .01], (400, 2))
        forward = paired_inference(values, ['A', 'B'], [('A', 'B')], 20, 300)
        reverse = paired_inference(values, ['A', 'B'], [('B', 'A')], 20, 300)
        for lo, hi in [('paired_sr_low', 'paired_sr_high'), ('paired_cagr_low', 'paired_cagr_high'),
                       ('simultaneous_active_low', 'simultaneous_active_high')]:
            self.assertAlmostEqual(forward[lo].iloc[0], -reverse[hi].iloc[0], places=12)


if __name__ == '__main__':
    unittest.main()
