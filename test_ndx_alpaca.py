"""Causality and matched-capacity checks for the separate intraday experiment."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import ndx_alpaca_models as models
from prepare_ndx_alpaca import calendar_frame, summarize, FEATURES
import prepare_ndx_alpaca as preparation


@contextmanager
def temporary_directory():
    base = Path(__file__).resolve().parent/'tmp'
    base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as folder:
        if not Path(folder).resolve().is_relative_to(base.resolve()):
            raise RuntimeError('Test output outside workspace')
        yield Path(folder)


class IntradayModelTests(unittest.TestCase):
    def test_missing_intraday_values_impute_only_training_statistics(self):
        raw = np.full((4, 16, 3, 13), np.nan, dtype='float32')
        raw[:2, ..., :5] = 1.
        raw[:2, ..., 5] = 2.
        raw[..., -1] = 0.
        mask = np.ones((4, 3), bool)
        x, mean, std = models.scale(raw, mask, np.arange(2))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[2:], 0.)
        changed = raw.copy()
        changed[2:] = 1e10
        _, other_mean, other_std = models.scale(changed, mask, np.arange(2))
        np.testing.assert_array_equal(mean, other_mean)
        np.testing.assert_array_equal(std, other_std)

    def test_inactive_stocks_do_not_change_scaling(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(4, 16, 3, 13)).astype('float32')
        mask = np.ones((4, 3), bool)
        mask[:, 2] = False
        _, mean, std = models.scale(raw, mask, np.arange(2))
        raw[:, :, 2] = 1e10
        _, other_mean, other_std = models.scale(raw, mask, np.arange(2))
        np.testing.assert_array_equal(mean, other_mean)
        np.testing.assert_array_equal(std, other_std)

    def test_views_have_equal_dimensions_and_only_use_signal_history(self):
        data = dict(x=np.ones((2, 16, 3, 5), dtype='float32'), origin=np.array([15, 20]))
        daily = models.inputs(data, 'daily_control')
        features = np.broadcast_to(np.arange(25)[:, None, None], (25, 3, 8)).copy()
        augmented = models.inputs(data, 'intraday_augmented', features)
        self.assertEqual(daily.shape, augmented.shape)
        np.testing.assert_array_equal(augmented[0, ..., -1], features[:16, ..., -1])
        features[21:] = 1e10
        np.testing.assert_array_equal(augmented, models.inputs(data, 'intraday_augmented', features))
        np.testing.assert_array_equal(daily[..., :5], augmented[..., :5])

    def test_all_variants_mask_inactive_inputs_and_have_matched_initialization(self):
        torch.set_num_threads(2)
        x = torch.randn(2, 16, 3, 13)*.1
        mask = torch.tensor([[1., 1., 0.], [1., 0., 1.]])
        changed = torch.where(mask[:, None, :, None].bool(), x, torch.full_like(x, float('nan')))
        kinds = ['think', 'think_mix', 'think_huber', 'think_mix_huber', 'think_mix_pseudohuber', 'mlp', 'stockmixer']
        for kind in kinds:
            with self.subTest(kind=kind):
                torch.manual_seed(7)
                a = models.make(kind, 3, [{0, 1}, {1, 2}])
                torch.manual_seed(7)
                b = models.make(kind, 3, [{0, 1}, {1, 2}])
                for key in a.state_dict():
                    torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
                torch.testing.assert_close(a(x, mask)[0], a(changed, mask)[0], rtol=0, atol=0)

    def test_selected_validation_is_recomputed_after_checkpoint_restore(self):
        data = dict(panel=SimpleNamespace(tickers=['A', 'B', 'C']),
                    mask=np.ones((4, 3), bool), labelmask=np.ones((4, 3), bool),
                    y=np.zeros((4, 3), dtype='float32'))
        x = np.zeros((4, 16, 3, 13), dtype='float32')
        plan = dict(lr=.001, weight_decay=.001, epochs=2, batch=2, gradient_norm_cap=1., patience=6)
        val_best = np.zeros((1, 3), dtype='float32')
        val_last = np.ones((1, 3), dtype='float32')
        test_pred = np.full((1, 3), .5, dtype='float32')
        with temporary_directory() as folder:
            with patch.object(models, 'configure_device', return_value=torch.device('cpu')):
                with patch.object(models, 'predict', side_effect=[val_best, val_last, val_best, test_pred]) as predict:
                    vp, tp, record = models.fit(data, x, np.array([0, 1]), np.array([2]), np.array([3]),
                        [{0, 1}, {1, 2}], 'mlp', 7, plan, folder)
            self.assertEqual(record['chosen_epoch'], 1)
            self.assertEqual(predict.call_count, 4)
            np.testing.assert_array_equal(vp, val_best)
            np.testing.assert_array_equal(tp, test_pred)
            self.assertEqual(json.loads((folder/'mlp_7_history.json').read_text())[-1]['epoch'], 2)


class SessionFeatureTests(unittest.TestCase):
    def test_reused_meta_ticker_cannot_enter_facebook_history(self):
        rules = json.loads(preparation.IDENTITY.read_text())['symbols']
        frame = pd.DataFrame(dict(date=['2021-06-30','2022-06-08','2022-06-09','2025-12-31']))
        valid, rejected = preparation.filter_identity(frame,'META',rules)
        self.assertEqual(valid.date.tolist(),['2022-06-09','2025-12-31'])
        self.assertEqual(rejected.date.tolist(),['2021-06-30','2022-06-08'])
        valid, rejected = preparation.filter_identity(frame,'FB',rules)
        self.assertEqual(valid.date.tolist(),['2021-06-30','2022-06-08'])
        self.assertEqual(rejected.date.tolist(),['2022-06-09','2025-12-31'])

    def test_rename_boundary_assigns_exactly_one_valid_symbol(self):
        rules = json.loads(preparation.IDENTITY.read_text())['symbols']
        frame = pd.DataFrame(dict(date=['2019-11-04','2019-11-05']))
        old, _ = preparation.filter_identity(frame,'CTRP',rules)
        new, _ = preparation.filter_identity(frame,'TCOM',rules)
        self.assertEqual(old.date.tolist(),['2019-11-04'])
        self.assertEqual(new.date.tolist(),['2019-11-05'])

    def fixture(self, date='2024-11-29', close='13:00'):
        calendar = calendar_frame([dict(date=date, open='09:30', close=close)])
        start, end = calendar.open_utc.iloc[0], calendar.close_utc.iloc[0]
        times = pd.date_range(start, end, freq='15min', inclusive='left')
        frame = pd.DataFrame(dict(timestamp_utc=times.astype(str),
            bar_end_utc=(times+pd.Timedelta(minutes=15)).astype(str),
            open=100., high=102., low=99., close=101., volume=1000., vwap=100.5))
        return frame, calendar

    def test_early_close_uses_fourteen_complete_bars(self):
        frame, calendar = self.fixture()
        row = summarize(frame, calendar).iloc[0]
        self.assertTrue(row.complete)
        self.assertEqual(row.expected, 14)
        self.assertEqual(row.intraday_coverage, 1.)
        self.assertAlmostEqual(row.rth_return, .01)
        self.assertAlmostEqual(row.late_volume_share, 4/14)

    def test_full_session_uses_twenty_six_bars(self):
        frame, calendar = self.fixture('2024-11-27', '16:00')
        row = summarize(frame, calendar).iloc[0]
        self.assertTrue(row.complete)
        self.assertEqual(row.expected, 26)

    def test_vectorized_summaries_match_explicit_reference(self):
        frame, calendar = self.fixture()
        first = summarize(frame,calendar)
        reference = preparation.summarize_reference(frame,calendar)
        pd.testing.assert_frame_equal(first,reference,rtol=1e-12,atol=1e-12)
        pd.testing.assert_frame_equal(summarize(frame.iloc[1:],calendar),
            preparation.summarize_reference(frame.iloc[1:],calendar),rtol=1e-12,atol=1e-12)

    def test_incomplete_session_preserves_coverage_and_missing_numeric_features(self):
        frame, calendar = self.fixture()
        row = summarize(frame.iloc[1:], calendar).iloc[0]
        self.assertFalse(row.complete)
        self.assertAlmostEqual(row.intraday_coverage, 13/14)
        self.assertTrue(row[FEATURES[:-1]].isna().all())

    def test_post_close_data_cannot_change_regular_session_features(self):
        frame, calendar = self.fixture()
        baseline = summarize(frame, calendar)
        later = frame.iloc[[-1]].copy()
        later.timestamp_utc = calendar.close_utc.iloc[0].isoformat()
        later.bar_end_utc = (calendar.close_utc.iloc[0]+pd.Timedelta(minutes=15)).isoformat()
        later['close'] = 1e10
        pd.testing.assert_frame_equal(baseline, summarize(pd.concat([frame, later]), calendar))

    def test_duplicate_bar_rejected(self):
        frame, calendar = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize(pd.concat([frame, frame.iloc[[0]]]), calendar)

    def test_derived_cache_is_verified_and_rebuilt_if_changed(self):
        frame, calendar = self.fixture()
        frame['symbol'] = 'TEST'
        with temporary_directory() as folder:
            raw = folder/'raw'
            raw.mkdir()
            file = raw/'bars.csv.gz'
            frame.to_csv(file,index=False,compression='gzip')
            calendar_path = folder/'calendar.json'
            calendar_path.write_text(json.dumps([dict(date='2024-11-29',open='09:30',close='13:00')]))
            record = dict(file='bars.csv.gz',sha256=preparation.sha(file),rows=len(frame),query=dict(symbol='TEST'))
            cache = folder/'cache'
            with patch.object(preparation,'RAW',raw), patch.object(preparation,'CALENDAR',calendar_path), patch.object(preparation,'CACHE',cache):
                first = preparation.load_summary(record,calendar)
                second = preparation.load_summary(record,calendar)
                pd.testing.assert_frame_equal(first,second)
                with next(cache.glob('*.csv.gz')).open('ab') as handle:
                    handle.write(b'changed')
                pd.testing.assert_frame_equal(first,preparation.load_summary(record,calendar))


if __name__ == '__main__':
    unittest.main()
