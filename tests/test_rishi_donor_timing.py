import unittest
import pandas as pd
from rishi_donor_timing import eligible_donor


def utc(value):
    return pd.Timestamp(value, tz='UTC')


class DonorTimingTests(unittest.TestCase):
    def test_only_same_id_historical_completed_nonoverlapping_label(self):
        donor=dict(id='A', window_start=utc('2022-01-03'), window_end=utc('2022-01-05'),
                   label_start=utc('2022-01-06'), label_end=utc('2022-01-10'))
        recipient=dict(id='A', window_start=utc('2022-01-06'), window_end=utc('2022-01-11'),
                       decision=utc('2022-01-12'), label_start=utc('2022-01-13'), label_end=utc('2022-01-18'))
        self.assertTrue(eligible_donor(donor,recipient))
        self.assertFalse(eligible_donor({**donor,'id':'B'},recipient))
        self.assertFalse(eligible_donor({**donor,'window_end':utc('2022-01-11'),
            'label_start':utc('2022-01-11 01:00'),'label_end':utc('2022-01-11 02:00')},recipient))
        self.assertFalse(eligible_donor({**donor,'label_end':utc('2022-01-13')},recipient))
        self.assertFalse(eligible_donor({**donor,'label_end':utc('2022-01-12 01:00')},recipient))
        self.assertTrue(eligible_donor({**donor,'label_end':recipient['decision']},recipient))
        local={k:(v.tz_convert('America/New_York') if isinstance(v,pd.Timestamp) else v) for k,v in donor.items()}
        self.assertTrue(eligible_donor(local,recipient))

    def test_reject_malformed_or_naive_timestamp(self):
        donor=dict(id='A',window_start=utc('2022-01-03'),window_end=utc('2022-01-05'),
                   label_start=utc('2022-01-06'),label_end=utc('2022-01-10'))
        recipient=dict(id='A',window_start=utc('2022-01-06'),window_end=utc('2022-01-11'),
                       decision=utc('2022-01-12'),label_start=utc('2022-01-13'),label_end=utc('2022-01-18'))
        with self.assertRaises(ValueError):eligible_donor({**donor,'window_end':pd.Timestamp('2022-01-05')},recipient)
        with self.assertRaises(ValueError):eligible_donor({**donor,'window_start':utc('2022-01-07')},recipient)
        with self.assertRaises(ValueError):eligible_donor({**donor,'id':''},recipient)
        with self.assertRaises(ValueError):eligible_donor(donor,{**recipient,'decision':recipient['label_start']})
