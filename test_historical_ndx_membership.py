import unittest
from unittest.mock import Mock
import pandas as pd
from historical_ndx_membership import monthly_members, read_year


class HistoricalIndexTests(unittest.TestCase):
    def test_later_addition_does_not_change_earlier_month(self):
        initial = {f'S{i:03}' for i in range(100)}
        event = dict(effective_date=pd.Timestamp('2022-06-20'), removed=['S000'], added=['NEW'])
        years = {2022: dict(initial=initial, events=[event])}
        frame = monthly_members(years, '2022-01-01', '2022-12-01')
        june = set(frame[frame.month == '2022-06-01'].ticker)
        july = set(frame[frame.month == '2022-07-01'].ticker)
        self.assertIn('S000', june)
        self.assertNotIn('NEW', june)
        self.assertIn('NEW', july)
        self.assertNotIn('S000', july)

    def test_share_classes_are_not_truncated_to_100_securities(self):
        initial = {f'S{i:03}' for i in range(101)}
        years = {2022: dict(initial=initial, events=[])}
        frame = monthly_members(years, '2022-01-01', '2022-01-01')
        self.assertEqual(len(frame), 101)
        self.assertTrue(frame.review_status.eq('provisional_not_training_input').all())

    def test_absent_removal_fails_instead_of_silently_repairing(self):
        path = Mock()
        path.name = 'bad.yaml'
        text = 'year: 2022\ntickers_on_Jan_1:\n'
        text += ''.join(f'  - S{i:03}\n' for i in range(100))
        text += "changes:\n  '2022-06-20':\n    difference: [ABSENT]\n    union: [NEW]\n"
        path.read_text.return_value = text
        with self.assertRaisesRegex(ValueError, 'absent member'):
            read_year(path)

    def test_on_semiconductor_symbol_is_not_yaml_boolean(self):
        path = Mock()
        path.name = 'on.yaml'
        text = 'year: 2022\ntickers_on_Jan_1:\n  - ON\n'
        text += ''.join(f'  - S{i:03}\n' for i in range(99))
        text += "changes:\n  '2022-06-20':\n    difference: [ON]\n    union: [NEW]\n"
        path.read_text.return_value = text
        data = read_year(path)
        self.assertIn('ON', data['initial'])
        self.assertEqual(data['events'][0]['removed'], ['ON'])


if __name__ == '__main__':
    unittest.main()
