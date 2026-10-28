import unittest
from rishi_sec_supplement import choose_candidate


class SecSupplementTests(unittest.TestCase):
    def test_current_candidate_is_only_discovery(self):
        self.assertEqual(choose_candidate('X',{'X':[12]},[]),('candidate',12))
        self.assertEqual(choose_candidate('X',{'X':[12,13]},[]),('ambiguous_current',None))
        self.assertEqual(choose_candidate('X',{},[]),('no_current_candidate',None))

    def test_prior_ambiguity_cannot_be_resolved_by_unrelated_current_cik(self):
        self.assertEqual(choose_candidate('X',{'X':[12]},[11,13]),('historical_conflict',None))
        self.assertEqual(choose_candidate('X',{'X':[12]},[11,12]),('candidate',12))
