import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wf_analysis as W


def _days(start, n):
    return np.arange(np.datetime64(start), np.datetime64(start) + n)


def test_concat_by_date_ok_and_rejects_overlap():
    a, b = _days("2019-01-01", 3), _days("2020-01-01", 2)
    d, x = W.concat_by_date([(a, np.ones((2, 3))), (b, np.zeros((2, 2)))])
    assert len(d) == 5 and x.shape == (2, 5) and x[0, 2] == 1 and x[0, 3] == 0
    with pytest.raises(ValueError):
        W.concat_by_date([(a, np.ones((2, 3))), (a, np.ones((2, 3)))])
    with pytest.raises(ValueError):
        W.concat_by_date([(b, np.ones((2, 2))), (a, np.ones((2, 3)))])


def test_year_slices_exact_years():
    d = np.arange(np.datetime64("2016-01-04"), np.datetime64("2023-12-30"))
    d = d[np.is_busday(d)]
    s = W.year_slices(d)
    for y, (ti, end) in s.items():
        assert d[ti].astype("datetime64[Y]").astype(int) + 1970 == y
        assert d[end - 1].astype("datetime64[Y]").astype(int) + 1970 == y
    assert s[2020][0] == s[2019][1]


def test_concat_nulls_shape():
    assert W.concat_nulls([np.zeros((7, 3)), np.ones((7, 4))]).shape == (7, 7)
