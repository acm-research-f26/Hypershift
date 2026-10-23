import numpy as np
import pytest
from collections import Counter

from hypershift.data import edgar as ED


def test_acceptance_et_inverts_sec_double_conversion():
    # index-page "Accepted" (ET) vs submissions JSON, checked on live filings 2026-10-23
    assert ED.acceptance_et('2026-01-29T02:07:34.000Z') == np.datetime64('2026-01-28T16:07:34')   # EST, MSFT 10-Q
    assert ED.acceptance_et('2026-07-31T14:01:02.000Z') == np.datetime64('2026-07-31T06:01:02')   # EDT, AAPL 10-Q
    assert ED.acceptance_et('2023-11-03T02:08:27.000Z') == np.datetime64('2023-11-02T18:08:27')   # EDT, AAPL 10-K


def test_available_day_is_first_trading_day_after_acceptance_date():
    cal = np.array(['2023-11-02', '2023-11-03', '2023-11-06', '2023-11-07'], dtype='datetime64[D]')
    assert ED.available_index(np.datetime64('2023-11-02T06:00'), cal) == 1       # morning filing: still next day
    assert ED.available_index(np.datetime64('2023-11-03T18:08'), cal) == 2       # Friday evening -> Monday
    assert ED.available_index(np.datetime64('2023-11-04T10:00'), cal) == 2       # Saturday -> Monday
    assert ED.available_index(np.datetime64('2023-11-07T09:00'), cal) == 4       # beyond calendar -> len(cal)


HTML = """<html><body>
<div style="display:none"><ix:header><ix:hidden>dei:EntityRegistrantName Hidden Corp</ix:hidden></ix:header></div>
<script>var tracking = 1;</script><style>p {color: red}</style>
<p>Our Risk factors &amp; uncertainties increased.</p>
<table><tr><td>Revenue</td><td>1,234</td><td>5,678</td><td>9,999</td></tr></table>
<table><tr><td>Item 1A. Risk factors discussion remains</td></tr></table>
<div style="font-size:10pt; DISPLAY: none">invisible words</div>
<p>Page 12 a</p>
</body></html>"""


def test_extract_tokens_drops_hidden_xbrl_scripts_numeric_tables():
    tok = ED.extract_tokens(HTML)
    assert 'hidden' not in tok and 'entityregistrantname' not in tok and 'invisible' not in tok
    assert 'tracking' not in tok and 'color' not in tok
    assert 'revenue' not in tok                       # numeric table removed
    assert tok['risk'] == 2 and tok['factors'] == 2 and tok['remains'] == 1   # text table kept
    assert tok['uncertainties'] == 1 and 'amp' not in tok
    assert 'page' in tok and 'a' not in tok and '12' not in tok


def test_cosine():
    a = Counter({'x': 2, 'y': 1})
    assert ED.cosine(a, a) == pytest.approx(1.)
    assert ED.cosine(a, Counter({'z': 3})) == 0.
    assert ED.cosine(a, Counter({'x': 1, 'y': 2})) == pytest.approx(4 / 5)


def _f(acc, form, report, accepted):
    return {'accession': acc, 'form': form, 'reportDate': report, 'acceptance_et': np.datetime64(accepted)}


def test_original_filings_and_prior_match():
    fs = [_f('a1', '10-K', '2016-12-31', '2017-02-20T16:00'),
          _f('a2', '10-K/A', '2016-12-31', '2017-04-01T10:00'),
          _f('a3', '10-Q', '2017-03-31', '2017-05-01T16:00'),
          _f('a4', '10-K', '2017-12-31', '2018-02-21T16:00'),
          _f('a5', '10-Q', '2018-03-31', '2018-05-02T16:00'),
          _f('a6', '10-KT', '2018-06-30', '2018-09-01T16:00'),
          _f('a7', '10-K', '2018-09-30', '2018-11-20T16:00')]     # fiscal-year change: 273 days after 2017-12-31
    orig = ED.original_filings(fs)
    assert [f['accession'] for f in orig] == ['a1', 'a3', 'a4', 'a5', 'a7']
    pm = ED.match_prior(orig)
    assert pm == {'a4': 'a1', 'a5': 'a3'}


@pytest.mark.parametrize('sic,ind', [(3571, 6), (2834, 10), (6022, 11), (4911, 8), (1311, 4), (5331, 9),
                                     (3711, 2), (2080, 1), (4813, 7), (2821, 5), (3560, 3), (7011, 12), (None, 12)])
def test_ff12(sic, ind):
    assert ED.ff12(sic) == ind


@pytest.mark.parametrize('wiki,edgar,ok', [
    ('3M Company', '3M CO', True), ('CSX Corp.', 'CSX CORP', True), ('AT&T Inc.', 'AT&T INC.', True),
    ("Kohl's Corp.", 'KOHLS Corp', True), ('DuPont', 'DU PONT E I DE NEMOURS & CO', True),
    ('Red Hat Inc.', 'RED HAT INC', True), ('BB&T Corporation', 'BB&T CORP', True),
    ('Endo International', 'ENDONOVO THERAPEUTICS, INC.', False), ('Twenty-First Century Fox', 'FOX FACTORY HOLDING CORP', False),
    ('Dow Inc.', 'Churchill Downs Inc', False), ('Gap Inc.', 'GAP INC', True)])
def test_name_match(wiki, edgar, ok):
    assert ED.name_match([wiki], [edgar]) == ok


def test_name_match_strict_and_accents():
    assert ED.name_match(['Estée Lauder Companies'], ['ESTEE LAUDER COMPANIES INC'])
    assert ED.name_match(['Paramount'], ['Paramount Gold Nevada Corp.'])                 # lenient: prefix
    assert not ED.name_match(['Paramount', 'CBS Corp.'], ['Paramount Gold Nevada Corp.'], strict=True)
    assert ED.name_match(['Red Hat Inc.'], ['RED HAT INC'], strict=True)
    assert ED.name_match(['Paramount Global'], ['Paramount Global'], strict=True)


def test_extract_tokens_handles_xml_encoding_declaration():
    doc = '<?xml version="1.0" encoding="utf-8"?>\n<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>' \
          '<div style="display:none"><ix:header>hidden facts</ix:header></div><p>Results of operations improved</p></body></html>'
    for d in (doc, doc.encode('utf-8')):
        tok = ED.extract_tokens(d)
        assert tok['results'] == 1 and tok['operations'] == 1 and 'hidden' not in tok


def test_available_index_never_before_official_filing_date():
    cal = np.array(['2023-11-02', '2023-11-03', '2023-11-06'], dtype='datetime64[D]')
    assert ED.available_index(np.datetime64('2023-11-02T10:00'), cal, filing_date='2023-11-03') == 2
    assert ED.available_index(np.datetime64('2023-11-02T10:00'), cal, filing_date='2023-11-02') == 1
