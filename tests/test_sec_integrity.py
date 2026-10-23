"""SEC data-integrity tests for the filing-text pipeline (docs/overnight_20261024/OVERNIGHT_TESTS.md).

Synthetic fixtures only; no network, no data files, no returns of any strategy. Complements tests/test_edgar.py,
tests/test_lazyprices.py and tests/test_pit.py without repeating them. A strict xfail marks a verified defect or a
documented deviation from docs/strategy_lazyprices_20261023/PREREG.md; each reason names the review item.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hypershift.data import edgar as ED
from hypershift.strategy import lazyprices as LP, pit_engine as PE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lazy_checks import signal_rows  # noqa: E402

BODY = ('<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>'
        '<div style="display:none"><ix:header><ix:hidden><ix:nonNumeric name="dei:DocumentType">10-K'
        '</ix:nonNumeric><ix:nonNumeric name="dei:EntityRegistrantName">Hiddenco</ix:nonNumeric></ix:hidden>'
        '</ix:header></div>'
        '<p>Our business faces competition and regulation. Revenue was '
        '<ix:nonFraction name="us-gaap:Revenues" scale="6">1,234</ix:nonFraction> million.</p>'
        '<p>Risk factors may adversely affect results.</p></body></html>')
WANT = Counter('our business faces competition and regulation revenue was million risk factors may adversely '
               'affect results'.split())


# ---------------------------------------------------------------- C1: XML declaration (verified on real DFIN 10-Ks)

@pytest.mark.parametrize('decl', ['', '<?xml version="1.0" ?>', '\n<?xml version="1.0"?>\n'])
def test_extract_tokens_without_encoding_declaration(decl):
    """Workiva-style documents start with `<?xml version="1.0" ?>` (no encoding attribute): parsed normally."""
    assert ED.extract_tokens((decl + BODY).encode()) == WANT


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    'C1 (CRITICAL): extract_tokens decodes bytes to str and lxml refuses a str whose XML declaration names an '
    'encoding. Microsoft 10-Ks 0001564590-22-026876 and 0000950170-23-035122 (DFIN inline XBRL) start with '
    '<?xml version="1.0" encoding="ASCII"?> and raise. lazy_signals.parse records the error and the pair gets no '
    'signal. Fix: patches/C1_xml_declaration_and_refetch.patch'))
@pytest.mark.parametrize('decl', ['<?xml version="1.0" encoding="ASCII"?>', "<?xml version='1.0' encoding='utf-8'?>",
                                  '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'])
def test_extract_tokens_with_encoding_declaration(decl):
    assert ED.extract_tokens((decl + BODY).encode()) == WANT


# ---------------------------------------------------------------- formatting-only changes

def test_markup_only_changes_do_not_change_tokens():
    """Inline tags, attributes, hidden XBRL facts, numeric-table values and in-paragraph whitespace are not signal."""
    base = ED.extract_tokens(BODY)
    variants = [
        BODY.replace('<p>Risk factors', '<p style="font-family:Times"><span>Risk</span> <font>factors</font>'),
        BODY.replace('Hiddenco', 'Renamed Hidden Holdings Inc'),                          # hidden dei facts
        BODY.replace('1,234', '9,876,543'),                                                # tagged number in text
        BODY.replace('competition and', 'competition\n   and').replace('may adversely', 'may&#160;adversely'),
        BODY.replace('</body>', '<table><tr><td>Revenue</td><td>1,234</td><td>2,345</td></tr></table></body>'),
    ]
    for v in variants:
        assert ED.extract_tokens(v) == base


def test_numeric_table_threshold_is_digits_over_letters_plus_digits():
    """Cohen et al.'s 15% rule: a table at 10% numeric characters is kept, one at 20% is dropped."""
    keep = '<table><tr><td>' + 'a' * 90 + ' ' + '1' * 10 + '</td></tr></table>'
    drop = '<table><tr><td>' + 'b' * 80 + ' ' + '1' * 20 + '</td></tr></table>'
    tok = ED.extract_tokens('<html><body><p>text here</p>\n' + keep + '\n' + drop + '</body></html>')
    assert tok['a' * 90] == 1 and ('b' * 80) not in tok


@pytest.mark.xfail(strict=True, reason=(
    'I1 (IMPORTANT, design limitation): the frozen rule joins element text without separators, so the same words '
    'tokenise differently when a filing agent emits block elements without whitespace between them. Measured on '
    'real 10-Ks: same-document cosine between the frozen and a whitespace-separated extraction 0.99968 (Apple '
    'FY2023) against a year-over-year cosine of 0.99921, i.e. formatting moves 1 - cosine by ~40% of a stable '
    "filer's yearly change"))
def test_block_whitespace_does_not_change_tokens():
    pretty = ('<html><body>\n<div><span>Risk Factors</span></div>\n<div><span>Our business faces competition'
              '</span></div>\n<table><tr><td>Name</td><td>Title</td></tr></table>\n</body></html>')
    minified = pretty.replace('\n', '')
    assert ED.extract_tokens(pretty) == ED.extract_tokens(minified)


def test_text_signal_is_deterministic_and_symmetric():
    a = ED.extract_tokens(BODY.encode())
    b = ED.extract_tokens(BODY.replace('competition', 'litigation').encode())
    assert ED.extract_tokens(BODY.encode()) == a == ED.extract_tokens(BODY)          # bytes and str agree
    assert ED.cosine(a, b) == ED.cosine(b, a) and 0 < ED.cosine(a, b) < 1
    assert ED.cosine(a, Counter()) == 0. == ED.cosine(Counter(), Counter())          # empty document: defined


# ---------------------------------------------------------------- timing

@pytest.mark.parametrize('json_utc,et', [
    ('2023-03-11T03:00:00.000Z', '2023-03-10T17:00:00'),   # Friday before DST starts (EST, +10 h)
    ('2023-03-13T17:00:00.000Z', '2023-03-13T09:00:00'),   # Monday after DST starts (EDT, +8 h)
    ('2023-11-04T00:30:00.000Z', '2023-11-03T16:30:00'),   # Friday before DST ends (EDT)
    ('2023-11-06T16:00:00.000Z', '2023-11-06T06:00:00'),   # Monday after DST ends (EST)
    ('2019-12-31T03:15:00.000Z', '2019-12-30T17:15:00')])  # year boundary
def test_acceptance_inversion_around_dst_and_year_end(json_utc, et):
    assert ED.acceptance_et(json_utc) == np.datetime64(et)


def _cal(start='2020-01-01', n=160, holidays=()):
    return np.busday_offset(start, np.arange(n), roll='forward', holidays=list(holidays))


def test_filing_on_last_session_before_formation_is_not_traded_at_formation():
    """Accepted on F-1 (before the open or after the close) -> available on F -> first usable at the next formation.
    Accepted on F-2 after the close -> available F-1 -> traded at the open of F, never at a same-day close."""
    cal = _cal()
    F = int(np.searchsorted(cal, np.datetime64('2020-05-01')))
    last = str(cal[F - 1])                                     # 2020-04-30
    prev = str(cal[F - 2])
    rows = []
    for e, ts in enumerate([last + 'T08:00', last + 'T16:05', prev + 'T17:45']):
        rows.append((e, ED.available_index(np.datetime64(ts), cal), .5, '10-Q'))
    sim, _ = LP.signals_at(rows, 3, F, cal)
    assert np.isnan(sim[0]) and np.isnan(sim[1]) and sim[2] == .5
    F2 = int(np.searchsorted(cal, np.datetime64('2020-06-01')))
    sim2, _ = LP.signals_at(rows, 3, F2, cal)
    assert (sim2 == .5).all()


def test_exchange_holiday_with_edgar_open():
    """Good Friday 2020-04-10: EDGAR accepts filings, NYSE is closed. Availability is the next trading day."""
    cal = _cal('2020-04-01', 20, holidays=['2020-04-10'])
    k = ED.available_index(np.datetime64('2020-04-10T10:00'), cal)
    assert cal[k] == np.datetime64('2020-04-13')
    k = ED.available_index(np.datetime64('2020-04-09T19:00'), cal)          # Thursday evening before the holiday
    assert cal[k] == np.datetime64('2020-04-13')


# ---------------------------------------------------------------- comparison selection

def _f(acc, form, report, accepted):
    return {'accession': acc, 'form': form, 'reportDate': report, 'acceptance_et': np.datetime64(accepted)}


def test_prior_must_be_accepted_before_current_never_a_future_filing():
    fs = [_f('late_prior', '10-K', '2018-12-31', '2020-03-15T10:00'),     # FY2018 10-K filed (very) late
          _f('cur', '10-K', '2019-12-31', '2020-02-20T16:00')]
    assert ED.match_prior(ED.original_filings(fs)) == {}


def test_amendment_never_becomes_the_current_or_prior_document():
    fs = [_f('k18', '10-K', '2018-12-31', '2019-02-20T16:00'),
          _f('k18a', '10-K/A', '2018-12-31', '2019-04-29T16:00'),           # Part III amendment, same period
          _f('k19', '10-K', '2019-12-31', '2020-02-20T16:00'),
          _f('k19a', '10-K/A', '2019-12-31', '2020-03-01T16:00')]
    orig = ED.original_filings(fs)
    assert [f['accession'] for f in orig] == ['k18', 'k19']
    assert ED.match_prior(orig) == {'k19': 'k18'}


def test_missing_or_degenerate_comparison_gives_no_signal_without_error():
    fs = [_f('q1', '10-Q', None, '2019-05-01T16:00'),                       # no report date
          _f('q2', '10-Q', '2020-03-31', '2020-05-01T16:00'),               # prior has no report date
          _f('k1', '10-K', '2020-12-31', '2021-02-20T16:00'),               # first 10-K of a new registrant
          _f('q3', '10-Q', '2021-03-28', '2021-05-01T16:00')]               # 52/53-week year: 362 days after q2
    pm = ED.match_prior(ED.original_filings(fs))
    assert pm == {'q3': 'q2'}


def test_duplicate_original_for_same_period_uses_latest_as_prior():
    """Two originals for one period (re-filed as 10-K, not 10-K/A): the later one is the prior, per PREREG."""
    fs = [_f('k18', '10-K', '2018-12-31', '2019-02-20T16:00'),
          _f('k18b', '10-K', '2018-12-31', '2019-02-22T09:00'),
          _f('k19', '10-K', '2019-12-31', '2020-02-20T16:00')]
    pm = ED.match_prior(ED.original_filings(fs))
    assert pm['k19'] == 'k18b' and 'k18b' not in pm and 'k18' not in pm


# ---------------------------------------------------------------- signal validity (PREREG "Validity" and "Comparison")

@pytest.mark.xfail(strict=True, reason=(
    'I3 (IMPORTANT, spec deviation): PREREG says the signal is the latest original filing and that a filing without '
    'a comparison gives no signal; signals_at skips a latest filing with no finite similarity and falls back to an '
    'older one inside the 3-month window. In the pipeline, filings without a prior or with a parse error are absent '
    'from the rows altogether, so the fallback happens silently'))
def test_latest_filing_without_comparison_gives_no_signal():
    cal = _cal()
    F = int(np.searchsorted(cal, np.datetime64('2020-06-01')))
    a_old = int(np.searchsorted(cal, np.datetime64('2020-03-05')))
    a_new = int(np.searchsorted(cal, np.datetime64('2020-05-06')))
    rows = [(0, a_old, .4, '10-K'), (0, a_new, np.nan, '10-Q')]      # latest 10-Q has no comparison
    sim, form = LP.signals_at(rows, 1, F, cal)
    assert np.isnan(sim[0]) and form[0] == ''


def test_changer_ties_are_broken_deterministically_by_entity_order():
    sim = np.array([.5, .5, .5, .5, .5, .9, .9, .9, .9, .9])
    form = np.array(['10-Q'] * 10)
    ch1 = LP.changers(sim, form, np.ones(10, bool))
    ch2 = LP.changers(sim.copy(), form.copy(), np.ones(10, bool))
    assert (ch1 == ch2).all() and np.flatnonzero(ch1).tolist() == [0, 1]


# ---------------------------------------------------------------- identity joins (scripts/lazy_checks.signal_rows)

def _write_inputs(tmp_path, rows, cmap):
    p = tmp_path / 'signals.csv'
    with open(p, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['cik', 'accession', 'form', 'acceptance_et', 'similarity', 'error'])
        w.writerows(rows)
    c = tmp_path / 'cik_map.json'
    c.write_text(json.dumps(cmap))
    return str(p), str(c)


def test_cik_segments_do_not_splice_predecessor_or_successor_filings(tmp_path):
    """Entity X: CIK 100 until 2020-03, CIK 200 from 2020-04 (holding-company reorganisation). A CIK-100 filing
    after its segment (the old registrant keeps filing as a subsidiary) must not reach X; a CIK-200 filing accepted
    before the 120-day look-back must not either."""
    cal = _cal('2019-01-01', 600)
    rows = [(100, 'a1', '10-Q', '2020-02-05T16:00:00', '.9', ''),
            (100, 'a2', '10-Q', '2020-08-05T16:00:00', '.1', ''),       # old CIK after its segment: dropped
            (200, 'b0', '10-Q', '2019-08-05T16:00:00', '.2', ''),       # new CIK long before its segment: dropped
            (200, 'b1', '10-Q', '2020-08-06T16:00:00', '.8', '')]
    cmap = {'X': {'segments': [['2019-01-01', '2020-03-31', 100], ['2020-04-01', '2021-12-31', 200]]}}
    p, c = _write_inputs(tmp_path, rows, cmap)
    out, _ = signal_rows(cal, ['X'], path=p, cmap_path=c)
    got = sorted((e, round(s, 3)) for e, a, s, f in out)
    assert got == [(0, .8), (0, .9)]


def test_errored_rows_never_become_signals(tmp_path):
    cal = _cal('2019-01-01', 400)
    rows = [(100, 'a1', '10-K', '2019-02-20T16:00:00', '0.000000', "ValueError('Unicode strings ...')"),
            (100, 'a2', '10-Q', '2019-05-01T16:00:00', '.95', '')]
    p, c = _write_inputs(tmp_path, rows, {'X': {'segments': [['2019-01-01', '2020-06-30', 100]]}})
    out, recs = signal_rows(cal, ['X'], path=p, cmap_path=c)
    assert [(e, f) for e, a, s, f in out] == [(0, '10-Q')] and len(recs) == 2


def test_one_accession_shared_by_two_share_classes_gives_one_row_per_entity(tmp_path):
    """GOOG/GOOGL, FOX/FOXA, NWS/NWSA, DISCA/DISCK and UA/UAA are simultaneous members sharing one CIK: each filing
    is one observation per share class (consistent with an equal-weight benchmark holding both classes; MINOR M2).
    Within one entity a filing never appears twice."""
    cal = _cal('2019-01-01', 400)
    rows = [(300, 'g1', '10-Q', '2019-04-29T16:00:00', '.7', '')]
    cmap = {'GOOG': {'segments': [['2019-01-01', '2020-06-30', 300]]},
            'GOOGL': {'segments': [['2019-01-01', '2020-06-30', 300]]}}
    p, c = _write_inputs(tmp_path, rows, cmap)
    out, _ = signal_rows(cal, ['GOOG', 'GOOGL'], path=p, cmap_path=c)
    assert sorted(e for e, *_ in out) == [0, 1]
    assert len({(e, a, f) for e, a, s, f in out}) == len(out)


# ---------------------------------------------------------------- name matching look-alikes

@pytest.mark.parametrize('wiki,edgar', [('Ford Motor Company', 'FORD CREDIT AUTO OWNER TRUST'),
                                        ('Ball Corp', 'BALLANTYNE STRONG INC'),
                                        ('Lennar Corp.', 'LENNOX INTERNATIONAL INC'),
                                        ('Fox Corporation', 'FOX FACTORY HOLDING CORP')])
def test_name_match_rejects_lookalikes(wiki, edgar):
    assert not ED.name_match([wiki], [edgar], strict=True)


@pytest.mark.xfail(strict=True, reason=(
    'M5 (MINOR): strict name matching accepts a reporting subsidiary whose name extends the parent name '
    '("Duke Energy" vs "DUKE ENERGY CAROLINAS, LLC"); subsidiaries file 10-Ks, so the span check passes too. '
    'Only the 5 fallback CIKs (CIK_MAP.md) rely on name matching; the primary source is the dated Wikipedia CIK'))
def test_name_match_rejects_reporting_subsidiary():
    assert not ED.name_match(['Duke Energy'], ['DUKE ENERGY CAROLINAS, LLC'], strict=True)


# ---------------------------------------------------------------- portfolio mechanics around corporate events

def _panel(N=4, T=40):
    close = np.full((N, T), 100.)
    close[2] = 100 * 1.01 ** np.arange(T)
    open_ = np.concatenate([close[:, :1], close[:, :-1]], 1)                  # no overnight move
    observed = np.ones((N, T), bool)
    observed[2, 21:] = False                                                   # name 2 delisted after t=20
    close[2, 21:] = np.nan
    open_[2, 21:] = np.nan
    return SimpleNamespace(close=close, open=open_, observed=observed, eligible=observed.copy(),
                           dates=np.arange('2020-01-01', T, dtype='datetime64[D]'))


def test_delisting_goes_to_cash_and_cash_is_redeployed_at_next_rebalance():
    d = _panel()
    tg = {10: np.full(4, .25), 30: np.full(4, .25)}
    r = PE.run_targets(tg, d, cost_bp=10)
    assert np.isfinite(r['gross']).all() and np.isfinite(r['net']).all()
    k21, k30 = 21 - 10, 30 - 10
    liquidated = r['delisted_weight'][k21]
    assert liquidated > .25 and r['gross'][k21] == 0.                         # weight grew, sold at last close
    assert r['traded'][k30] == pytest.approx(liquidated)                       # only the cash is redeployed
    assert r['net'][k30] == pytest.approx(-10e-4 * liquidated)


def test_halted_name_keeps_weight_and_targets_scale_to_remaining_nav():
    d = _panel()
    d.observed[2, 21:] = True
    d.close[2, 21:] = d.close[2, 20]
    d.open[2, 21:] = d.close[2, 20]
    d.observed[1, 29:32] = False                                               # name 1 halted over the rebalance
    d.open[1, 29:32] = np.nan
    tg = {10: np.full(4, .25), 30: np.array([0., 0., .5, .5])}
    r = PE.run_targets(tg, d, cost_bp=0)
    assert np.isfinite(r['gross']).all()
    # name 1 cannot trade at t=30: it keeps its weight w1; names 0,2,3 are set to targets scaled to 1 - w1
    assert r['traded'][30 - 10] < 1.0


def test_avoidance_weight_is_not_capped_at_1_25x():
    """PREREG: 'Maximum weight per name is about 1.25 x B's'. This is descriptive, not a cap: an industry with
    4 changers out of 5 gives its remaining name 5 x B (MINOR M3; lazy_eval reports the realised maximum)."""
    elig = np.ones(10, bool)
    ind = np.array([7] * 5 + [1] * 5)
    ch = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    b, a = LP.avoid_weights(elig, ch, ind)
    assert a[4] == pytest.approx(5 * b[4]) and a.sum() == pytest.approx(1.)
