"""Conservative timing guard for future same-security donor windows.

This does not choose donors, inspect financial values, or authorize interventions.
All label intervals are closed, and all timestamps must be timezone-aware.
"""
import pandas as pd


def _timestamp(record, key):
    try:
        value=pd.Timestamp(record[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {key}') from exc
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError(f'{key} must be a timezone-aware timestamp')
    return value.tz_convert('UTC')


def _window(record, recipient=False):
    if not isinstance(record.get('id'),str) or not record['id'].strip():
        raise ValueError('Stable security ID required')
    a,b,c,d=(_timestamp(record,k) for k in ('window_start','window_end','label_start','label_end'))
    if not (a<=b<c<=d):
        raise ValueError('Invalid window/label chronology')
    if recipient:
        decision=_timestamp(record,'decision')
        if not b<=decision<c:
            raise ValueError('Decision must follow window and precede label')
        return a,b,c,d,decision
    return a,b,c,d


def eligible_donor(donor, recipient):
    """Return whether an earlier same-ID donor is causal under strict guards."""
    _,donor_end,_,donor_label_end=_window(donor)
    _,recipient_end,recipient_label_start,_,decision=_window(recipient,recipient=True)
    return bool(donor['id']==recipient['id'] and donor_end<recipient_end and
                donor_label_end<=decision and donor_label_end<recipient_label_start)
