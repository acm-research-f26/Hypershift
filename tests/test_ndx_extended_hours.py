import json
import unittest
import numpy as np
import pandas as pd
import torch

from ndx_extended_hours import ExtendedModel,raw_auxiliary,scale_auxiliary,timing
from prepare_ndx_extended_hours import summarize,FIELDS
from prepare_ndx_alpaca import calendar_frame,filter_identity


def calendar():
    return calendar_frame([dict(date='2024-11-27',open='09:30',close='16:00'),
        dict(date='2024-11-29',open='09:30',close='13:00'),dict(date='2024-12-02',open='09:30',close='16:00')])


def bars(date,start,periods):
    ts=pd.date_range(date+' '+start,periods=periods,freq='15min',tz='America/New_York').tz_convert('UTC')
    price=np.arange(periods,dtype=float)+100
    return pd.DataFrame(dict(timestamp_utc=ts.astype(str),bar_end_utc=(ts+pd.Timedelta(minutes=15)).astype(str),
        open=price,high=price+2,low=price-1,close=price+1,vwap=price+.5,volume=100.,trade_count=10))


class SessionTests(unittest.TestCase):
    def test_early_close_and_ah_anchor_are_calendar_aware(self):
        f=pd.concat([bars('2024-11-29','12:45',1),bars('2024-11-29','13:00',28)],ignore_index=True)
        s=summarize(f,calendar(),'after_hours').iloc[0]
        self.assertEqual(s.observed,28);self.assertEqual(s.coverage,1.)
        self.assertAlmostEqual(s['return'],128/101-1)
        self.assertEqual(s.last_trade_age,0.)

    def test_premarket_cutoff_excludes_unfinished_bars_and_future_regular_data(self):
        f=bars('2024-12-02','04:00',24)
        before=summarize(f,calendar(),'premarket')
        self.assertEqual(before.observed.iloc[0],21)
        altered=f.copy();altered.loc[21:,['open','high','low','close','vwap']]*=1e6
        after=summarize(altered,calendar(),'premarket')
        pd.testing.assert_frame_equal(before,after)

    def test_sparse_or_zero_volume_bars_never_create_numeric_prices(self):
        f=bars('2024-12-02','04:00',4);f.loc[3,'volume']=0
        s=summarize(f,calendar(),'premarket').iloc[0]
        self.assertEqual(s.observed,3);self.assertTrue(s[FIELDS[:7]].isna().all())
        self.assertGreater(s.last_trade_age,0.)

    def test_same_window_split_units_leave_returns_and_dollar_notional_unchanged(self):
        f=bars('2024-12-02','04:00',21);g=f.copy()
        g[['open','high','low','close','vwap']]/=4;g.volume*=4
        a=summarize(f,calendar(),'premarket');b=summarize(g,calendar(),'premarket')
        np.testing.assert_allclose(a[FIELDS],b[FIELDS],atol=1e-12,rtol=0)

    def test_identity_rules_apply_to_bar_session_date(self):
        frame=pd.DataFrame(dict(date=['2022-06-08','2022-06-09']))
        good,bad=filter_identity(frame,'META',dict(META=dict(first_valid_date='2022-06-09')))
        self.assertEqual(good.date.tolist(),['2022-06-09']);self.assertEqual(len(bad),1)

    def test_dst_windows_remain_local_and_decision_precedes_early_close(self):
        cal=calendar_frame([dict(date='2024-03-08',open='09:30',close='16:00'),dict(date='2024-03-11',open='09:30',close='16:00')])
        self.assertNotEqual(cal.open_utc.iloc[0].hour,cal.open_utc.iloc[1].hour)
        base=dict(panel=type('Panel',(),dict(dates=calendar().index))(),origin=np.array([0,1]),entry=np.array([1,2]))
        value=timing(base,calendar());self.assertTrue((pd.to_datetime(value.decision_utc)<pd.to_datetime(value.fill_utc)).all())


class AlignmentTests(unittest.TestCase):
    def test_pm_shift_is_next_exchange_row_not_next_calendar_day(self):
        cube=np.broadcast_to(np.arange(22)[:,None,None],(22,3,10)).astype('float32').copy()
        base=dict(x=np.zeros((2,16,3,5),dtype='float32'),origin=np.array([16,20]))
        raw=raw_auxiliary(base,dict(after_hours=cube,premarket=cube*100),'after_hours_premarket')
        self.assertEqual(raw[0,-1,0,0],16);self.assertEqual(raw[0,-1,0,10],1700)
        self.assertEqual(raw[1,-1,0,10],2100)

    def test_future_and_inactive_values_cannot_change_training_normalizer(self):
        rng=np.random.default_rng(7);raw=rng.normal(size=(20,16,3,20));mask=np.ones((20,3),bool);mask[:,2]=False
        first=scale_auxiliary(raw,mask,np.arange(10))
        raw[10:]=1e20;raw[:,:,2]=1e20
        second=scale_auxiliary(raw,mask,np.arange(10))
        np.testing.assert_array_equal(first[1],second[1]);np.testing.assert_array_equal(first[2],second[2])


class NetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);self.model=ExtendedModel(6,[set(range(6))],'after_hours')
        self.x=torch.randn(2,16,6,5)*.1;self.extra=torch.randn(2,6,320);self.mask=torch.ones(2,6);self.mask[:,-1]=0

    def test_zero_start_preserves_core_exactly_and_core_remains_trainable(self):
        baseline=self.model.core(self.x,self.mask)[0];score=self.model(self.x,self.extra,self.mask)[0]
        torch.testing.assert_close(score,baseline,rtol=0,atol=0)
        (score-torch.randn_like(score)).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in self.model.core.parameters()))
        self.assertGreater(float(self.model.last.weight.grad.norm()),0.)

    def test_daily_control_is_exact_original_initialization_and_ignores_auxiliary(self):
        from robust_models import make_model
        torch.manual_seed(19);model=ExtendedModel(6,[set(range(6))],'daily')
        torch.manual_seed(19);original=make_model('think_mix',6,[set(range(6))],causal_membership=True)
        for name,value in original.state_dict().items():torch.testing.assert_close(value,model.core.state_dict()[name],rtol=0,atol=0)
        a=model(self.x,self.extra,self.mask)[0];b=model(self.x,self.extra*1e9,self.mask)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_inactive_nan_cannot_affect_predictions(self):
        torch.nn.init.normal_(self.model.last.weight)
        a=self.model(self.x,self.extra,self.mask)[0]
        self.extra[:,-1]=float('nan');self.x[:,:,-1]=float('nan')
        b=self.model(self.x,self.extra,self.mask)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
