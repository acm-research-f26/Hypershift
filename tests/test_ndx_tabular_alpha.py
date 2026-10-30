import unittest
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from ndx_tabular_alpha import causal_styles,standardize_train_only,normalized_labels


class TabularAlphaTests(unittest.TestCase):
    def test_long_history_features_cannot_see_future(self):
        rng=np.random.default_rng(7);p=100*np.exp(np.cumsum(rng.normal(0,.01,(350,4)),axis=0))
        r=np.r_[np.zeros((1,4)),p[1:]/p[:-1]-1]
        data=dict(marks=p,returns=r,adv=np.full_like(p,1e7))
        q=np.linspace(100,150,350);a,names,c=causal_styles(data,q)
        changed={k:v.copy() for k,v in data.items()}
        for value in changed.values():value[301:]*=2
        future=q.copy();future[301:]*=3;b,other,d=causal_styles(changed,future)
        np.testing.assert_array_equal(a[:301],b[:301]);np.testing.assert_array_equal(c[:301],d[:301])
        self.assertEqual(names,other)

    def test_imputation_and_scaling_use_observed_training_only(self):
        x=np.arange(24,dtype=float).reshape(3,4,2);x[0,1,0]=np.nan
        mask=np.ones((3,4),bool);mask[0,3]=False
        a,mean,sd=standardize_train_only(x,np.array([0]),mask)
        x[1:]=1e8;x[0,3]=1e8
        b,m2,s2=standardize_train_only(x,np.array([0]),mask)
        np.testing.assert_array_equal(mean,m2);np.testing.assert_array_equal(sd,s2)
        np.testing.assert_array_equal(a[0,:3],b[0,:3])

    def test_labels_are_date_centered_and_unobserved_values_excluded(self):
        y=np.array([[1.,2.,3.,np.nan]]);mask=np.array([[1,1,1,0]],bool)
        a=normalized_labels(y,mask);b=normalized_labels(y+100,mask)
        np.testing.assert_allclose(a,b);self.assertEqual(a[0,-1],0)

    def test_forest_is_reproducible_and_shared_across_stock_ids(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(300,8));y=x[:,0]+rng.normal(size=300)*.1
        outputs=[]
        for _ in range(2):
            model=ExtraTreesRegressor(n_estimators=24,max_depth=4,min_samples_leaf=8,
                bootstrap=True,max_samples=.8,max_features=.75,n_jobs=2,random_state=19).fit(x,y)
            model.n_jobs=1;outputs.append(model.predict(np.r_[x[:10],x[:10]]))
        np.testing.assert_array_equal(outputs[0],outputs[1])
        np.testing.assert_array_equal(outputs[0][:10],outputs[0][10:])


if __name__=='__main__':unittest.main()
