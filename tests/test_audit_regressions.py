import unittest
import numpy as np
import pandas as pd
from fastlpdidpy import fit, IdentificationError

def fixture():
    i=np.repeat(np.arange(1,13),10);t=np.tile(np.arange(1,11),12)
    g=np.where(i<=4,3,np.where(i<=8,5,0));d=(g>0)&(t>=g)
    return pd.DataFrame(dict(i=i,t=t,g=g,y=i*.2+t*.3+np.sin(i*t*.7)+d*(2+.2*g+.1*(t-g))))

def call(d,**kw):
    return fit(d,'y','t',unit='i',gvar='g',event_window=(0,2),**kw)

class AuditRegressions(unittest.TestCase):
    def test_singleton_cohort_reaggregated(self):
        d=fixture();d=d.loc[~d.i.isin([6,7,8])]
        self.assertTrue(np.isfinite(call(d,inference='point')['estimate']))
        jk=call(d,inference='jackknife')['inference']['jackknife']
        expected=[call(d[d.i!=i],inference='point')['estimate'] for i in sorted(d.i.unique())]
        np.testing.assert_allclose(jk['deletions'],expected)

    def test_supported_deletions_still_work(self):
        out=call(fixture(),inference='jackknife')
        self.assertEqual(len(out['inference']['jackknife']['deletions']),12)
        self.assertTrue(np.isfinite(out['inference']['jackknife']['se']))

    def test_unused_input_cluster_is_explicit(self):
        d=fixture();extra=pd.DataFrame([dict(i=99,t=1,g=0,y=1.)])
        out=call(pd.concat([d,extra]),inference='jackknife')
        jk=out['inference']['jackknife']
        self.assertEqual(out['n_clusters'],12)
        self.assertEqual(jk['cluster_count'],13)
        self.assertEqual(jk['inactive_clusters'],[99])

if __name__=='__main__': unittest.main()
