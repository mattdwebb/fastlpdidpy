from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pytest
from fastlpdidpy import fit, IdentificationError


def fixture():
    i = np.repeat(np.arange(1, 13), 10)
    t = np.tile(np.arange(1, 11), 12)
    g = np.where(i <= 4, 3, np.where(i <= 8, 5, 0))
    d = ((g > 0) & (t >= g)).astype(int)
    x = np.cos(i * .71 + t * .33) + i / 12
    y = i*.2+t*.3+np.sin(i*t*.7)+d*(2+.2*g+.1*(t-g))
    return pd.DataFrame(dict(i=i,t=t,g=g,d=d,x=x,y=y,y2=np.cos(i*t*.41)))


def call(f=None, **kw):
    return fit(fixture() if f is None else f,"y","t",unit="i",gvar="g",event_window=(-2,2),**kw)


@pytest.mark.parametrize("weighting",["variance","reweighted"])
@pytest.mark.parametrize("covariates",[None,["x"]])
def test_linearity(weighting,covariates):
    f=fixture(); a=call(f,weighting=weighting,covariates=covariates)
    b=call(f.assign(y=f.y2),weighting=weighting,covariates=covariates)
    z=call(f.assign(y=1.31*f.y-.73*f.y2),weighting=weighting,covariates=covariates)
    assert z["estimate"] == pytest.approx(1.31*a["estimate"]-.73*b["estimate"],abs=1e-11)
    assert z["sample_rows"] == a["sample_rows"]


@pytest.mark.parametrize("weighting",["variance","reweighted"])
def test_homogeneous_effect(weighting):
    f=fixture(); f.y=.2*f.i+.3*f.t+2*f.d
    r=call(f,weighting=weighting)
    assert r["estimate"] == pytest.approx(2,abs=1e-12)
    assert [x["estimate"] for x in r["horizons"]] == pytest.approx([0,0,2,2,2],abs=1e-12)


def test_treatment_equivalence():
    a=call()
    b=fit(fixture(),"y","t",unit="i",treatment="d",event_window=(-2,2))
    assert a["estimate"] == b["estimate"]


def test_bad_history():
    f=fixture(); f.loc[(f.i==1)&(f.t==7),"d"]=0
    with pytest.raises(ValueError,match="nonabsorbing"):
        fit(f,"y","t",unit="i",treatment="d",event_window=(0,2))


def test_no_defaults_and_unknown_options():
    with pytest.raises(ValueError,match="explicit event_window"):
        fit(fixture(),"y","t",unit="i",gvar="g")
    with pytest.raises(NotImplementedError,match="unsupported options"):
        call(weights="x")
    with pytest.raises(NotImplementedError,match="cross sections"):
        call(data_mode="rcs")


def test_gaps_are_not_compressed():
    f=fixture(); f=f.loc[~((f.i==1)&(f.t==2))]
    r=call(f)
    assert r["n_treated"] == 7
    assert r["estimate"] == pytest.approx(call(f.sample(frac=1,random_state=3))["estimate"])


def test_missing_pooled_not_skipped():
    f=fixture(); f.loc[(f.i==1)&(f.t==4),"y"]=np.nan
    r=call(f)
    assert r["n_treated"]==7
    assert r["horizons"][2]["n_treated"]==8


def test_jackknife():
    f=fixture(); r=call(f,inference="jackknife")
    vals=[call(f.loc[f.i!=i])["estimate"] for i in range(1,13)]
    se=np.sqrt(11/12*np.sum((vals-np.mean(vals))**2))
    assert r["inference"]["jackknife"]["se"]==pytest.approx(se,abs=1e-12)
    assert r["inference"]["jackknife"]["df"]==11


def test_reassignment():
    f=fixture(); obs=f.groupby("i").g.first().to_numpy()
    ass=np.array([np.roll(obs,1),np.roll(obs,2)])
    r=call(f,inference=["ri_beta"],assignments=ass)
    vals=[]
    for row in ass:
        ff=f.copy(); ff.g=ff.i.map(dict(zip(range(1,13),row)))
        z=call(ff,inference="point")
        vals.append(z["estimate"])
    assert r["inference"]["ri_beta"]["draws"]==pytest.approx(vals)
    with pytest.raises(ValueError,match="unique"):
        call(inference="ri_beta",assignments=np.array([obs]))


def test_seed_and_native_ri():
    a=call(inference="ri_beta",reps=3,seed=92)
    b=call(inference="ri_beta",reps=3,seed=92)
    assert a["assignments"]==b["assignments"]
    r=call(inference="ri_t_native",assignments=a["assignments"])
    assert 0 < r["inference"]["ri_t_native"]["p"] <= 1


def _worker(weighting):
    return call(weighting=weighting)["estimate"]


def test_process_isolation():
    with ProcessPoolExecutor(max_workers=2) as pool:
        actual=list(pool.map(_worker,["variance","reweighted"]))
    assert actual==pytest.approx([_worker("variance"),_worker("reweighted")])


def test_r_lm_oracle():
    path=Path(__file__).parent/"fixtures"/"r_lm_oracle.csv"
    assert path.exists(), "Generate independent base-R lm oracle first"
    for row in pd.read_csv(path).itertuples():
        r=call(weighting=row.weighting,covariates=["x"] if row.covariates else None)
        got=r if row.horizon==999 else next(x for x in r["horizons"] if x["horizon"]==row.horizon)
        assert got["estimate"]==pytest.approx(row.estimate,abs=1e-10)
        if not row.covariates or row.weighting=="variance":
            assert got["native_se"]==pytest.approx(row.se,abs=1e-10)


def test_official_stata_oracle_and_r_cross_language():
    for name in ["stata_oracle.csv","r_port.csv"]:
        for row in pd.read_csv(Path(__file__).parent/"fixtures"/name).itertuples():
            r=call(weighting=row.weighting,covariates=["x"] if row.covariates else None)
            got=r if row.horizon==999 else next(x for x in r["horizons"] if x["horizon"]==row.horizon)
            assert got["estimate"]==pytest.approx(row.estimate,abs=1e-9)
            assert got["native_se"]==pytest.approx(row.se,abs=1e-9)
    f=fixture(); obs=f.groupby("i").g.first().to_numpy()
    ass=np.array([np.roll(obs,1),np.roll(obs,2)])
    r=call(inference=["jackknife","ri_beta","ri_t_native"],assignments=ass)
    expected=pd.read_csv(Path(__file__).parent/"fixtures"/"r_inference.csv").iloc[0]
    assert r["inference"]["jackknife"]["se"]==pytest.approx(expected.jk_se,abs=1e-10)
    for m in ["ri_beta","ri_t_native"]:
        assert r["inference"][m]["p"]==pytest.approx(expected[m],abs=1e-12)


def test_student_t_tails():
    from fastlpdidpy.core import _tpvalue
    # Independent exact Cauchy/Student-t(1) identity and published t critical values.
    for t in [0,.001,.1,1,4,100]:
        assert _tpvalue(t,1)==pytest.approx(1-2*np.arctan(abs(t))/np.pi,abs=1e-13)
    assert _tpvalue(2.2281388519649385,10)==pytest.approx(.05,abs=3e-12)
    assert _tpvalue(1.9839715185235522,100)==pytest.approx(.05,abs=1e-11)


if __name__=="__main__":
    root=Path(__file__).parent/"fixtures"
    root.mkdir(exist_ok=True)
    fixture().to_csv(root/"panel.csv",index=False)
