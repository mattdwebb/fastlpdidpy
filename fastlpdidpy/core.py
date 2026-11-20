"""Native DGJT reference estimator; no cross-language runtime dependencies."""
from itertools import permutations
from math import factorial, lgamma, log, log1p, exp, erfc, sqrt
import numpy as np
import pandas as pd


class IdentificationError(ValueError):
    """The requested LP contrast is unidentified on its clean sample."""


def _tpvalue(t, df):
    """Two-sided Student t tail via regularized incomplete beta continued fraction."""
    x = df / (df + t*t)
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    a, b = df/2, .5
    swap = x > (a+1)/(a+b+2)
    if swap:
        a, b, x = b, a, 1-x
    tiny = 1e-300
    c = 1.0
    d = 1-(a+b)*x/(a+1)
    d = 1/max(d, tiny)
    h = d
    for m in range(1, 10001):
        for aa in (m*(b-m)*x/((a+2*m-1)*(a+2*m)),
                   -(a+m)*(a+b+m)*x/((a+2*m)*(a+2*m+1))):
            d = 1+aa*d
            if abs(d)<tiny:
                d=tiny
            c = 1+aa/c
            if abs(c)<tiny:
                c=tiny
            d=1/d
            change=d*c
            h*=change
        if abs(change-1)<2e-14:
            break
    else:
        raise ArithmeticError("Student t tail did not converge")
    p=exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*log(x)+b*log1p(-x))*h/a
    return float(np.clip(1-p if swap else p,0,1))


def _prepare(data, y, time, unit, gvar, treatment, cluster, covariates):
    if unit is None:
        raise NotImplementedError("UNSUPPORTED: LP-DiD requires a genuine panel unit")
    if gvar is None and treatment is None:
        raise ValueError("supply gvar or treatment")
    f = pd.DataFrame(data).copy().sort_values([unit, time]).reset_index(drop=True)
    cluster = cluster or unit
    if f[[unit, time, cluster]].isna().any().any():
        raise ValueError("unit, time and cluster cannot be missing")
    if f.duplicated([unit, time]).any():
        raise ValueError("duplicate unit-time rows")
    t = f[time].to_numpy(dtype=float)
    if not np.isfinite(t).all() or not np.equal(t, np.floor(t)).all():
        raise ValueError("time must be integer period indices; gaps are not compressed")
    f["_time"] = t
    f["_unit"] = f[unit]
    f["_cluster"] = f[cluster]
    if f.groupby("_unit")["_cluster"].nunique().max() > 1:
        raise ValueError("inference cluster must be constant within unit")
    if gvar is not None:
        g = f[gvar].astype(float).fillna(np.inf).replace(0, np.inf)
        if (np.isneginf(g) | (np.isfinite(g) & (g != np.floor(g)))).any():
            raise ValueError("cohorts must be integer dates, zero, missing, or +inf")
        f["_g"] = g
        if f.groupby("_unit")["_g"].nunique().max() > 1:
            raise ValueError("gvar must be constant within unit")
    if treatment is not None:
        d = f[treatment].to_numpy(dtype=float)
        if not np.isin(d, [0, 1]).all():
            raise ValueError("treatment must be nonmissing and binary")
        f["_d"] = d
        if (f.groupby("_unit")["_d"].diff() < 0).any():
            raise ValueError("nonabsorbing treatment is unsupported")
        if gvar is not None and not np.array_equal(d, (t >= f["_g"]).astype(float)):
            raise ValueError("gvar and treatment are inconsistent")
        if gvar is None:
            # Unobserved switches are not assigned an invented adoption date.
            prev = f.groupby("_unit")["_d"].shift()
            pt = f.groupby("_unit")["_time"].shift()
            if ((d == 1) & (prev == 0) & (t - pt != 1)).any():
                raise ValueError("adoption inside a panel gap: supply known gvar")
            g = f.loc[f._d == 1].groupby("_unit")["_time"].min()
            f["_g"] = f._unit.map(g).fillna(np.inf)
    else:
        f["_d"] = (t >= f._g).astype(float)
    f["_y"] = f[y].astype(float)
    for j, col in enumerate(covariates):
        f[f"_x{j}"] = f[col].astype(float)
    return f


def _at(f, col, offset):
    source = f.set_index(["_unit", "_time"])[col]
    key = pd.MultiIndex.from_arrays([f._unit, f._time + offset])
    return source.reindex(key).to_numpy(dtype=float)


def _regress(f, horizons, weighting, nx, lag):
    z = np.mean(np.column_stack([_at(f, "_y", h) for h in horizons]), axis=1) - _at(f, "_y", -1)
    d = f._d.to_numpy() - _at(f, "_d", -1)
    end = max(0, max(horizons))
    clean = (d == 1) | (_at(f, "_d", end) == 0)
    x = np.column_stack([_at(f, f"_x{j}", -lag) for j in range(nx)]) if nx else np.empty((len(f), 0))
    keep = clean & np.isfinite(d) & np.isfinite(z) & np.isfinite(x).all(axis=1)
    s = pd.DataFrame({"t": f._time, "d": d})
    n = s.loc[keep].groupby("t").d.transform("size")
    nt = s.loc[keep].groupby("t").d.transform("sum")
    if weighting == "reweighted" and nx:
        keep[np.flatnonzero(keep)] &= (nt < n).to_numpy()
    elif weighting == "reweighted":
        keep[np.flatnonzero(keep)] &= ((nt > 0) & (nt < n)).to_numpy()
    else:
        # reghdfe removes singleton time groups, including all-control cells.
        keep[np.flatnonzero(keep)] &= (n > 1).to_numpy()
    ids = np.flatnonzero(keep)
    if not len(ids) or not np.any(d[ids] == 1) or not np.any(d[ids] == 0):
        raise IdentificationError("no treated/clean-control overlap")
    z, d, x = z[ids], d[ids], x[ids]
    times = f._time.to_numpy()[ids]
    fe = (times[:, None] == np.unique(times)[None, :]).astype(float)
    c = pd.factorize(f._cluster.iloc[ids], sort=True)[0]
    G = len(np.unique(c))
    if weighting == "reweighted" and nx:
        X = np.column_stack([fe, x])
        X0 = X[d == 0]
        if np.linalg.matrix_rank(X0) < X.shape[1]:
            raise IdentificationError("singular control regression")
        bread = np.linalg.inv(X0.T @ X0)
        beta = bread @ X0.T @ z[d == 0]
        resid = z - X @ beta
        estimate = np.mean(resid[d == 1])
        m = X[d == 1].mean(axis=0)
        score = (d * (resid - estimate) - (1-d) * resid * (X @ bread @ m) * sum(d)) / sum(d)
        correction = 1.0
        convention = "RA unconditional cluster sandwich CR0"
        rank = X.shape[1]
    else:
        X = np.column_stack([d, fe, x])
        w = np.ones(len(d))
        if weighting == "reweighted":
            for t in np.unique(times):
                ix = times == t
                w[ix] = ix.sum() / np.sum(d[ix] == 0)
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            raise IdentificationError("singular LP regression")
        bread = np.linalg.inv(X.T @ (w[:, None] * X))
        beta = bread @ X.T @ (w * z)
        estimate = beta[0]
        score = (X @ bread[:, 0]) * w * (z - X @ beta)
        correction = G / (G-1) * (len(d)-1) / (len(d)-rank) if G > 1 and len(d) > rank else np.nan
        convention = "cluster CR1: G/(G-1)*(N-1)/(N-K), time FE counted"
    scores = np.bincount(c, weights=score, minlength=G)
    se = np.sqrt(correction * np.dot(scores, scores)) if G > 1 else np.nan
    p = None if not np.isfinite(se) or se == 0 else (erfc(abs(estimate/se)/sqrt(2)) if nx and weighting == "reweighted" else _tpvalue(estimate/se,G-1))
    return dict(estimate=float(estimate), native_se=float(se), native_df=(None if nx and weighting == "reweighted" else G-1),
                native_p=p, native_p_status="normal for RA, Student t(G-1) otherwise",
                native_vce=convention, native_validation="official lpdid 1.1.0 deterministic fixture; not full certification",
                nobs=len(ids), n_treated=int(sum(d)), n_clusters=G,
                cohorts_used=sorted(np.unique(times[d == 1]).tolist()), sample_rows=ids.tolist(),
                support_times=sorted(np.unique(times).tolist()),
                estimation_clusters=sorted(f._cluster.iloc[ids].unique().tolist()))


def fit(data, y, time, unit=None, gvar=None, treatment=None, cluster=None,
        covariates=None, event_window=None, data_mode="auto", **options):
    """Fit pooled and horizon LP-DiD. See README for exact supported options."""
    if data_mode not in ("auto", "panel", "rcs"):
        raise ValueError("data_mode must be auto, panel, or rcs")
    if data_mode == "rcs":
        raise NotImplementedError("UNSUPPORTED: repeated cross sections lack unit outcome differences")
    allowed = {"weighting", "covariate_lag", "inference", "assignments", "reps", "seed", "null"}
    if set(options) - allowed:
        raise NotImplementedError(f"unsupported options: {sorted(set(options)-allowed)}")
    if event_window is None or len(event_window) != 2:
        raise ValueError("explicit event_window=(start,end) required; no default horizon")
    lo, hi = event_window
    if not all(np.isfinite(v) and int(v) == v for v in [lo, hi]) or lo > hi or hi < 0:
        raise ValueError("integer event window must be ordered and include a nonnegative horizon")
    lo, hi = int(lo), int(hi)
    weighting = options.get("weighting", "reweighted")
    if weighting not in ("reweighted", "variance"):
        raise ValueError("weighting must be reweighted or variance")
    covariates = [covariates] if isinstance(covariates, str) else list(covariates or [])
    lag = options.get("covariate_lag", 1)
    if lag not in (0, 1):
        raise ValueError("covariate_lag must be 1 (default) or 0 (explicit pretransformed covariates)")
    f = _prepare(data, y, time, unit, gvar, treatment, cluster, covariates)
    post = list(range(max(0, lo), hi+1))
    point = lambda ff: _regress(ff, post, weighting, len(covariates), lag)
    out = point(f)
    out.update(status="PASS", estimator="lpdid", data_mode="panel", weighting=weighting,
               target_label=f"{weighting} pooled LP-DiD {'RA ' if covariates and weighting == 'reweighted' else ''}ATT [{post[0]},{hi}]",
               event_window=[lo, hi], pooled_horizons=post, covariates_used=covariates,
               covariate_timing=f"t-{lag}", horizons=[], inference={})
    for h in range(lo, hi+1):
        if h == -1:
            row = dict(estimate=0., native_se=None, status="REFERENCE")
        else:
            try:
                row = _regress(f, [h], weighting, len(covariates), lag)
                row["status"] = "PASS"
            except IdentificationError as exc:
                row = dict(estimate=None, status="IDENTIFICATION_FAILURE", error=str(exc))
        out["horizons"].append(dict(horizon=h, **row))
    methods = options.get("inference", ["native"])
    methods = [methods] if isinstance(methods, str) else list(methods)
    if set(methods) - {"point", "native", "jackknife", "ri_beta", "ri_t_native"}:
        raise ValueError("unknown inference method")
    null = float(options.get("null", 0))
    if null != 0:
        raise NotImplementedError("only null=0 is supported; nonzero sharp-null outcome adjustment is not implemented")
    clusters = sorted(f._cluster.unique())

    def jk(ff):
        vals = []
        full = point(ff)
        for c in clusters:
            try:
                reduced = point(ff.loc[ff._cluster != c].reset_index(drop=True))
                vals.append(reduced['estimate'])
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise IdentificationError(f"jackknife deletion {c!r} failed: {exc}") from exc
        vals = np.array(vals)
        se = np.sqrt((len(vals)-1)/len(vals) * np.sum((vals-vals.mean())**2))
        tolerance = 64*np.finfo(float).eps*max(1.,abs(full['estimate']),float(np.max(np.abs(vals))))
        if not np.isfinite(se) or se <= tolerance:
            raise IdentificationError('zero jackknife variance')
        p = _tpvalue((full["estimate"]-null)/se,len(vals)-1)
        return dict(se=float(se), df=len(vals)-1, deletions=vals.tolist(), p=p,
                    cluster_count=len(clusters), cluster_scope='all input inference clusters',
                    inactive_clusters=[c for c in clusters if c not in full['estimation_clusters']],
                    support='surviving pooled cohort/time support')

    if "jackknife" in methods:
        out["inference"]["jackknife"] = jk(f)
    if any(m in methods for m in ["ri_beta","ri_t_native"]):
        cg = f.groupby("_cluster")._g
        if cg.nunique().max() != 1:
            raise ValueError("RI requires one treatment cohort per inference/assignment cluster")
        observed = cg.first().reindex(clusters).to_numpy()
        supplied = options.get("assignments")
        if supplied is None:
            counts = np.unique(observed, return_counts=True)[1]
            total = factorial(len(observed))
            for n in counts:
                total //= factorial(int(n))
            reps = options.get("reps", 99)
            if not isinstance(reps, int) or reps < 1:
                raise ValueError("reps must be positive integer")
            R = min(reps, total-1)
            rng = np.random.default_rng(options.get("seed", 0))
            if total <= 10000 and len(observed) <= 9:
                pool = sorted(set(permutations(observed)) - {tuple(observed)})
                chosen = rng.choice(len(pool), R, replace=False)
                assignments = np.array([pool[i] for i in chosen])
            else:
                seen = {tuple(observed)}
                rows = []
                for _ in range(max(1000, 100*R)):
                    a = tuple(rng.permutation(observed))
                    if a not in seen:
                        rows.append(a)
                        seen.add(a)
                    if len(rows) == R:
                        break
                if len(rows) != R:
                    raise ValueError("unique assignment sampling exhausted; supply assignments")
                assignments = np.asarray(rows)
        else:
            assignments = np.asarray(supplied, dtype=float)
            assignments = np.where(np.isnan(assignments) | (assignments == 0), np.inf, assignments)
        if assignments.ndim != 2 or assignments.shape[1] != len(clusters) or not len(assignments):
            raise ValueError("assignments must be nonempty R by number-of-clusters")
        seen = {tuple(observed)}
        for a in assignments:
            if not np.array_equal(np.sort(a), np.sort(observed)) or tuple(a) in seen:
                raise ValueError("assignments must preserve cohort counts, be unique, and exclude observed")
            seen.add(tuple(a))
        obsstat = {"ri_beta": out["estimate"]-null}
        draws = {"ri_beta": []}
        if "ri_t_native" in methods:
            if not np.isfinite(out["native_se"]) or out["native_se"]<=np.finfo(float).eps:
                raise IdentificationError("invalid native variance")
            obsstat["ri_t_native"]=(out["estimate"]-null)/out["native_se"]
            draws["ri_t_native"]=[]
        for a in assignments:
            ff = f.copy()
            ff["_g"] = ff._cluster.map(dict(zip(clusters, a)))
            ff["_d"] = (ff._time >= ff._g).astype(float)
            result = point(ff)
            b = result["estimate"] - null
            draws["ri_beta"].append(b)
            if "ri_t_native" in methods:
                se=result["native_se"]
                if not np.isfinite(se) or se<=np.finfo(float).eps:
                    raise IdentificationError("invalid randomized native variance")
                draws["ri_t_native"].append(b/se)
        for method in draws:
            if method in methods:
                v = np.asarray(draws[method])
                tolerance = 64*np.finfo(float).eps*np.maximum(1.,np.maximum(np.abs(v),abs(obsstat[method])))
                p = (1+np.sum(np.abs(v) >= abs(obsstat[method])-tolerance))/(len(v)+1)
                out["inference"][method] = dict(p=float(p), statistic=obsstat[method], draws=v.tolist())
        out["assignments"] = np.where(np.isfinite(assignments), assignments, 0).tolist()
        out["assignment_clusters"] = clusters
    return out
