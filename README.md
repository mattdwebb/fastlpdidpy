# fastlpdidpy

**Local-projection DiD in Python.** Estimate effects at selected times after adoption using a panel of the same units observed over time.

The Dube–Girardi–Jordà–Taylor estimator compares changes around adoption with clean controls at each horizon, then constructs the specified pooled effect. The example uses school-year means: schools, not individual pupils, are the panel units.

## Install and run

Use Python 3.10 or newer. Install directly from GitHub; this package is not distributed on PyPI. Clone the repository
(or download and extract its ZIP), then run these commands from its root:

```sh
git clone https://github.com/mattdwebb/fastlpdidpy.git
cd fastlpdidpy
python -m pip install "fastlpdidpy[examples] @ git+https://github.com/mattdwebb/fastlpdidpy.git"
python examples/quickstart.py
```

The installer supplies the numerical dependencies and pandas for this example.
No R or Stata installation is required. A virtual environment is recommended
when installing this package. Record the source commit for reproducible work.

## A school-policy example

The example is a panel of **12 schools over eight years** (96 school-year observations). The outcome is the school’s mean test score, with its mean baseline score as a covariate. School identifiers refer to the same units in every year. Four schools adopt a policy in 2013, four in 2015, and four never adopt. The outcome is in score points, and the inference cluster is the school. These are fictional teaching data, not evidence about an actual school policy. The [data guide](examples/DATA_GUIDE.md) defines every column and links the deterministic generator.

The complete calculation in `examples/quickstart.py` is:

```python
import pandas as pd
from fastlpdidpy import fit

data = pd.read_csv("examples/school_panel.csv")
result = fit(data, "test_score", "year", unit="school", gvar="first_treat",
             covariates=["baseline_score"], event_window=(0, 1),
             inference=["native", "jackknife"])
estimate, native_se = result["estimate"], result["native_se"]
jackknife = result["inference"]["jackknife"]
jk_se, jk_p = jackknife["se"], jackknife["p"]
print(f"Estimate: {estimate:.6f} score points")
print("Native clustered SE: " + (f"{native_se:.6f}" if native_se is not None else "not available"))
print(f"Delete-school jackknife SE: {jk_se:.6f}")
print(f"Jackknife p-value: {jk_p:.6f}")
print("Schools: 12; jackknife reference: t(11)")
```

Actual output (rounded to six decimal places):

```text
Estimate: 1.601279 score points
Native clustered SE: 0.331234
Delete-school jackknife SE: 0.365699
Jackknife p-value: 0.001102
Schools: 12; jackknife reference: t(11)
```

The adjusted estimate is **1.60 score points**. The delete-school standard error is **0.37 points**: it measures how sensitive the estimate is to leaving out schools one at a time. The reported jackknife p-value tests a zero effect using Student t with 11 degrees of freedom. The native and jackknife standard errors are different calculations around the same point estimate; a larger or smaller standard error is not automatically the better one.

The generator adds a two-point policy effect, along with school effects, year
effects and varying disturbances. The fitted estimate need not be exactly two.
This example teaches the interface; it does not measure test size or establish
the causal assumptions needed for an application.

## Adapt the analysis

Change `covariates=["baseline_score"]` to `covariates=[]` for an unadjusted specification. For your own data, replace the CSV and column names, keep a constant adoption year within each school, and check the supported input rules below. Numeric covariates must be prepared before fitting.

Save the compact result, for example:

```python
pd.DataFrame([{"estimate": estimate, "native_se": native_se,
               "jackknife_se": jk_se, "jackknife_p": jk_p}]).to_csv(
    "policy_results.csv", index=False)
```

Use `inference="native"` or `inference="jackknife"` to request either
calculation alone. `result["horizons"]` contains the separate horizon estimates.
The example averages horizons zero and one; changing `event_window` changes
the requested pooled effect. With covariates the native reference is normal;
the standalone jackknife uses t with the number of input schools minus one.


## Supported scope and failures

Only genuine absorbing-treatment panels are supported; repeated cross sections are not. Choose the pooled `event_window` explicitly. Reweighted estimation is the default; `weighting="variance"` selects the alternative variance weighting. Covariates use their lag-one values by default. Observation weights, nonabsorbing treatment and nonzero-null RI are unsupported.

The jackknife uses every input school deletion. If a cohort disappears, the
estimator recomputes the supported aggregate and its weights over surviving
estimable effects. It does not discard failed deletions to produce a standard
error. A remaining aggregate must still be identified. Estimates from different
DiD methods need not target the same weighted effect when treatment effects vary.

## Help, references and checks

The [release-candidate validation record](RELEASE_VALIDATION.md) gives the exact

See [method](METHOD_SPECIFICATION.md), [inference](INFERENCE.md), [validation](VALIDATION.md), and [paper/source attribution](PROVENANCE.md). These pages contain the detailed estimator and inference
conventions. Report reproducible problems through the repository’s
[issue tracker](https://github.com/mattdwebb/fastlpdidpy/issues), including package
version, a small synthetic example and the full error message.

To run the numerical tests and the executable README check:

```sh
git clone https://github.com/mattdwebb/fastlpdidpy.git
cd fastlpdidpy
python -m pip install "fastlpdidpy[test,examples] @ git+https://github.com/mattdwebb/fastlpdidpy.git"
python -m pytest -q
```

The [MIT license](LICENSE) applies to this implementation; linked provenance
notes identify external methods and validation software. Cite the underlying
estimator as well as the software revision used.
