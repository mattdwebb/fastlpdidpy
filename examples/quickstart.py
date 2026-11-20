from pathlib import Path
import pandas as pd
from fastlpdidpy import fit

data = pd.read_csv(Path(__file__).with_name("school_panel.csv"))
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
