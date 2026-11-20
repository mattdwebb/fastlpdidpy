# Release candidate, 20 November 2026

- Integrated the preserved local estimator/deletion corrections with the current remote base.
- Added an executable school-policy guide with actual output and interpretation; retained historical developer notes separately.
- Reconciled license metadata, packaged example data, and added install/example checks.
- Clarified that all jackknife deletions are required, while an estimable surviving aggregate may remain after a cohort disappears.

# Changelog

## 0.1.0 Development (2026-09-29)

Native absorbing-panel pooled and horizon LP-DiD in Python/R/Stata.
Explicit variance/equal-treated weighting, covariate regression adjustment,
cluster inference, full-recomputation JK and RI, deterministic supplied assignments.
Official lpdid 1.1.0 comparisons and independent lm regression tests.
Unsupported features fail explicitly. This is a validated reference subset,
not a full-feature parity or full-certification release.
