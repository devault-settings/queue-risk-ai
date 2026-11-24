# Post-hoc test scores

After the signed-off test run of Nov 24 (reports/final_run_log.md). These do NOT change the reported results or the headline model.

## 2026-11-24 21:59 | commit b47606e | environment: GPU (Colab) | run: notebooks/tabpfn_posthoc_test_ran.ipynb

- LASSO logistic: test AUC 0.566 [0.516, 0.621] | out-of-time 2015-18 0.669 | top 20% built 5.8% (1.57x average)
- Gradient boosting: test AUC 0.531 [0.473, 0.587] | out-of-time 2015-18 0.641 | top 20% built 5.2% (1.40x average)
- TabPFN (pre-trained transformer): test AUC 0.530 [0.471, 0.589] | out-of-time 2015-18 0.645 | top 20% built 4.7% (1.27x average)

**Correction (added Nov 24, on the laptop):** the "LASSO logistic" line above is NOT a LASSO. Colab's scikit-learn is older than 1.8, where `l1_ratio=1.0` is ignored unless `penalty="elasticnet"`, so it fit an L2 (ridge) model. Refitting L2 at C = 0.1 on the laptop reproduces it exactly (out-of-time 0.669, test 0.566). The real LASSO (signed-off run, laptop, scikit-learn 1.9.1) scores **0.613** on test and 0.686 out of time. Fixed in `models.L1_PENALTY`; the selftest now checks that the LASSO really is L1. TabPFN and gradient boosting are unaffected.
