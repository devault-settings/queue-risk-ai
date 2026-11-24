# The one-time test run: checklist

The 2019–2020 cohort (3,184 projects) has never been scored. It gets scored **once**, by the frozen specification in [`src/queue_risk/frozen.py`](../src/queue_risk/frozen.py), with the whole team agreeing beforehand. Whatever it shows is reported as-is: no re-tuning, no re-picking models afterwards.

## Why the specification is frozen in a file
Model selection depends on the environment. On Colab, LASSO selected C = 0.01; on the laptop, C = 0.1. If `--final` re-selected settings, the "final" model would depend on whose machine ran it. So `--final` never re-selects anything: it uses only `frozen.py`. Before it touches the test cohort, `preflight()` checks that:
- the configuration (outcome, horizon, regions, windows, features) matches the spec;
- the data reproduce the frozen counts for each window;
- LASSO reproduces its frozen out-of-time AUC (0.686 ± 0.003), which confirms the environment matches the one the README numbers came from;
- no team decision is pending, and a sign-off was given.

## Before the run (team, together)
- [ ] **Decide the odd rows:** drop projects with 0 or missing capacity and solar/wind paired with fossil units ("Wind+Gas")? Validation is unaffected either way (0.686). Set `"exclude_odd_rows"` in `frozen.py` to `True` or `False`.
- [ ] **Review the frozen models and settings** in `frozen.py`. Headline: LASSO (C = 0.1); also flat rate, historical rate, logistic, CART, random forest, gradient boosting. Any change happens now, in a commit that says why.
- [ ] **Decide on TabPFN in the test.** It needs a GPU, so `--final` skips it on the laptop. Options: (a) report its validation numbers only; (b) run the test on Colab too. Option (b) means LASSO's reproduction check must pass there, and Colab's library versions may move it beyond ±0.003. If so, record that and run TabPFN alone.
- [ ] **Dry run on the machine that will run the test:** `python -m queue_risk.final`. Everything must show `ok` except the sign-off.
- [ ] **Agree who runs it and when**, ideally with everyone watching (screen share).

## The run
```bash
git pull                                   # everyone on the same commit
python tests/selftest.py                   # must end "All checks passed."
python -m queue_risk.pipeline --final --signoff "Audrey, Paul, Charles, Sara, <date>"
python -m queue_risk.stack    --final --signoff "Audrey, Paul, Charles, Sara, <date>"
python -m queue_risk.timeline --final --signoff "Audrey, Paul, Charles, Sara, <date>"
```
Each run appends its results, commit hash and sign-off to `reports/final_run_log.md`. The pipeline also writes `reports/results_within_test.csv`.

## After the run
- [ ] Commit `reports/final_run_log.md` and `reports/results_within_test.csv` right away, unchanged.
- [ ] Fill the "pending" test rows in the README, the one-page draft and Sara's slide 5, **from the log**.
- [ ] If the test disagrees with validation, say so plainly. Lecture 13's pilot-to-rollout lesson applies: the 2019–2020 base rate (3.7%) is the lowest of any cohort.
- [ ] Don't change anything and re-run. If a bug turns up, fix it, then report both runs and explain why.
