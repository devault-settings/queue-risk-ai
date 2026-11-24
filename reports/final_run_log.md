# Final (test-set) runs

Appended automatically by `--final`. Do not edit.

## pipeline --final: 2026-11-24 16:37 | commit f726fcb | sign-off: Audrey DeVault, Paul Bustarret, Charles Loriot, Sara Alafifi, Nov 24 2026

- Flat rate (rule of thumb): test AUC 0.500 [0.500, 0.500], PR-AUC 0.037
- Historical rate (region x tech): test AUC 0.587 [0.539, 0.636], PR-AUC 0.046
- Logistic regression: test AUC 0.545 [0.496, 0.597], PR-AUC 0.043
- LASSO logistic: test AUC 0.613 [0.561, 0.666], PR-AUC 0.060
- CART: test AUC 0.475 [0.431, 0.521], PR-AUC 0.041
- Random forest: test AUC 0.454 [0.400, 0.511], PR-AUC 0.034
- Gradient boosting: test AUC 0.530 [0.472, 0.588], PR-AUC 0.043
- TabPFN (pre-trained transformer): skipped (not available in this environment)
- table: reports\results_within_test.csv

## stack --final: 2026-11-24 16:39 | commit f726fcb | sign-off: Audrey DeVault, Paul Bustarret, Charles Loriot, Sara Alafifi, Nov 24 2026

- first stage: LASSO logistic (frozen); second stage fit on 2015-2018
- test 3,181 projects: tabular only AUC 0.613, + opposition AUC 0.623
- AUC gain from opposition +0.010, 95% CI [+0.001, +0.018], positive in 99% of draws

## timeline --final: 2026-11-24 16:42 | commit f726fcb | sign-off: Audrey DeVault, Paul Bustarret, Charles Loriot, Sara Alafifi, Nov 24 2026

- decision cohort 2019-2020: 3,181 projects
- horizon 2, trained on 2005-2017 (6,433), scored on 3,181, AUC 0.777, predicted rate 0.008, actual rate 0.004
- horizon 3, trained on 2005-2016 (5,473), scored on 3,181, AUC 0.761, predicted rate 0.018, actual rate 0.008
- horizon 4, trained on 2005-2015 (4,570), scored on 3,181, AUC 0.650, predicted rate 0.046, actual rate 0.017
- horizon 5, trained on 2005-2014 (4,020), scored on 3,181, AUC 0.559, predicted rate 0.061, actual rate 0.037
- horizon 7, trained on 2005-2012 (3,404), scored on window not closed yet, AUC nan, predicted rate nan, actual rate nan
