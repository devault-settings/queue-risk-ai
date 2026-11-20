# Model experiment log

Every experiment, including the ones that failed, so nobody repeats them and so the slides can show what was tried.

All results are on the **5-year outcome** with **validation data only**, using two checks:
- **Rolling CV:** mean AUC over the yearly folds 2010–2014, fitting on earlier years each time.
- **Out-of-time:** train on 2005–2014, score 2015–2018. This is the harder check, because the queue changed. Neither check touches the 2019–2020 test cohort.

## Nov 18: text-only model for late fusion ([02_paul_text_models.ipynb](../notebooks/02_paul_text_models.ipynb))

**Stage 1, the reader** (paragraph → proceeded vs. blocked/delayed). Trained on Claude's 425 resolved labels, tested once on the 90 resolved human labels. Numbers are from the Colab T4 run on Nov 19 (512 tokens, early stopping, 3 seeds averaged).

| Reader | CV AUC (silver) | Gold AUC [95% CI] |
|---|---|---|
| Keyword rules (`sabin.rule_outcome`) | 0.802 | 0.754 [0.64, 0.86] |
| TF-IDF + logistic regression | 0.815 | 0.840 [0.75, 0.92] |
| Frozen MiniLM + logistic regression | 0.698 | 0.738 [0.62, 0.84] |
| Frozen DistilBERT + logistic regression | 0.785 | 0.882 [0.81, 0.94] |
| **Fine-tuned DistilBERT** (selected) | **0.916** | **0.962 [0.92, 0.99]** |

- **Fine-tuned DistilBERT wins clearly.** On the gold set it beats the keyword rules by +0.209 AUC (95% CI [+0.094, +0.339]) and TF-IDF by +0.120 ([+0.036, +0.210]).
- It recalls 93% of the blocked/delayed paragraphs and 79% of the proceeded ones. It gets 10 of 90 gold paragraphs wrong at a 0.5 cutoff: 6 proceeded paragraphs scored too low, and 4 delayed ones scored too high.
- Out-of-fold over all 515 resolved paragraphs, its AUC is 0.938. These are the reader outcomes that stage 2 uses.

**Stage 2, county text history → built within 5 years**, rolling origin 2015–2018, mean within-year AUC:

| Model | Mean within-year AUC |
|---|---|
| Any earlier contested project | 0.525 |
| + counts, reader outcomes, concerns | 0.525–0.526 |
| + state fallback | 0.513 (hurts) |
| Oracle (labels instead of the reader) | 0.526 |

**Reading it:**
- Only 10% of 2015–2018 projects have county text.
- Where a county has an earlier contested project, 5-year build rates are 4% vs. 10% elsewhere (2010–2018).
- Whether that earlier opposition *succeeded* doesn't matter detectably: "mostly blocked" and "mostly proceeded" counties both build at 4%. So a better reader won't help stage 2: on the GPU run, swapping the reader for the human labels (oracle) gives the same 0.526. The value of the text is in the fusion.

**Dating fix:**
- The labeling sheet's cleaned text drops footnote citations that `sabin_contested.csv` had glued onto paragraphs.
- `text.load_paragraphs()` re-dates from the cleaned text, which changes `first_year` for 123 paragraphs and `last_year` for 114.
- `stack.py` still uses the old dates.

## Nov 15: blank-field leakage fixed; consistent out-of-time reporting

**Found by Charles Loriot** (notebooks/01_charles_tabular_models.ipynb): the "missing" flags were top SHAP drivers. Verified:
- **POI name blanks are outcome-dependent.** In SPP the name is missing for 14% of withdrawn projects and 0% of every other status; ERCOT shows 0% active, 7.5% operational, 29% withdrawn.
- **County blanks too.** ERCOT: 0.6% of active, 36% of operational and 59% of withdrawn projects; CAISO withdrawn 17% vs. about 1% otherwise.
- **Service-type blanks too.** CAISO: 0% active vs. 26% withdrawn; MISO is uneven as well.

The blanks are written *after* the outcome, so they leak it.

**Fix** ([features._hide_missingness](../src/queue_risk/features.py)):
- No missing-value indicators in the model pipelines.
- POI and county-history features left missing, then imputed without a flag.
- ERCOT's missing state set to TX.
- `service` dropped.

**He also caught** that the README paired the selected model's validation AUC (gradient boosting, depth 4) with an out-of-time AUC from a *different* configuration. The pipeline now scores every **selected** model on 2015–2018 as well, and `stack.py` reuses the selected settings (`models.selected()`).

| Model (selected settings) | Rolling CV, before → after fix | Out-of-time 2015–18, before → after |
|---|---|---|
| Logistic regression | 0.660 → 0.615 | — → 0.641 |
| LASSO (C = 0.1) | 0.678 → 0.631 | 0.647 → **0.686** |
| CART | 0.662 → 0.593 | — → 0.610 |
| Random forest | 0.695 → 0.629 | 0.551 → 0.588 |
| Gradient boosting (depth 4) | **0.701 → 0.663** | 0.594 → 0.633 |

**Reading it:** validation fell and out-of-time rose. That's the signature of a leak that lived in old resolved records and didn't transfer.
- LASSO by year, 2015–2018: 0.627 / 0.637 / 0.668 / 0.737. The "2017–2018 breakdown" was mostly the leak.
- The POI features still help without it. Gradient boosting: rolling CV 0.631 → 0.663, out-of-time 0.607 → 0.633. LASSO out-of-time: 0.668 → 0.686.
- **Opposition, second stage:** odds ×0.55 (LASSO stage 1) or ×0.58 (gradient boosting). AUC gain +0.005 with either, interval includes zero.
- **Recommendation:** LASSO as the headline model, since it transfers best, with gradient boosting as the comparison.

**Superseded:** the Nov 12 numbers below include the leak. They're kept as a record.

## Nov 12 (evening): CPU-only push on validation AUC

| Change | Rolling CV | Out-of-time 2015–18 | Verdict |
|---|---|---|---|
| Baseline (18 features) | 0.652 | 0.572–0.580 | — |
| **+ point-of-interconnection (POI) track record**: projects at the same POI and transmission owner before entry (entries, MW, built, withdrawn, built share) | **0.691** | — | **Kept.** Improved all 5 folds |
| + POI text: voltage parsed from the POI name; line tap vs. substation | 0.667 | — | Kept, together with the track record |
| **POI text + track record together** | **0.701** | **0.625** (every year up: 0.632 / 0.673 / 0.528 / 0.543) | **Kept: new default** |
| + transmission-owner track record and 90-day request bunching | 0.641 | — | Dropped (hurts) |
| + Sabin prior contested projects, as a feature | 0.653 alone; about +0.005 inside the combined set | — | Tested in the second stage instead (see below) |
| + 2010 Census county density, land area, centroid lat/lon | 0.675 alone; 0.703 with POI | 0.593 alone; **0.613 with POI** (2017–18 worse) | **Switch only** (`extras=("poi","census")`): it helps rolling CV but transfers worse |
| + EIA state retail price (previous year, 3-year change) | 0.695 (down from 0.701) | — | Dropped |
| Gradient-boosting tuning grid (32 settings: learning rate, iterations, leaves, minimum leaf size, L2) | 0.701 at best | — | No gain over the defaults |
| Rank-average ensemble, gradient boosting + LASSO (Lecture 7) | 0.698–0.702 | — | No meaningful gain |
| Recency-weighted training (weight = exp((year − last year)/τ), τ = 6, 3, 1.5) | 0.681–0.687 | 0.563–0.618 | Dropped: worse on both checks |

### Ladder after the change (`python -m queue_risk.pipeline`, 26 features)

| Model | Before | After |
|---|---|---|
| Flat rate | 0.500 | 0.500 |
| Historical rate | 0.620 | 0.620 |
| Logistic regression | 0.617 | 0.660 |
| LASSO | 0.636 | 0.678 |
| CART | 0.611 | 0.662 |
| Random forest | 0.626 | 0.695 |
| **Gradient boosting** | 0.652 | **0.701** |

### Opposition, second stage (`python -m queue_risk.stack`, 2015–2018, after the POI change)

- **Tabular model alone, out of time:** AUC 0.626 (was 0.572).
- **Effect of the opposition flag:** a county with an earlier contested project gives odds of about ×0.57 at the same tabular score (was ×0.52).
- **Ranking gain:** AUC +0.009, 95% CI [−0.002, +0.020], positive in every year. The tabular model now explains some of what the flag was picking up, but the flag still adds a little.

## Caveats on the new features

- **POI names come from the 2025 snapshot.** If an operator renames a POI as a project advances, the name could carry a little post-entry information. The track-record counts are dated (only projects resolved *before* entry count), so the counts themselves are clean. The name grouping is the residual risk.
- **Census uses 2010 values for every cohort,** a mild anachronism for 2005–2009 entries.
- **Weak cohorts.** Even with the new features, 2017–2018 are only ranked at about 0.53–0.56. Drift is still the main problem for the 2019–2020 test.

## Ideas not yet tried

- A 3-year horizon, so training can include 2015–2016 cohorts that are closer to the test.
- Training on 2005–2014 plus the 2015–2016 cohorts with partial windows, via a discrete-time hazard model.
- Distance from the county centroid to the nearest built project, which needs coordinates.
- TabPFN (GPU, scheduled for tomorrow).
