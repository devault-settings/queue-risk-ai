# Model experiment log

Every experiment, including the ones that failed, so nobody repeats them and so the slides can show what was tried.

All results are on the **5-year outcome** with **validation data only**, using two checks:
- **Rolling CV:** mean AUC over the yearly folds 2010–2014, fitting on earlier years each time.
- **Out-of-time:** train on 2005–2014, score 2015–2018. This is the harder check, because the queue changed. Neither check touches the 2019–2020 test cohort.


## Nov 24: THE ONE-TIME TEST RUN (2019–2020; signed off by all four)

Run from commit `f726fcb` with the frozen spec ([frozen.py](../src/queue_risk/frozen.py)): odd rows excluded, TabPFN validation-only. The preflight passed, including LASSO reproducing 0.686 out of time. Raw output: [reports/final_run_log.md](../reports/final_run_log.md), committed unchanged. Test cohort: 3,181 projects, 118 built within 5 years (3.7%).

| Model | Test AUC (2019–20) [95% CI] | Out-of-time AUC (2015–18) |
|---|---|---|
| **LASSO** (headline) | **0.613** [0.561, 0.666] | 0.686 |
| Historical rate (rule of thumb) | 0.587 [0.539, 0.636] | 0.593 |
| Logistic regression | 0.545 [0.496, 0.597] | 0.641 |
| Gradient boosting | 0.530 [0.472, 0.588] | 0.633 |
| Flat rate | 0.500 | 0.500 |
| CART | 0.475 [0.431, 0.521] | 0.610 |
| Random forest | 0.454 [0.400, 0.511] | 0.588 |

- **LASSO remains the best model and beats chance**, but its edge over the rule of thumb is small (0.613 vs. 0.587, overlapping intervals). By year: 0.555 for 2019 and 0.658 for 2020.
- **The flexible models collapse**: gradient boosting 0.530, CART 0.475, random forest 0.454. That's the drift pattern the 2017–18 validation hinted at, and it vindicates choosing LASSO on out-of-time performance.
- **Shortlist value** (reporting, computed after the run from the same frozen LASSO, no changes):
  - Top 20% (636 projects): 39 built (6.1%) vs. 23.6 expected at random, so **1.65×**, down from 2.0× on 2015–18.
  - Top 50: 10% built.
- **Calibration:** LASSO predicts 2.3× more builds than happened. Rankings transfer; probabilities need recalibrating on the latest closed cohort before anyone quotes them.
- **Opposition (stack --final):** + opposition test AUC **0.623**, gain **+0.010 [+0.001, +0.018]**, positive in 99% of draws. That's stronger than validation suggested (+0.001 with corrected dates), but small and borderline.
- **Time-to-build (timeline --final):** the 2-year AUC is **0.777**, 3-year **0.761**, 4-year 0.650, 5-year 0.559. Short horizons rank well, and their training windows reach closer to the test years. Probabilities run high (5-year: 6.1% predicted vs. 3.7% actual).
- **Wasted-spend tables** from `pipeline --final` are too noisy to quote: about one build per 25 picks, and several are infinite (no builds in the shortlist).

## Nov 24 (evening): opposition second stage with Paul's corrected paragraph dates

`stack.py` now dates paragraphs with `text.load_paragraphs()`. These are re-dated from the cleaned labeling text, without the footnote citations that had added later years. Same 2015–2018 validation design, laptop:

| First stage | Old dates: odds, AUC gain | Corrected dates: odds, AUC gain [95% CI] |
|---|---|---|
| LASSO | ×0.55, +0.005 | **×0.63, +0.001 [−0.005, +0.008]** (positive in 66% of draws) |
| Gradient boosting | ×0.58, +0.005 | **×0.67, +0.000 [−0.008, +0.007]** |

- The share of 2015–2018 projects in a county with an earlier contested project falls from 13% to 10%.
- **Reading:** contested counties still build less often, but the flag **no longer improves the ranking** on top of the tabular model. Part of the earlier gain came from the footnote dates, as Paul suspected (Nov 18 entry).
- **To reconcile with Paul:** his late fusion (Nov 20, below) reports +0.011 (95% CI +0.002 to +0.019) on mean within-year AUC with the re-dated flag plus the reader score. Here, the per-year gains are +0.008 / +0.006 / −0.001 / −0.003. The second-stage designs differ (his: mean within-year AUC; here: pooled AUC, leave-one-year-out). Agree on one before quoting either.

## Nov 24: TabPFN (Colab T4; [run 1](../notebooks/tabpfn_colab_ran_1.ipynb), [run 2](../notebooks/tabpfn_colab_ran_2.ipynb))

TabPFN 3.5 (pre-trained tabular transformer), imputed before it sees the data, so no blank-field leak. Validation only. **All rows come from the same Colab environment.** Its library versions shift LASSO and gradient boosting slightly from the laptop numbers, and LASSO's selected C there was 0.01, not 0.1.

| Model | Rolling CV | Out-of-time 2015–18 | 2015 / 2016 / 2017 / 2018 (stage 1 alone) |
|---|---|---|---|
| Historical rate | 0.620 | 0.593 | — |
| **LASSO** | 0.635 | **0.682** | 0.635 / 0.634 / 0.646 / 0.717 |
| Gradient boosting | 0.666 | 0.651 | 0.667 / 0.633 / 0.591 / 0.643 |
| **TabPFN** | **0.677** | 0.646 | 0.696 / 0.670 / 0.566 / 0.564 |

- **TabPFN fits the training years best but transfers worst to 2017–18.** Pre-training doesn't protect against the queue changing. LASSO remains the headline model. For the slides: Lecture 8's "trees usually win on tables" holds, and the simplest regularized model beats both.
- **Opposition as a second stage, by first-stage model:**

| First stage | AUC gain (95% CI) | Contested county: odds |
|---|---|---|
| LASSO | +0.007 [−0.001, +0.017] | ×0.52 |
| Gradient boosting | +0.006 [−0.003, +0.015] | ×0.59 |
| TabPFN | +0.005 [−0.002, +0.013] | ×0.61 |

  The effect is consistent across all three; the ranking gain is small and every interval touches zero.
- **Run 1** failed only at the Sabin step (`pdftotext` missing after a stale `apt` index). That's fixed in `tabpfn_colab.ipynb`.

## Nov 20: how lucky is one demo sample? ([demo/export_data.py](../demo/export_data.py))

The demo page shows 50 random 2015–2018 projects and asks how many of the model's top 10 were built. With only about 4–5 builds per sample, one draw can mislead, so `export_data.py` redraws 50 projects 20,000 times from the same scored cohort (LASSO, fit on 2005–2014, scored out of time):

| | Model's top 10 | Random 10 (expected) |
|---|---|---|
| Average over 20,000 draws | 1.7 builds | 0.9 |
| Draw on the page (seed 15072) | 3 | 1.0 (luckier than 85% of draws) |
| First draw (seed 15071), not used | 4 | 1.2 (luckier than 95% of draws) |

- **Distribution of top-10 builds:** 0 in 15% of draws, 1 in 32%, 2 in 29%, 3 in 16%, 4 or more in 7%.
- **The full-cohort number is the one to quote:** the model's top 20% (698 projects) held 123 builds vs. 61.6 at random (17.6% vs. 8.8% built), about 2×.
- **Seed rule:** after seeing the first draw was unusually lucky, we committed to the next seed before looking and kept it. Re-picking until a draw "looks right" would be hand-picking.

## Nov 20: late fusion of tabular + text ([03_merged_late_fusion.ipynb](../notebooks/03_merged_late_fusion.ipynb))

**Setup:**
- **Tabular input:** `stack._stage1`, i.e. LASSO / gradient boosting at their selected settings, fit on 2005–2014, scored out of time.
- **Text input:** `text_scores.csv` (`p_text`, `has_text`). All 3,490 projects from 2015–2018 join, and their outcomes agree.
- **Second stage:** logistic regression on the log-odds, fit leave-one-year-out.
- **Metric:** mean within-year AUC.

| Row | LASSO stage 1 | GB stage 1 |
|---|---|---|
| Tabular only | 0.632 | 0.605 |
| Text only | 0.526 | 0.526 |
| Tabular + opposition flag (re-dated) | 0.642 | 0.608 |
| **Late fusion** (tabular + text log-odds), selected | **0.643** | **0.611** |
| + coverage interaction | 0.642 | 0.608 |
| Rank average (no fitting) | 0.640 | 0.612 |

**Size of the gain:**
- **LASSO, late fusion:** +0.011 within-year, 95% CI [+0.002, +0.019], positive in 99% of bootstrap draws. It is up in every year.
- **Strictly rolling second stage** (2016–2018): +0.006 with LASSO, +0.001 with GB.
- **The flag alone gives most of it.** The reader's richer score adds about 0.001.
- **The re-dated flag gains less than `stack.py`'s old-date flag** (+0.007 vs. +0.014 pooled), so part of the earlier gain came from the footnote dates.

**Where the gain comes from:**
- **Across groups:** the fusion lowers contested-county projects as a group (5.2% built vs. 9.2% elsewhere; mean P 8.7% → 5.4%).
- **Inside that group, ranking gets slightly worse:** 0.670 → 0.654.

**At the top of the list it hurts:**
- In the four top-25 shortlists, the fusion swaps 10 projects. All 10 dropped ones come from contested counties, and 4 of them were built. None of the 10 added ones were built.
- Built projects in the shortlists: 20 → 16.
- **Reading:** use the text as a portfolio prior and a diligence flag, not as an automatic downgrade.

**Calibration** is unchanged (Brier 0.079 for both).

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
