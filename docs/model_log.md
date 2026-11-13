# Model experiment log

Every experiment, including the ones that failed, so nobody repeats them and so the slides can show what was tried.

All results are on the **5-year outcome** with **validation data only**, using two checks:
- **Rolling CV:** mean AUC over the yearly folds 2010–2014, fitting on earlier years each time.
- **Out-of-time:** train on 2005–2014, score 2015–2018. This is the harder check, because the queue changed. Neither check touches the 2019–2020 test cohort.

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
