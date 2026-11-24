# Queue Risk AI

**Which clean-energy projects in the US interconnection queue will actually get built?**

15.071 The AI Edge (MIT Sloan, Fall 2026), Project 1: Predictive AI. Team: Audrey DeVault, Paul Bustarret, Charles Loriot, Sara Alafifi.

> Work in progress. This README becomes the one-page deliverable for a data scientist by Nov 20. The course concepts behind each design choice are mapped in [docs/course_concepts.md](docs/course_concepts.md).

## What it is

Only about 13% of the capacity that entered US interconnection queues from 2000 to 2020 was operating by end-2025 (Berkeley Lab, *Queued Up*). An infrastructure fund that backs early-stage projects spends development capital on many projects that later withdraw.

On the day a solar, wind or storage project enters the queue, we predict **whether it will be operating within 5 years**, plus a time-to-build curve (the chance of operating within 2, 3, 4, 5 and 7 years). The fund backs the top **K** projects, and we measure the result as **wasted development spend per built MW**, compared against the rules of thumb investors use today.

## Two audiences (both kept for now)

**1. Investors buying development rights: back early, or wait for the agreement?** ([strategy.py](src/queue_risk/strategy.py))

Today's rule is to ignore queue projects until they sign an interconnection agreement (IA). Waiting is safer but pays a late-stage price. Backing at entry is cheaper, but many more projects fail. The yardstick is **cost per MW that actually gets built**, and backing early wins when the model's hit rate clears a break-even bar.

What we can measure, and how confident we are in each input:

| Input | Value | Confidence |
|---|---|---|
| Built within 5 years of IA, IA signed 2015–2020, MW-weighted | low 36%, pooled 58%, high 74% | **Low to medium.** Only 48% of IAs have a recorded date (CAISO and ERCOT mostly; PJM 10%, SPP 0%), and the rate swings by signing year |
| Built within 5 years of entry, backing every 2015–2020 entry | 5.9% of MW | Medium. Six operators, labels fully observed |
| Median time from entry to IA | 2.6 years | Medium, same coverage caveat as the first row |
| Price of rights, early vs. late stage | $0.02–0.05/W vs. $0.07–0.20/W | **Low to medium.** Published industry ranges ([Ascend Analytics](https://www.ascendanalytics.com/blog/increasing-early-stage-project-success-in-a-highly-competitive-renewable-energy-m-a-landscape)), consistent with SEC filings: an $0.008/W greenfield floor, and late-stage sales at $0.08 and about $0.145/W (queue stage to confirm) |
| Development spend to IA, per 100 MW project | $0.4M / $1.45M / $3.65M | **Medium.** Studies from FERC Order 2023 (verified against the order), permitting from NREL benchmarks; land low and mid are assumptions ([docs/assumptions.md](docs/assumptions.md)) |
| Cost of capital | 6% / 10% / 15% a year | **Medium.** NREL ATB 2025 (renewables WACC; construction equity return); the 15% high is an assumption for early-stage risk |

**Break-even hit rate** across all 243 combinations of these ranges:
- 25th to 75th percentile: **18% to 43%**; median 28%. Sourcing the costs raised the bar from 13–34%, median 21%, under the earlier placeholders.
- Extremes: 5% to over 100%.
- At mid-range assumptions (prices, spend and cost of capital): **17–35%**, depending on the post-IA build rate. Price difference alone: 9–19%.

**Where the model stands (validation only, not yet tested):** the 25 projects gradient boosting ranks highest in each validation year got 17–84% of their MW built within 5 years, against base rates of 4–23%. That clears the median break-even bar (28%) in three of five years (2012–2014) but not in 2010–2011, when base rates were as low as today's. LASSO's top 25 got 7–41% and clear it in two of five. On the 2015–2018 cohorts, the demo's full-cohort check finds LASSO's top 20% built 17.6% of the time, against 8.8% at random: about twice the base rate, but below the median bar. (That figure counts projects; the bar is weighted by MW, so the comparison is approximate.)

The honest claim for now: the model makes early backing *competitive* under some assumptions, not reliably profitable. The 2019–2020 test run will say which side of the band it lands on.

Caveats:
- The comparison is conservative for backing early: built within 5 years of *entry* is stricter than within 5 years of the *IA*, which can be up to ~7.6 years after entry.
- It ignores what a built project is worth beyond the late-stage price.

**2. Project sponsors: what choices are associated with better odds?** ([sponsor.py](src/queue_risk/sponsor.py))

The 5-year model, refit on all closed cohorts (2005–2020), scores a *planned* project:
- `siting_screen()` places it in every county of a region with queue history and ranks the counties.
- `what_if()` changes one choice at a time (size, battery pairing, service type).

**These are associations, not causal effects.** A county where past projects got built may differ in transmission headroom, land or permitting in ways the model never sees, and moving a project there doesn't bring those conditions with it. Two more limits:
- **Only rankings mean anything.** The probability levels drift between cohorts and aren't calibrated.
- **The size curve is lumpy.** The model scores a 200 MW project higher than both 50 MW and 400 MW. That's a tree artifact, not an optimal size.

Use it to decide where to look harder, not where to build.

## Local opposition, from text (Sabin Center reports)

The Sabin Center's annual *Opposition to Renewable Energy Facilities in the United States* (editions 2022–2026, cumulative) is parsed by [sabin.py](src/queue_risk/sabin.py). The PDFs come from the Internet Archive, because the archive's own links sit behind a bot check; they are saved to `data/external/sabin/` and gitignored.

- **Contested projects:** 572 narrative paragraphs in 48 states, 98% matched to a county. Each is dated by the earliest year in its text, which reaches back to 2000.
- **Restrictions:** about 450 jurisdictions in 42 states (429 counties, 9 municipalities, 13 state laws). They are dated only by the first edition that lists them (2021 onward), so they can't inform older cohorts.

**Does it add signal?** ([stack.py](src/queue_risk/stack.py), validation on 2015–2018 entries only.) The test is two-stage, as in Lecture 7: the frozen tabular model's score, plus the opposition flag, fit and scored leave-one-year-out.
- **Effect size** (with Paul's corrected paragraph dates, Nov 24): a county with an earlier contested project has **about 0.63 times the odds** of a 5-year build at the same tabular score (LASSO first stage; ×0.67 with gradient boosting).
- **Ranking gain:** essentially none. AUC +0.001, 95% CI [−0.005, +0.008] with LASSO; +0.000 with gradient boosting. With the old footnote-inflated dates it was +0.005, so part of that came from the dating. Paul's late fusion reports +0.011 on a different metric; see [docs/model_log.md](docs/model_log.md) before quoting either.
- **Caveat on the list itself:** Sabin compiles its list after the fact, so a 2015 controversy is in it partly because researchers looked back later. That was knowable in 2015, but it isn't a record collected in 2015.
- **Drift, revisited:** the earlier "2017–2018 breakdown" (AUC about 0.55) was largely the blank-field leak, which held in old resolved records but not newer ones. Without it, LASSO scores 0.668 and 0.737 on those years.

**The NLP step (needs labels):** label 300 paragraphs for outcome and concerns ([guidelines](docs/labeling_guidelines.md)), then compare keyword rules, TF-IDF, frozen DistilBERT and fine-tuned DistilBERT, all on a CPU. The question is whether *successful* opposition (blocked projects) carries more signal than any opposition.

## On schedule? Before, on, or after the projected date

`python -m queue_risk.schedule`. This is a descriptive metric for both audiences, not a model feature, because the proposed date gets revised after entry.

| | CAISO | ERCOT | PJM |
|---|---|---|---|
| Completed projects measured | 166 | 293 | 495 |
| Early / on time (±90 days) / late | 2% / 2% / 96% | 13% / 84% / 2% | 22% / 62% / 16% |
| Median slip | 2.9 years | 0 | 0 |

**Read this table with care.** The file stores only the *latest* proposed date, so it mostly measures how often each operator revises that date, not real differences in delay:
- ERCOT republishes projected dates monthly, so they converge on the actual date.
- CAISO's dates look like the original filings.
- In MISO (100%) and SPP (88%), the proposed date was overwritten with the actual one, so they are excluded.

Every figure is a lower bound on real delay.

The sturdier measure is **active projects already past their proposed date** at end-2025: 37–38% in MISO and PJM, 4–7% in CAISO, ERCOT and SPP. That's also a lower bound, because postponed projects don't count.

Measuring delay against the *original* projection needs earlier snapshots of the queue: the previous Queued Up edition, older editions via the Wayback Machine, or the operators' monthly queue reports.

## Two outcomes, and why we lead with "within 5 years"

- **Within N years** (default). Every project is labeled once N years have passed since it entered: built in time = 1; still active, suspended or withdrawn = 0. Nothing is censored, so we can test on the most recent cohorts whose window has closed (2019–2020 entries). Training uses 2005–2014 entries, whose 5-year outcomes were known by 2019, so a fund could really have built this model at decision time.
- **Ever built** (`--outcome ever`). Uses resolved projects only. It is right-censored: withdrawals resolve faster than builds, so recent cohorts look worse than they are. Its test window has to stop at 2015–2017.

"Within N years" needs a completion date, so it covers the six operators that report one for 96% of built projects: CAISO, ERCOT, MISO, NYISO, PJM and SPP, together 66% of queue projects. ISO-NE reports no completion dates, and the non-ISO West and Southeast report few.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python tests/selftest.py                  # checks leakage, splits, metrics and data checkpoints (~1 min)
python -m queue_risk.pipeline             # model selection on rolling validation (~3 min)
python -m queue_risk.pipeline --final     # one-time test-set run. Don't run until the team agrees.
python -m queue_risk.timeline             # time-to-build curve, validated on 2013-2014 entries
python -m queue_risk.strategy             # investors: back early vs. wait for the IA
python -m queue_risk.sponsor --region ERCOT --tech Solar --mw 200   # sponsors: siting screen, what-if
python -m queue_risk.schedule             # before / on / after the projected online date
python -m queue_risk.quality              # data-quality audit -> docs/data_quality.md
python -m queue_risk.pipeline --models "Gradient,TabPFN" --ablation   # with vs. without text features
```

## The pipeline

| Step | Module | Lecture |
|---|---|---|
| Load, de-duplicate, label: within N years (default) or ever built | [data.py](src/queue_risk/data.py) | — |
| Time-to-build curve: one model per horizon, each trained only on cohorts whose window had closed | [timeline.py](src/queue_risk/timeline.py) | 6 |
| Entry-time features: project attributes; county, region and **point-of-interconnection** track records as of entry; voltage and line-tap vs. substation parsed from the POI name | [features.py](src/queue_risk/features.py) | 2, 8 |
| Temporal split; rolling-origin validation (fit on years before t, validate on t) | [split.py](src/queue_risk/split.py) | 4, 6 |
| Benchmark ladder: flat rate → historical rate → logistic → LASSO → CART → random forest → gradient boosting | [models.py](src/queue_risk/models.py) | 3–6 |
| AUC, PR-AUC, Brier, precision@K (ties scored pro rata), calibration, bootstrap CIs | [evaluate.py](src/queue_risk/evaluate.py) | 3, 5 |
| Back the top K; wasted spend per built MW, swept over K and cost per project | [impact.py](src/queue_risk/impact.py) | 2, 3, 5 |
| News → county-year opposition score: TF-IDF vs. frozen vs. fine-tuned DistilBERT | [text.py](src/queue_risk/text.py) (interface only so far) | 10–12 |
| TabPFN, a pre-trained tabular transformer (optional; GPU; [Colab notebook](notebooks/tabpfn_colab.ipynb)) | [models.py](src/queue_risk/models.py) | 8, 12 |
| Ablation: every model with and without the news-opposition features | `--ablation` in [pipeline.py](src/queue_risk/pipeline.py) | 7, 12 |
| Schedule: before, on, or after the projected online date | [schedule.py](src/queue_risk/schedule.py) | — |
| Interactive pitch page | [demo/](demo/README.md) (simple version built Nov 20) | 5, 6 demos |

## Where the numbers come from

`tests/selftest.py` refuses to pass if any of these move.

| Checkpoint | Within 5 years (default) | Ever built |
|---|---|---|
| Rows after de-duplication | 38,066 (135 repeated IDs removed) | same |
| Training cohort (2005–2014 entries) | 4,029 projects, 11.1% built | 5,862 projects, 17.7% built |
| Test cohort, not yet scored | 2019–2020: 3,184 projects, 3.7% built | 2015–2017: 3,557 projects, 19.7% built |

**Test results (one-time run, Nov 24, signed off by all four; [log](reports/final_run_log.md)):** the 2019–2020 cohort, 3,181 projects, 3.7% built within 5 years.

| Model | Test AUC (2019–20) [95% CI] | Out-of-time AUC (2015–18) |
|---|---|---|
| **LASSO** (headline) | **0.613** [0.561, 0.666] | 0.686 |
| Historical rate (rule of thumb) | 0.587 [0.539, 0.636] | 0.593 |
| Logistic regression | 0.545 [0.496, 0.597] | 0.641 |
| Gradient boosting | 0.530 [0.472, 0.588] | 0.633 |
| Flat rate | 0.500 | 0.500 |
| CART | 0.475 [0.431, 0.521] | 0.610 |
| Random forest | 0.454 [0.400, 0.511] | 0.588 |

- **LASSO's top 20% were built 1.65× as often as average** (6.1% vs. 3.7%), down from 2× on 2015–18.
- **Its edge over the rule of thumb is small** (intervals overlap), and the flexible models collapse under drift.
- **Opposition as a second stage:** test AUC 0.613 → 0.623 (+0.010 [+0.001, +0.018]).
- **Time to build:** 2-year AUC 0.777, 3-year 0.761.
- **Probabilities run about 2× high,** so recalibrate before quoting them.

**Validation results (Nov 15, after the blank-field leakage fix):** 25 features, including the point-of-interconnection track record and voltage; no missing-value flags. Two numbers per model, both for the *same* selected configuration:
- **Rolling CV:** fit on earlier years, validate on each of 2010–2014.
- **Out-of-time:** fit on 2005–2014, score 2015–2018. These cohorts sit between training and test, so this check is the closest to how the model would be used.

The "ever built" column predates the data-quality fixes and the new features. Every experiment is in [docs/model_log.md](docs/model_log.md).

| Model | Within 5 years: rolling CV | Within 5 years: out-of-time | Ever built: rolling CV |
|---|---|---|---|
| Flat rate (rule of thumb) | 0.500 | 0.500 | 0.500 |
| Historical rate, region × technology | 0.620 | 0.593 | 0.554 |
| CART | 0.593 | 0.610 | 0.588 |
| Logistic regression | 0.615 | 0.641 | 0.631 |
| **LASSO** (C = 0.1) | 0.631 | **0.686** | 0.632 |
| Random forest | 0.629 | 0.588 | 0.633 |
| **Gradient boosting** | **0.663** | 0.633 | 0.667 |

**Which model?** Gradient boosting wins on rolling CV, but **LASSO transfers best to the later cohorts**: 0.686 overall, and 0.668 and 0.737 in 2017 and 2018. That's Lecture 4's case for regularization under a shifting environment, and it is the closer stand-in for the 2019–2020 test. Recommendation: LASSO as the headline model, gradient boosting as the comparison.

**Time-to-build curve, validated by deciding on 2013–2014 entries** (rerun Nov 24 with the current features and the leakage fix). Each horizon trains only on cohorts whose window had closed by 2013.

| Horizon | Trained on | AUC | Predicted rate | Actual rate |
|---|---|---|---|---|
| 2 years | 2005–2011 (3,215) | 0.745 | 1.1% | 5.5% |
| 3 years | 2005–2010 (2,626) | 0.629 | 3.7% | 11.5% |
| 5 years | 2005–2008 (1,523) | 0.553 | 9.3% | 19.6% |
| 7 years | 2005–2006 (539) | 0.517 | 19.2% | 22.7% |

There are two lessons:
- **Data runs short at long horizons.** The honest training window shrinks as the horizon grows. A discrete-time hazard model, which reuses partial histories, is the fix if there's time.
- **Rates drift between cohorts.** Ranking (AUC) carries over; calibrated rates do not, so they need recalibrating on the latest closed cohort.

## Things worth knowing about the data

- **Blank fields leak the outcome (found Nov 15 by Charles Loriot's SHAP check).** In this snapshot, resolved projects have sparser records than active ones. In SPP, the connection-point name is missing *only* for withdrawn projects; in ERCOT, the county is missing for 0.6% of active projects but 36–59% of resolved ones. The model therefore never sees whether a field was blank: no missing-value flags, ERCOT's missing state filled as Texas, and the service-type feature dropped. Validation scores fell (gradient boosting 0.701 → 0.663) but out-of-time scores rose (LASSO 0.647 → 0.686), which is the signature of a leak that doesn't transfer.
- **Residual risk, untested:** capacity and battery pairing come from the latest record, and projects are sometimes resized or hybridized as they advance.

The full audit, with every issue, its row count and what we do about it, is generated from the file by `python -m queue_risk.quality` and written to [docs/data_quality.md](docs/data_quality.md). The essentials:

- **Status decides whether a project was built, never a date field.** 222 withdrawn and 133 active or suspended projects carry an operation date.
- **Excluded rows.** 97 requests modify an existing plant rather than propose a new one, and 18 rows have an operation date before their queue date.
- **Future dates.** 52 dates fall after the file's end-2025 as-of date; they are treated as missing.

- **Right-censoring (the "ever built" outcome only).** Withdrawals resolve faster than builds, so once active projects are dropped, recent cohorts look worse than they are. The fixed-horizon outcome avoids this.
- **A real decline, not an artifact.** Even with fully observed 5-year labels, the build rate falls from 21% for 2014 entries to 3.9% for 2020 entries. The queue is flooding with speculative requests. Ranking still works under this drift, but probabilities need recalibrating.
- **Missing dates.**
  - 27% of built projects have no completion date, concentrated outside the six operators above. Inside them, 4% are missing; those take the developer's proposed online date, which gives the same 5-year label 90% of the time. It's used for the label only, never as a feature.
  - 34% of withdrawn projects have no withdrawal date. The track-record features treat an unknown date as "not known at entry," which undercounts past withdrawals.
- **Duplicates and dirt.** 135 `entity|q_id` pairs repeat, mostly 1997–2002 gas plants re-listed without county. The more complete row of each pair is kept. One FIPS code is a garbled 9-digit value and is treated as missing.
- **The "signed agreement" rule is a later decision.** Developers often ignore projects until they sign an interconnection agreement, but that information arrives years after entry. It's a different decision point, not a fair head-to-head benchmark.
- **This is not a causal claim.** The model ranks how likely a project is to be built. It doesn't say that backing a project makes it more likely to be built.

## Shared interfaces

- **County key:** `fips`, a 5-digit string.
- **Opposition score table:** `data/processed/opposition_scores.csv` with columns `fips, year, n_articles, share_opposition`. A project entering in year t gets its county's year t−1 score. The pipeline picks it up automatically when the file exists.
- **Split years, K, cost assumptions:** [config.py](src/queue_risk/config.py) only. The cost range there is a **placeholder** until it is sourced.

## Repo layout

```
src/queue_risk/   the pipeline (above)
tests/            selftest.py: ok/FAIL checks, exits non-zero on failure
demo/             the demo page for the video (export_data.py, then serve.py)
data/raw/         Berkeley Lab Queued Up file (committed, public)
data/processed/   modeling and score tables (gitignored)
notebooks/        exploration; name as NN_owner_topic.ipynb
reports/          figures and results tables
docs/             project plan; course concepts map
```

## Data sources & compliance

- Lawrence Berkeley National Laboratory, *Queued Up* project-level data file (data through end-2025), https://emp.lbl.gov/queues
- Sabin Center reports (Columbia Law School, open access): the PDFs, parsed paragraphs and labels are versioned in the repo (team decision, Nov 24: freely available without a login). Cite the reports; if republication is ever questioned, replace the copies with links. No personal contact fields.
