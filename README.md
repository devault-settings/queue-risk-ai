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
| Built within 5 years of IA, IA signed 2015–2020, MW-weighted | low 44%, pooled 63%, high 80% | **Low to medium.** Only 48% of IAs have a recorded date (CAISO and ERCOT mostly; PJM 10%, SPP 0%), and the rate swings by signing year |
| Built within 5 years of entry, backing every 2015–2020 entry | 6.9% of MW | Medium. Six operators, labels fully observed |
| Median time from entry to IA | 2.6 years | Medium, same coverage caveat as the first row |
| Price of rights, early vs. late stage | $0.02–0.05/W vs. $0.07–0.20/W | **Low.** Published industry ranges, not observed trades ([Ascend Analytics](https://www.ascendanalytics.com/blog/increasing-early-stage-project-success-in-a-highly-competitive-renewable-energy-m-a-landscape)) |
| Development spend to IA; cost of capital | $0–2M per project; 0–15% a year | **Placeholders**, not sourced |

**Break-even hit rate** across all 243 combinations of these ranges:
- 25th to 75th percentile: **15% to 37%**; median 23%.
- Extremes: 4% to over 100%.
- Price difference alone, at mid prices: 11–21%.

**Where the model stands (validation only, not yet tested):** the 25 top-ranked projects in each cohort got 11–59% of their MW built within 5 years, against base rates of 4–24%. That's a 2.5–7× lift. It clears the median break-even bar in the 2012–2014 cohorts, **but not in 2010–2011, where base rates were as low as they are today.**

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
```

## The pipeline

| Step | Module | Lecture |
|---|---|---|
| Load, de-duplicate, label: within N years (default) or ever built | [data.py](src/queue_risk/data.py) | — |
| Time-to-build curve: one model per horizon, each trained only on cohorts whose window had closed | [timeline.py](src/queue_risk/timeline.py) | 6 |
| Entry-time features: project attributes plus county and region track records as of the entry date | [features.py](src/queue_risk/features.py) | 2, 8 |
| Temporal split; rolling-origin validation (fit on years before t, validate on t) | [split.py](src/queue_risk/split.py) | 4, 6 |
| Benchmark ladder: flat rate → historical rate → logistic → LASSO → CART → random forest → gradient boosting | [models.py](src/queue_risk/models.py) | 3–6 |
| AUC, PR-AUC, Brier, precision@K (ties scored pro rata), calibration, bootstrap CIs | [evaluate.py](src/queue_risk/evaluate.py) | 3, 5 |
| Back the top K; wasted spend per built MW, swept over K and cost per project | [impact.py](src/queue_risk/impact.py) | 2, 3, 5 |
| News → county-year opposition score: TF-IDF vs. frozen vs. fine-tuned DistilBERT | [text.py](src/queue_risk/text.py) (interface only so far) | 10–12 |
| Interactive pitch page | [demo/](demo/README.md) (planned) | 5, 6 demos |

## Where the numbers come from

`tests/selftest.py` refuses to pass if any of these move.

| Checkpoint | Within 5 years (default) | Ever built |
|---|---|---|
| Rows after de-duplication | 38,066 (135 repeated IDs removed) | same |
| Training cohort (2005–2014 entries) | 4,039 projects, 11.7% built | 5,872 projects, 17.9% built |
| Test cohort, not yet scored | 2019–2020: 3,188 projects, 4.4% built | 2015–2017: 3,559 projects, 19.7% built |

**Validation results (Nov 12):** rolling folds over 2010–2014, 18 features, no text score yet.

| Model | AUC, within 5 years | AUC, ever built |
|---|---|---|
| Flat rate (rule of thumb) | 0.500 | 0.500 |
| Historical rate, region × technology | 0.613 | 0.554 |
| CART | 0.612 | 0.588 |
| Logistic regression / LASSO | 0.622 / 0.627 | 0.631 / 0.632 |
| Random forest | 0.613 | 0.633 |
| **Gradient boosting** | **0.655** | **0.667** |

**Time-to-build curve, validated by deciding on 2013–2014 entries.** Each horizon trains only on cohorts whose window had closed by 2013.

| Horizon | Trained on | AUC | Predicted rate | Actual rate |
|---|---|---|---|---|
| 2 years | 2005–2011 (3,220) | 0.735 | 1.4% | 6.4% |
| 3 years | 2005–2010 (2,631) | 0.715 | 3.5% | 12.5% |
| 5 years | 2005–2008 (1,524) | 0.564 | 10.0% | 21.1% |
| 7 years | 2005–2006 (539) | 0.534 | 16.9% | 24.8% |

There are two lessons:
- **Data runs short at long horizons.** The honest training window shrinks as the horizon grows. A discrete-time hazard model, which reuses partial histories, is the fix if there's time.
- **Rates drift between cohorts.** Ranking (AUC) carries over; calibrated rates do not, so they need recalibrating on the latest closed cohort.

## Things worth knowing about the data

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
demo/             the Deal Desk page for the pitch (planned; serve.py works)
data/raw/         Berkeley Lab Queued Up file (committed, public)
data/processed/   modeling and score tables (gitignored)
notebooks/        exploration; name as NN_owner_topic.ipynb
reports/          figures and results tables
docs/             project plan; course concepts map
```

## Data sources & compliance

- Lawrence Berkeley National Laboratory, *Queued Up* project-level data file (data through end-2025), https://emp.lbl.gov/queues
- Text data: we publish article URLs and derived scores only, never article text, and no personal contact fields.
