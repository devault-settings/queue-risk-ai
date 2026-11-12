# Queue Risk AI

**Which clean-energy projects in the US interconnection queue will actually get built?**

15.071 The AI Edge (MIT Sloan, Fall 2026), Project 1: Predictive AI. Team: Audrey DeVault, Paul Bustarret, Charles Loriot, Sara Alafifi.

> Work in progress. This README becomes the one-page deliverable for a data scientist by Nov 20. The course concepts behind each design choice are mapped in [docs/course_concepts.md](docs/course_concepts.md).

## What it is

Only about 13% of the capacity that entered US interconnection queues from 2000 to 2020 was operating by end-2025 (Berkeley Lab, *Queued Up*). An infrastructure fund that backs early-stage projects spends development capital on many projects that later withdraw.

On the day a solar, wind or storage project enters the queue, we predict whether it will reach commercial operation. The fund backs the top **K** projects, and we measure the result as **wasted development spend per built MW**, compared against the rules of thumb investors use today.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python tests/selftest.py                  # checks leakage, splits, metrics and data checkpoints (~1 min)
python -m queue_risk.pipeline             # model selection on rolling validation (~3 min)
python -m queue_risk.pipeline --final     # one-time test-set run. Don't run until the team agrees.
```

## The pipeline

| Step | Module | Lecture |
|---|---|---|
| Load, de-duplicate, label (`built` = operational 1 / withdrawn 0) | [data.py](src/queue_risk/data.py) | — |
| Entry-time features: project attributes plus county and region track records as of the entry date | [features.py](src/queue_risk/features.py) | 2, 8 |
| Temporal split; rolling-origin validation (fit on years before t, validate on t) | [split.py](src/queue_risk/split.py) | 4, 6 |
| Benchmark ladder: flat rate → historical rate → logistic → LASSO → CART → random forest → gradient boosting | [models.py](src/queue_risk/models.py) | 3–6 |
| AUC, PR-AUC, Brier, precision@K (ties scored pro rata), calibration, bootstrap CIs | [evaluate.py](src/queue_risk/evaluate.py) | 3, 5 |
| Back the top K; wasted spend per built MW, swept over K and cost per project | [impact.py](src/queue_risk/impact.py) | 2, 3, 5 |
| News → county-year opposition score: TF-IDF vs. frozen vs. fine-tuned DistilBERT | [text.py](src/queue_risk/text.py) (interface only so far) | 10–12 |
| Interactive pitch page | [demo/](demo/README.md) (planned) | 5, 6 demos |

## Where the numbers come from

`tests/selftest.py` refuses to pass if any of these move.

| Checkpoint | Value |
|---|---|
| Rows after de-duplication | 38,066 (135 repeated IDs removed) |
| Resolved solar/wind/storage projects, 2005–2017 entries | 9,431 |
| Training cohort (2005–2014 entries) | 5,872 projects, 17.9% built |
| Test cohort (2015–2017 entries), not yet scored | 3,559 projects, 19.7% built |

**Validation results (Nov 12):** rolling folds over 2010–2014, 18 features, no text score yet.

| Model | Validation AUC |
|---|---|
| Flat rate (rule of thumb) | 0.500 |
| Historical rate, region × technology | 0.554 |
| CART | 0.588 |
| Logistic regression / LASSO | 0.631 / 0.632 |
| Random forest | 0.633 |
| **Gradient boosting** | **0.667** |

## Things worth knowing about the data

- **Right-censoring.** Withdrawals resolve faster than builds, so once active projects are dropped, recent cohorts look worse than they are. Among resolved projects, the build rate is about 24% for 2015 entries but under 3% from 2021 on. That's why the test window ends at 2017 (≥ 84% of entries resolved). The alternative fixes, a fixed-horizon label or a survival model, are open for the team to decide.
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
