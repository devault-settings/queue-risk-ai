# Queue Risk AI

**Which clean-energy projects in the US interconnection queue will actually get built?**

15.071 The AI Edge (MIT Sloan, Fall 2026), Project 1: Predictive AI. Team: Audrey DeVault, Paul Bustarret, Charles Loriot, Sara Alafifi.

> Work in progress. This README is the one-page deliverable for a data scientist. Fill in the TODO sections before Nov 20.

## Problem

Only about 13% of the capacity that entered US interconnection queues from 2000 to 2020 was operating by end-2025 (Berkeley Lab, *Queued Up*). Infrastructure investors spend development capital on projects that later withdraw. We predict, **at queue entry**, the probability that a solar, wind or storage project reaches commercial operation. We then rank projects into a deal shortlist and measure wasted development spend per built MW.

## Approach

| Component | Data | Model |
|---|---|---|
| Tabular | Berkeley Lab Queued Up (project level), Census ACS, EIA | Logistic regression, random forest, LightGBM/XGBoost; SHAP |
| Text | GDELT local news (county-tagged) | TF-IDF + LR baseline vs. fine-tuned DistilBERT/DeBERTa, scored into a county-year opposition score |

- **Label:** `built = 1` if `q_status == operational`, `0` if `withdrawn`. Active and suspended projects are excluded from training.
- **Features:** only information known at queue entry. See `LEAKAGE_COLS` in [src/queue_risk/config.py](src/queue_risk/config.py).
- **Validation:** rolling time-based CV (train through year t, validate on t+1). The held-out test cohort is used once at the end.
- **Benchmarks:**
  - historical rate by grid operator × technology
  - a flat completion rule
  - alternative models
  - the best model with and without the opposition score

## Repo layout

```
data/raw/         Berkeley Lab Queued Up file (committed, public)
data/external/    Census ACS, EIA, GDELT pulls (gitignored; rebuild with scripts)
data/interim/     intermediate tables (gitignored)
data/processed/   frozen modeling table + opposition score table (gitignored)
src/queue_risk/   shared code: config.py (paths, splits), data.py (loading), ...
notebooks/        exploration and analysis; name as NN_owner_topic.ipynb
reports/figures/  charts used in the slides
models/           saved models (gitignored)
docs/             project plan and notes
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                    # makes `import queue_risk` work everywhere
python -m queue_risk.data                           # smoke test: prints label counts by entry year
```

Fine-tune the transformer in Colab (GPU), then commit only the scored output table, never the checkpoints.

## Shared interfaces

- **County key:** `fips`, a 5-digit zero-padded string (e.g. `"04005"`). Every table uses it.
- **Opposition score table** (`data/processed/opposition_scores.csv`): `fips, year, n_articles, share_opposition`. `year` is the year the news was published. The model joins the score for `q_year - 1`.
- **Split years:** set once in `config.py` (`TRAIN_YEARS`, `TEST_YEARS`). Don't hard-code years in notebooks.

## Known issues (as of Nov 12)

- **Right-censoring.** Withdrawals resolve faster than completions, so recent cohorts look worse than they are once active projects are dropped. Among resolved clean-energy projects, the build rate is about 24% for 2015 entries, 12% for 2019 and under 3% from 2021 on. Only about 74% of 2019 entries are resolved. Options:
  - end the test window around 2017–2018
  - define the label as "operational within N years of entry"
  - use a survival model
- **Duplicate IDs.** 135 `entity|q_id` pairs repeat. Dedupe before building features.
- **Missing fields.** `fips_code` is missing for about 4.6% of rows, and `developer` / `project_name` are mostly empty.

## Results

TODO: test-set metrics with bootstrap CIs, benchmark table, impact curve.

## Data sources & compliance

- Lawrence Berkeley National Laboratory, *Queued Up* project-level data file (data through end-2025), https://emp.lbl.gov/queues
- Text data: we publish article URLs and derived scores only, never article text. No personal contact fields.
