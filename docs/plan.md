# Project plan

Planning notes carried over from the proposal phase (Nov 12, 2026). The README has the current state.

## Assignment requirements (Project 1, due Nov 20, 2026)

- Team of 3 (4 allowed). Build a **predictive AI** solution to a real business or policy problem.
- Combine **ML on tabular data** with **deep learning on unstructured data** (text, images).
- Public data only (Kaggle, UCI, APIs, government data). No confidential, proprietary or personally identifiable data, since projects may go in public portfolios.
- Answer four questions:
  1. What problem are we solving? (stakeholder, current process, opportunity)
  2. What can AI do? (data plus model design)
  3. Does it work? (rigorous evaluation against benchmarks)
  4. Does it matter? (translate into revenue, cost, risk and so on)
- Grading emphasizes:
  - a well-defined prediction task
  - rigorous model selection and validation
  - three kinds of benchmark: predictive baseline, business rule, alternative models
  - quantified impact, with visualizations
- Project 2 (later in the semester) may extend the same idea with more advanced AI.

## The project in one paragraph

Predict which US clean-energy projects (solar, wind, storage) in the grid interconnection queue will reach commercial operation, so an infrastructure investor can fund the projects most likely to be built. Context from Berkeley Lab's July 2026 update:

- 2,061 GW was waiting in queues at end-2025.
- Only 13% of capacity that entered queues from 2000 to 2020 was online by end-2025.
- 75% of that capacity withdrew.
- Getting to operation usually takes more than 5 years.

**Stakeholder:** an investment director at an infrastructure fund. A utility transmission planner is the secondary user.

## Prediction task

- **Target:** at queue entry, does the project reach operation (1) or withdraw (0)?
- **Projects still in the queue:** exclude them from training. A survival model is an optional extension.
- **Inputs (only what is known at queue entry):**
  - technology, MW, grid operator, state and county, entry year, hybrid or storage pairing
  - county saturation (count and MW of earlier same-technology projects)
  - county demographics (Census ACS)
  - state power prices (EIA)
  - the prior-year county opposition score (from the text model)
- **Leakage rule:** never use milestone dates, study results or interconnection agreement status as features.

## Data sources (all public)

| Source | Type | Use |
|---|---|---|
| Berkeley Lab "Queued Up" project-level data file (https://emp.lbl.gov/queues) | Tabular | Labels and core features; covers 7 grid operators plus about 50 other balancing areas |
| Census ACS | Tabular | County population density, income, land area |
| EIA | Tabular | State retail and wholesale prices |
| GDELT news index and linked articles | Text | Local opposition and support signal, by county-year |
| Columbia Sabin Center opposition reports | Text | Local restrictions; a check on the text model |
| Grid operator interconnection study PDFs | Text (backup) | Estimated network upgrade costs |

**Compliance:** publish article links and derived scores only, never the article text. Drop any contact or person fields.

## Model design

1. **Text model:**
   - Pull county-tagged news on solar, wind and storage projects from GDELT (2008 to 2024).
   - Hand-label 300 to 500 snippets as opposition, support or neutral. LLM-drafted labels are OK if a person checks them.
   - Fine-tune DistilBERT or DeBERTa-v3-small, and compare against a TF-IDF + logistic regression text baseline.
   - Aggregate into a county-year opposition score: the share of opposition articles, plus their count.
2. **Tabular model:**
   - LightGBM or XGBoost, compared against logistic regression and random forest.
   - SHAP values to explain what drives withdrawal.
3. **Output:** a calibrated probability per project, ranked into a deal shortlist.

## Evaluation

- **Final test set:** the most recent resolved cohort (tentatively 2017 to 2019 entries), used once at the end.
- **Tuning:** rolling time-based cross-validation (train through year t, validate on t+1). No random k-fold.
- **Uncertainty:** bootstrap 95% confidence intervals on test metrics.
- **Benchmarks:**
  - Predictive baseline: historical completion rate by grid operator × technology.
  - Business rule: a flat ~20% completion assumption; "only back projects with signed agreements."
  - Alternative models: logistic regression, random forest.
  - Ablation: best model without the opposition score, to show what the deep learning part adds.
  - Text: TF-IDF + logistic regression vs. the fine-tuned transformer.
- **Metrics:**
  - ROC-AUC and PR-AUC
  - precision among the top-k ranked projects
  - calibration (reliability plot, Brier score)
  - macro-F1 for the text model

## Impact

- **Headline metric:** wasted development spend per built MW = (sum of pre-construction spend on withdrawn projects) / (MW of built projects).
- **Pre-construction spend per project (C):** still an open assumption. Source a range and show results across that range.
- **Planned visuals:**
  - a curve of wasted spend per built MW vs. number of projects backed, with lines for random selection, the business rule and the model
  - a sensitivity table
  - a SHAP driver chart
  - a county map of opposition scores vs. withdrawal rates
- **Policy angle:** which factors drive withdrawals (grid operator process, opposition, saturation).

## Risks and fallbacks

- **Opposition score adds little signal:** spot-check about 100 counties early. Fallback: upgrade costs extracted from study PDFs.
- **Thin rural news coverage:** use state-level fallback scores plus a "no coverage" indicator.
- **Too few resolved recent outcomes:** choose the test window by outcome coverage.
- **Labeling time:** cap at 300 snippets.
- **Challenged cost assumptions:** present ranges, not a single number.

## Idea discussed but not yet decided: grid congestion features

A teammate suggested using grid congestion. The recommendation was to add it as a **feature group with an ablation**, not as a separate forecasting model.

- **Data:** the congestion component of locational marginal prices, from CAISO OASIS, ERCOT, PJM Data Miner or MISO.
- **Features:** average congestion price, share of negative-price hours, price volatility at the nearest node or zone.
- **Key nuance:** direction matters.
  - Import-constrained load pockets have higher local prices, which is good for new generation.
  - Export-constrained, generation-heavy areas have depressed prices, curtailment and upgrade costs, which is bad.
- **Practical approach:** start with ERCOT and/or CAISO and aggregate to county or pricing zone, since mapping projects to exact nodes is hard.

## STATUS AS OF NOV 12, 2026: not started, due Nov 20, team of 4

Work hasn't started yet. The team is keeping the **full scope** above: the full text model, Census and EIA joins, SHAP, map, and the survival model as a stretch goal. With 4 people working in parallel, the plan runs four workstreams at once.

### Workstreams (one owner each)

| Owner | Workstream | Done when |
|---|---|---|
| Data lead | Berkeley Lab queue file to a clean modeling table: resolved projects, entry-time features only, county saturation, Census ACS and EIA joins, leakage review | Modeling table frozen by end of Nov 14 |
| NLP lead | GDELT news pull, county matching, 300 to 500 labels, fine-tuned DistilBERT/DeBERTa vs. TF-IDF baseline, county-year opposition score | Score table delivered by end of Nov 16 |
| Modeling lead | Baselines (historical rate, business rule), logistic regression, random forest, LightGBM, rolling time-based CV, bootstrap CIs, SHAP, ablation with and without the text score; survival model as stretch | Final test-set results by Nov 18 |
| Impact and pitch lead | Source a range for pre-construction cost; wasted-spend-per-built-MW curve; sensitivity table; county map; write-up and slides | Final pitch Nov 19 |

**Optional fifth thread** for whoever finishes early: grid congestion features (see the section above), added as a separate ablation so it can't hold up the main results.

### Day-by-day

| Dates | Milestone |
|---|---|
| Nov 12 to 13 | Data downloaded; first baseline model running on a rough table; GDELT pull and labeling started; cost-assumption research started |
| Nov 14 | Modeling table frozen; labeling guidelines set and inter-labeler check done |
| Nov 15 to 16 | Tuned tabular models; text model trained and scored; opposition score merged |
| Nov 17 | Final test-set evaluation, ablations, SHAP; congestion ablation if attempted |
| Nov 18 | Impact charts and sensitivity table finished |
| Nov 19 | Write-up and slides; dry-run pitch |
| **Nov 20** | **Due** |

### Interfaces to agree on day 1 (so workstreams don't block each other)

- **County key:** 5-digit FIPS code, used by every table.
- **Opposition score table:** columns `fips, year, n_articles, share_opposition`. The modeling lead can build against a dummy version until the real one lands.
- **Train/test split years:** fixed once by the data lead after checking outcome coverage, then shared with everyone.

### What to build first in a coding session

1. Download the Berkeley Lab queue data file and inspect its columns and outcome labels.
2. Build the modeling table: resolved projects only, entry-time features only, outcome = operational vs. withdrawn, FIPS county key.
3. Fit the baselines and LightGBM with time-based validation, so a working end-to-end result exists early.

## Project 2 extension

An AI due-diligence agent: given a project ID, it pulls the model's probability and top drivers, searches local news and filings, and drafts an investment memo.

## Short description submitted for the class form

Our project predicts which US clean-energy projects in the grid interconnection queue will actually get built, helping infrastructure investors decide which early-stage projects to fund. Only about 13% of queued capacity from 2000 to 2020 came online by the end of 2025, so investors waste significant development spending on projects that withdraw. Using Berkeley Lab's public project-level queue data, we will train gradient-boosted tree models to predict each project's chance of reaching commercial operation. We will also fine-tune a transformer on local news text to score community opposition by county and add it as a predictor. We will compare our model to the simple rules of thumb investors use today and estimate how much money it could save them by avoiding projects that later withdraw.
