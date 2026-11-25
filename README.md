# Queue Risk AI

**Which new clean-energy projects in the US interconnection queue will actually get built?**
15.071 The AI Edge (MIT Sloan, Fall 2026), Project 1 · Audrey DeVault, Paul Bustarret, Charles Loriot, Sara Alafifi


## Problem and decision
Only about 13% of capacity entering US interconnection queues gets built. On the day a solar, wind or storage project enters the queue, we estimate **P(operating within 5 years)**. An investor uses it to rank which early-stage projects to back. A developer uses the same model to screen where similar projects have done well.

## Data
- **Berkeley Lab *Queued Up*** (project-level, through end-2025). Scope: six operators that record completion dates (CAISO, ERCOT, MISO, NYISO, PJM, SPP).
- **Label:** built within 5 years of entry. Every project gets a label once its window closes, so recent cohorts are not censored.
- **Split:** train 2005–2014 (4,020 projects); out-of-time validation 2015–2018 (3,484); test 2019–2020 (3,181). Projects with 0 MW or paired with fossil units are excluded (team decision). The gap equals the horizon, so the test replays a real decision.
- **Text:** 572 contested-project paragraphs from the Sabin Center's opposition reports (2022–2026 editions), matched to counties and dated from the text.

## Approach
1. **Features known at entry:** technology, size, region and state, plus track records *as of the entry date* for the county, the region × technology and the point of interconnection (POI). Voltage and line-vs-substation are parsed from the POI name.
2. **Benchmark ladder:** flat rate → historical rate → logistic → LASSO → CART → random forest → gradient boosting → TabPFN. Tuned on rolling yearly folds, then scored out of time.
3. **Text, two stages:** fine-tuned DistilBERT reads each paragraph (did the contested project proceed?); a county's earlier opposition becomes a score; late fusion with the tabular log-odds.
4. **Decision layer:** back the top K; cost per built MW vs. waiting for the interconnection agreement, swept across sourced cost ranges.

## Results
| Model | Rolling CV AUC | Out-of-time AUC (2015–18) |
|---|---|---|
| Historical rate (region × technology) | 0.620 | 0.593 |
| **LASSO** (headline: transfers best) | 0.631 | **0.686** |
| Gradient boosting | 0.663 | 0.633 |
| TabPFN (pre-trained transformer, Colab GPU)* | 0.677 | 0.646 |
| **Test 2019–2020 (one-time, frozen):** LASSO **0.613** [0.561, 0.666]; historical rate 0.587; gradient boosting 0.530; random forest 0.454 | | |

\*TabPFN comes from a Colab run, whose library versions shift the other models slightly (LASSO 0.635 / 0.682, gradient boosting 0.666 / 0.651 there).

- **Ranking value:** LASSO's top 20% were built **about 2×** as often as average on 2015–18 (17.6% vs. 8.8%), and **1.65×** on the 2019–20 test (6.1% vs. 3.7%). Its edge over the rule of thumb on the test is small (0.613 vs. 0.587), and the flexible models collapse under drift.
- **Text reader:** fine-tuned DistilBERT, AUC **0.962** [0.92, 0.99] on 100 human-labeled paragraphs (TF-IDF 0.840, keyword rules 0.754).
- **Opposition:** on the test, adding the opposition flag lifted AUC from 0.613 to 0.623 (+0.010 [+0.001, +0.018]). In validation, projects in a county with earlier contested projects have **about 0.63× the odds** of being built (5.2% vs. 9.2%, 2015–18). But the ranking gain on top of the tabular model is near zero (+0.001 pooled AUC; Paul's fusion reports +0.011 within-year; *to reconcile*), and automatic downgrades lower the top-25 shortlists (20 → 16 built). So opposition is a **due-diligence flag**, not a model input.
- **Back early vs. wait for the agreement:** the break-even hit rate is 18–43% (median 28%) across the sourced assumptions. Today's 2× lift makes early backing *competitive under some assumptions*, not reliably profitable.

## Reproduce
```bash
pip install -r requirements.txt && pip install -e .
python tests/selftest.py               # leakage, splits, labels, metrics, data checkpoints
python -m queue_risk.pipeline          # ladder: rolling CV + out-of-time for each selected model
python -m queue_risk.sabin             # download + parse Sabin reports (needs pdftotext)
python -m queue_risk.stack             # opposition as a second stage
python demo/export_data.py && python demo/serve.py   # demo page
```
GPU (Colab): `notebooks/tabpfn_colab.ipynb` (TabPFN); `notebooks/02_paul_text_models.ipynb` (DistilBERT). The Sabin reports (public), their parse, and the paragraph labels are committed under `data/`, so no step needs a rebuild.

## Limitations
- **Leakage handled:** blank fields are recorded after the outcome in this snapshot, so the model never sees whether a field was blank. Proposed dates, study results and agreement status are never features.
- **Drift:** build rates fell from 20% (2014 entries) to 3% (2020). Rankings transfer; probabilities need recalibrating.
- **Labels:** 472 of 572 paragraph labels were LLM-drafted under written guidelines; humans labeled the 100 used for testing.
- **Not causal:** the model ranks the odds of being built; it doesn't say backing or siting a project changes them.

Repo map: `src/queue_risk/` (pipeline) · `tests/` · `notebooks/` · `demo/` · `docs/` (detailed README, data quality, assumptions, model log, handoff, final-run checklist)

Full detail, every caveat and the validation history: [docs/README_detailed.md](docs/README_detailed.md).
