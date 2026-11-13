# Handoff: Queue Risk AI (15.071 Project 1)

*As of Thu Nov 12, 2026 (evening). Due **Fri Nov 20**; Mon Nov 16 is a holiday.*
*Repo:* https://github.com/devault-settings/queue-risk-ai (private). Start with the [README](../README.md); this file is the short version plus the to-do list.

---

## 1. The project in three sentences

On the day a solar, wind or storage project enters a US interconnection queue, we predict **whether it will be operating within 5 years** (plus a time-to-build curve), using Berkeley Lab's public *Queued Up* data and local-opposition text from the Sabin Center's annual reports.

We serve two possible audiences, and haven't chosen yet:
- **Investors**, deciding whether to back projects early instead of waiting for the signed interconnection agreement (IA).
- **Project sponsors**, deciding where and how to site projects to improve their odds.

Every result so far is on **validation data only**. The test cohort (2019–2020 entries) hasn't been scored.

---

## 2. What's done

### Data and labels
- **Queued Up file, cleaned** ([data.py](../src/queue_risk/data.py)). A generated audit ([data_quality.md](data_quality.md)) lists every issue with its row count and the fix we applied. The biggest catch: 222 withdrawn and 133 active projects carry a completion date. A project's *status* decides whether it was built, never a date field.
- **The outcome.** "Built within 5 years of entry" ([README](../README.md#two-outcomes-and-why-we-lead-with-within-5-years)). Once 5 years have passed, every project has an answer, so nothing is censored and we can test on recent cohorts:
  - Training uses 2005–2014 entries, whose 5-year outcomes were known by 2019, the test-decision date.
  - It's limited to the six grid operators that report completion dates: CAISO, ERCOT, MISO, NYISO, PJM, SPP (66% of projects).
  - The old "ever built" outcome still runs with `--outcome ever`.
- **Entry-time features** ([features.py](../src/queue_risk/features.py)): project attributes, plus each county's and region's track record *as of the entry date* (Lecture 8: inputs built from past data only). The entry year itself is deliberately not a feature.

### Models and evaluation
- **Benchmark ladder** ([models.py](../src/queue_risk/models.py)): flat-rate rule → historical rate → logistic regression → LASSO → CART → random forest → gradient boosting, plus TabPFN when it's installed.
  - **Tuning** uses rolling time-based folds: fit on years before t, validate on t.
  - **The test set** is touched only by `--final`.
- **Validation AUC (5-year outcome):**

| Model | AUC |
|---|---|
| Flat rate | 0.500 |
| Historical rate | 0.620 |
| CART | 0.662 |
| Logistic / LASSO | 0.660 / 0.678 |
| Random forest | 0.695 |
| **Gradient boosting** | **0.701** |

- **Time-to-build curve** ([timeline.py](../src/queue_risk/timeline.py)): one model per horizon, each trained only on cohorts whose window had closed. It ranks well at 2 years (AUC 0.73) but runs short of training data at 5–7 years (about 0.55). Its probabilities come out too low because build rates drift between cohorts.

### Impact and decision tools
- **Back early vs. wait for the IA** ([strategy.py](../src/queue_risk/strategy.py)): cost per built MW, plus a break-even hit rate across 243 assumption combinations.
  - The break-even rate is **13–34%** (middle half; median 21%). Backing every project blindly gets 5.9% of capacity built within 5 years.
  - On validation, the model's top 25 projects per year got **8–56%** built, clearing the median bar in 4 of 5 years.
  - **Honest claim:** early backing becomes *competitive under some assumptions*, not reliably profitable.
- **Sponsor tools** ([sponsor.py](../src/queue_risk/sponsor.py)): a county siting screen and one-change-at-a-time what-ifs. These are associations, not causes, and only the rankings mean anything.
- **Schedule metric** ([schedule.py](../src/queue_risk/schedule.py)): before / on / after the projected date. It's mostly a measure of how often each operator revises its projected dates, so read it with care. "Active and already past the projected date" is the sturdier measure: 37–38% in MISO and PJM vs. 4–7% elsewhere.

### Local opposition (the unstructured-text part)
- **Parsing** ([sabin.py](../src/queue_risk/sabin.py)): Sabin Center editions 2022–2026 parse into **572 contested-project paragraphs** (98% matched to a county, dated back to 2000) and **about 450 restricted jurisdictions**.
- **Signal test** ([stack.py](../src/queue_risk/stack.py), validation on 2015–2018 only):
  - A county with an earlier contested project gives **about half the odds** of a 5-year build, even holding the tabular model's score fixed.
  - The ranking gain is AUC **+0.015, 95% CI [−0.003, +0.031]**: positive every year, but not conclusive.

### Infrastructure
- **Tests:** `python tests/selftest.py`, with ok/FAIL checks covering leakage, splits, labels, metrics, parsing and the data checkpoints.
- **Docs:** [course_concepts.md](course_concepts.md) maps each lecture to the code; [assumptions.md](assumptions.md) is the sourcing log for every outside number.
- **GPU work:** a [Colab notebook](../notebooks/tabpfn_colab.ipynb) runs TabPFN vs. gradient boosting, each with and without the text features (`--ablation`).

---

**Update, Nov 12 evening:** adding the point-of-interconnection (POI) track record, plus the voltage and line-vs-substation type parsed from the POI name, raised validation AUC from 0.652 to **0.701**. Census density, EIA prices, tuning, ensembles and recency weighting didn't help. Everything tried is in [model_log.md](model_log.md).

## 3. What remains, in priority order

| # | Task | Who / needs | Target |
|---|---|---|---|
| 1 | **Label 300 Sabin paragraphs** for outcome and concerns ([guidelines](labeling_guidelines.md); sheet at `data/interim/sabin_label_sheet.csv`, gitignored). Do 50 first and review | Audrey, through Mon | Mon Nov 16 |
| 2 | **Agreement check:** a teammate labels the same 50; report Cohen's kappa (≥ 0.6, or tighten the guidelines) | Teammate | Mon–Tue |
| 3 | **Train the text readers** on the labels (keyword rules vs. TF-IDF vs. frozen vs. fine-tuned DistilBERT; CPU is fine). Report macro-F1 and recall on `blocked`. Label all 572 paragraphs, then test "prior *blocked* projects" in stack.py | Code is next up; needs #1 | Tue Nov 17 |
| 4 | **Source the outside numbers** ([assumptions.md](assumptions.md)): NREL Q1-2025 cost benchmark (pre-agreement soft costs), the FERC Order 2023 deposit tiers (check 200 vs. 250 MW against the order), cost of capital (NREL ATB / Lazard). Then rerun strategy.py | Anyone | Tue |
| 5 | **TabPFN Colab run** (reminder set). Needs `GITHUB_TOKEN` and `TABPFN_TOKEN` in Colab Secrets | Audrey | Any time |
| 6 | **Agree, then run the one-time test:** `pipeline --final`, `timeline --final`, `stack --final`. Settle the outcome definition, features and models *before* running | Whole team | Tue Nov 17 |
| 7 | **SHAP driver chart** (`pip install shap`) for the best model | Anyone | Wed |
| 8 | **Impact visuals:** wasted spend per built MW vs. projects backed (rule vs. model vs. hindsight); break-even band chart; county map of opposition vs. build rates | Anyone | Wed Nov 18 |
| 9 | **Demo page** ([demo/README.md](../demo/README.md), the "Deal Desk") plus `export_data.py` with checkpoints. Optional, but it makes the video | Anyone | Wed–Thu |
| 10 | **Deliverables:** 5–10 slides, 2-minute video, final one-page README | Whole team | Thu Nov 19 |
| 11 | **Team logistics:** owners per row above; collect GitHub usernames so Audrey can add collaborators; pick the audience (investor vs. sponsor) | Whole team | Next meeting |

**Not done from the original proposal:** Census ACS and EIA price joins, and the grid-congestion features. Drop them unless someone has spare time; the opposition signal is the stronger story.

---

## 4. Decisions made (and why), so nobody re-litigates them by accident

- **Outcome = built within 5 years**, not "ever built." It avoids the censoring bias and allows a recent test. Its cost: it captures only about 46% of eventual builds, and the pitch should say so.
- **Six date-reporting operators only.** The others don't record completion dates.
- **No proposed dates, study results or agreement status as model inputs.** All are known only after entry, so using them would be leakage.
- **The text feature enters as a second stage, not as a tabular input.** Sabin events are thin before about 2010, so they can't inform the 2005–2014 training years.
- **Sabin before news.** The free GDELT service only returns recent months; historical news needs BigQuery or Media Cloud accounts.
- **Every outside number is a range, never a single figure.** Single-source figures are marked "indicative."

---

## 5. Interesting threads worth exploring (if time allows)

1. **The model breaks down on recent years.** The tabular model ranks 2017–2018 entries barely better than chance (AUC about 0.50 vs. 0.65 in rolling validation), and the build rate falls from 21% (2014 entries) to 4% (2020). Is the queue's "speculative flood" a regime change? Ideas:
   - Weight recent cohorts more heavily.
   - Use a shorter horizon (3 years) so training can include newer cohorts.
   - Treat FERC Order 2023's cluster studies as a structural break.

   This is maybe the most important finding for the pitch: *history transfers poorly, so models need constant refreshing.*
2. **Does successful opposition matter more than any opposition?** That's the payoff of the labeling. Related: do concerns differ by technology (fire safety for batteries, viewshed for wind)?
3. **Restriction dates from the footnotes.** Sabin's footnotes cite ordinance dates (for example "Ordinance No. 2024-BC-O-06 (March 19, 2024)"). Parsing them would let restrictions inform cohorts before 2021, instead of only from their first edition.
4. **Delay against the *original* projection.** The previous Queued Up edition (data through 2024) and Wayback snapshots hold older proposed dates. Comparing snapshots would measure real schedule slip, which no one currently publishes.
5. **A survival model for time to build.** A discrete-time hazard model would reuse partial histories and fix the data shortage at 5–7 years.
6. **Region-specific prices** in the strategy comparison: ERCOT development rights trade at $30–75k/MW vs. about $200k/MW for NYISO ready-to-build.
7. **Calibration.** The rankings carry over between cohorts; the probabilities don't. Recalibrating on the latest closed cohort is cheap and makes the investor numbers usable.
8. **For Project 2:** an agent that drafts a due-diligence memo per project. It would pull the model's score and drivers, the county's Sabin record and recent news.

---

## 6. How to pick it up

```bash
git clone https://github.com/devault-settings/queue-risk-ai && cd queue-risk-ai
python -m venv .venv && .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python tests/selftest.py                                 # should end "All checks passed."
python -m queue_risk.pipeline --fast                     # quick end-to-end, validation only
```

**The Sabin PDFs** aren't in the repo (gitignored). `python -m queue_risk.sabin` downloads any missing editions from the Internet Archive, then rebuilds the opposition tables and the labeling sheet. It needs `pdftotext` (poppler) on the PATH.

**Ground rules:**
- Never run `--final` alone.
- Never commit report or article text.
- Every number in the slides should come from code that's in the repo.
