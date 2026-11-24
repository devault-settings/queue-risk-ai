# Handoff: Queue Risk AI (15.071 Project 1)

*As of Thu Nov 12, 2026 (evening). Due **Fri Nov 20**; Mon Nov 16 is a holiday.*
*Repo:* https://github.com/devault-settings/queue-risk-ai (private). Start with the [README](../README.md); this file is the short version plus the to-do list.

---

## ▶ Update, Tue Nov 24 (evening): THE TEST HAS BEEN RUN

- **Signed off by all four**, run from the frozen spec (odd rows excluded; TabPFN validation-only). Results committed unchanged in [reports/final_run_log.md](../reports/final_run_log.md); summary in [model_log.md](model_log.md).
- **LASSO test AUC 0.613** [0.561, 0.666] vs. the rule of thumb's 0.587. The top 20% were built 1.65× as often as average. The flexible models collapsed (gradient boosting 0.530, random forest 0.454). Opposition adds +0.010 [+0.001, +0.018]. The 2-year time-to-build AUC is 0.777.
- **Filled in:** the README test table, the one-page draft, the video script, and the deck notes (slides 5–6).
- **Still to do:**
  - Sara updates slides 4, 5, 6, 7 and 8 (see the deck notes).
  - Decide the README swap. (Sabin text in the repo: **decided Nov 24, keep it**; the reports are open access. Cite them, and swap copies for links if republication is ever questioned.)
  - Reconcile the opposition metric with Paul.
  - Ask Paul what `queue_with_costs.csv` is.
  - **Post-hoc TabPFN test** (not part of the sign-off): `notebooks/tabpfn_posthoc_test.ipynb`, logged to `reports/posthoc_log.md`.
  - The demo's export checkpoints still expect 3,490 projects (now 3,484 with odd rows excluded). Rerun `demo/export_data.py` only if the demo needs regenerating, after updating its checkpoints.
- **Do not re-run `--final`.** If a bug turns up, fix it and report both runs.

## ▶ Update, Tue Nov 24 (evening): test run frozen

- **The final test now runs only from a frozen spec** ([frozen.py](../src/queue_risk/frozen.py)), with a checklist in [FINAL_RUN.md](FINAL_RUN.md). Model selection is environment-dependent (Colab picked LASSO C = 0.01, the laptop C = 0.1), so `--final` never re-selects anything.
  - A preflight check reproduces the counts and LASSO's 0.686, and refuses to run while a team decision is pending or without `--signoff`.
  - Dry run: `python -m queue_risk.final`.
- **TabPFN is done** (Colab run 2): best rolling CV (0.677), but it transfers worst to 2017–18. The opposition effect holds with every first-stage model ([model_log.md](model_log.md)).
- **Paul committed** the Sabin PDFs, the parsed paragraphs, the labels and `text_scores.csv`. `sabin.py` now keeps the committed parse unless run with `--rebuild`, because the labels are keyed to it and other machines' PDF tools produce slightly different text (curly vs. straight quotes).

## ▶ Update, Tue Nov 24 (TabPFN, costs, odd rows)

- **TabPFN, first Colab run** (`notebooks/tabpfn_colab_ran_1.ipynb`, validation only). Same environment, rolling CV / out-of-time 2015–18:
  - TabPFN **0.677** / 0.646
  - Gradient boosting 0.666 / 0.651
  - LASSO 0.635 / **0.682**
  - Historical rate 0.620 / 0.593

  TabPFN has the best rolling CV; LASSO still transfers best. The stacking half failed on a Colab install error (fixed in the notebook); rerun pending. On Colab, LASSO and gradient boosting differ slightly from the laptop numbers (library versions), so compare within one environment.
- **Spend lost per withdrawn project** is now built from Sara's sources: **$2.40M / $4.95M / $8.65M** per 100 MW project (price of rights + spend toward the IA). It's an upper bound, and it replaces the $1M / $3M / $10M placeholder ([assumptions.md](assumptions.md)).
- **Odd rows (Charles's question), sized:** in the modeling table through 2018 there are only **3** projects with 0 or missing capacity (all still active) and **12** solar/wind projects paired with gas (11 withdrawn). Dropping both changes nothing that matters: LASSO out-of-time AUC 0.686 → 0.686–0.687; top 20% still 123 built of about 697. They do occupy 4 of the top-20% slots, which is why one shows up in the demo. **Suggested rule:** exclude 0-capacity and fossil-paired rows as out of scope. It's a one-line change in `data.in_scope`, for the team to approve before the final test.
- **Time-to-build table refreshed** (README) with the current features: 2-year AUC 0.745, falling to 0.517 at 7 years.

## ▶ Update, Tue Nov 24 (cost sourcing merged)

- **Sara's `sara/cost-sourcing` branch is merged into `main`.**
  - Development spend before the agreement: **$0.4M / $1.45M / $3.65M** per 100 MW project (FERC Order 2023 study deposits, NREL permitting, land).
  - Cost of capital: **6% / 10% / 15%** (NREL ATB 2025).
  - Price ranges are unchanged, now cross-checked against SEC filings.
  - Sources and page references are in [assumptions.md](assumptions.md).
- **Effect:** the break-even hit rate for backing early rose to **18–43%** (median 28%), from 13–34% (median 21%). At mid assumptions it's 17–35%. `strategy.py` now computes its "price alone" line directly, since the sourced ranges no longer include zero.
- **Still a placeholder:** `DEV_COST_RANGE_USD` (spend lost per withdrawn project, used by `impact.py`).

## ▶ Update, Fri Nov 20 (demo page)

- **Charles built the demo page** for the video, a simpler version of the Deal Desk ([demo/README.md](../demo/README.md)). Run `python demo/export_data.py`, then `python demo/serve.py`.
  - 50 random projects from 2015–2018, scored by the out-of-time LASSO, with three states: entry-day list → **Run the model** → **Show what actually happened**.
  - **Tabular model only:** no opposition flag or Sabin text on the page.
  - `export_data.py` refuses to write unless LASSO's out-of-time AUC is still 0.686. It never loads or scores the 2019–2020 test cohort. `--final` has still not been run.
- **Numbers on the page:**
  - Header: AUC 0.69 [95% CI 0.66–0.71] on 2015–18 entries; base rate 8.8% (308 of 3,490 built within 5 years).
  - This sample: 3 builds in the model's top 10, against 1.0 expected at random and 5 with perfect hindsight.
  - Full cohort: the model's top 20% (698 projects) held 123 builds, against 61.6 at random, so **about 2×**. This is the claim to quote.
- **The sample is luckier than typical, and the page says so.** Across 20,000 random 50-project draws, the top 10 averaged 1.7 builds vs. 0.9 at random. This draw is luckier than 85% of them. The first draw (seed 15071) was luckier than 95%, so we moved to the next seed, committed to before looking (details in the demo README). **Don't re-pick the seed.**
- **For the video:** say "about twice as often," not "three times." Don't call AUC "accuracy."
- **Notebook 01's saved outputs are stale.** They predate the Nov 15 leak fix and show LASSO at 0.647 out of time. The current code gives 0.686. Rerun the notebook before quoting anything from it.
- **Data question for the team:** some in-scope projects look odd. AB1-042 (PJM battery) is listed at 0 MW and ranks 2nd in the demo; R101 (MISO) is a 4.6 MW "Wind+Gas" project and ranks 1st. Both are as listed in Queued Up. Decide whether 0 MW rows should be filtered in `data.in_scope` (that would change the modeling table, so rerun the checkpoints).

## ▶ Update, Sun Nov 15

- **Charles added** a walkthrough notebook (`notebooks/01_charles_tabular_models.ipynb`), which runs the existing pipeline in the lecture style, plus six slide-ready figures in `reports/figures/`. He also fixed a `sabin.py` parsing difference that shows up with newer PDF tools; the output is identical on Audrey's machine. Nothing existing was removed, and the test cohort is still unscored.
- **His SHAP check exposed a leak:** blank fields (connection point, county, service type) are recorded after the outcome. It's fixed. The validation and out-of-time numbers are now reported for the same selected models. See [model_log.md](model_log.md#nov-15-blank-field-leakage-fixed-consistent-out-of-time-reporting).
- **Current numbers:**
  - Rolling validation: gradient boosting 0.663 (best), LASSO 0.631.
  - Out-of-time 2015–2018: **LASSO 0.686 (best)**, gradient boosting 0.633.
  - **Recommendation:** LASSO as the headline model.
  - The opposition flag: odds about ×0.55, small ranking gain (+0.005, not conclusive).
- **Small follow-up for the notebook:** cell 67 pools build rates over all years, including 2019–2020. Limit it to the training years. Its SHAP section should be rerun after this fix, since the missing-value flags it found are gone.

## ▶ Pick up here (Audrey, Fri Nov 13)

**State at end of Thu Nov 12:**
- Everything is pushed; the last commit is "Add point-of-interconnection features; log CPU model experiments."
- `python tests/selftest.py` passes.
- *Superseded Nov 15; see the update below.* Validation AUC was 0.701 with gradient boosting and 0.626 out-of-time on 2015–2018.
- The 2019–2020 test cohort has **not** been scored.

**First things tomorrow:**
1. **TabPFN in Colab.** The reminder set in the Claude session (9:47am) only fires if that session is still open, so this line is the backup. Decide first how opposition enters the comparison: (a) export the Sabin county-year signal to `data/processed/opposition_scores.csv` so the notebook's `--ablation` compares with and without it, or (b) add TabPFN as an alternative first stage in `stack.py` (more consistent with how the text signal is tested now). Then: accept the license at ux.priorlabs.ai → add `GITHUB_TOKEN` and `TABPFN_TOKEN` to Colab Secrets → T4 GPU → run `notebooks/tabpfn_colab.ipynb`.
2. **Start labeling.** Open `data/interim/sabin_label_sheet.csv` in Excel and label the first 50 following [labeling_guidelines.md](labeling_guidelines.md), then review them with Claude before doing the other 250. Target: all 300 by Mon Nov 16.
3. **Optional, CPU:** try the 3-year-horizon idea from [model_log.md](model_log.md), which lets training include cohorts closer to the test years.

**Files that live only on this laptop** (gitignored; all can be regenerated):

| Path | What | How to regenerate |
|---|---|---|
| `data/external/sabin/` | Sabin report PDFs and extracted text | `python -m queue_risk.sabin` (downloads from the Internet Archive) |
| `data/interim/sabin_*.csv` | Parsed paragraphs, restrictions, **the labeling sheet** | Same command. It never overwrites an existing label sheet, but the file isn't in git, so **back it up** (OneDrive or similar) once you start labeling |
| `data/external/census/` | 2010 county gazetteer | Downloaded automatically when the Census feature switch is used |
| `data/external/eia/` | State electricity prices | Not used (dropped in the experiments); safe to delete |
| `reports/results_*.csv` | Pipeline result tables | `python -m queue_risk.pipeline` |

**Still open for the team** (sections 3 and 4 below): owners for each task, GitHub usernames so Audrey can add collaborators, investor vs. sponsor framing, sourcing the cost ranges, and agreeing before anyone runs `--final`.

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
| CART | 0.593 |
| Logistic / LASSO | 0.615 / 0.631 (out-of-time 0.641 / **0.686**) |
| Random forest | 0.629 |
| **Gradient boosting** | **0.663** (out-of-time 0.633) |

- **Time-to-build curve** ([timeline.py](../src/queue_risk/timeline.py)): one model per horizon, each trained only on cohorts whose window had closed. It ranks well at 2 years (AUC 0.73) but runs short of training data at 5–7 years (about 0.55). Its probabilities come out too low because build rates drift between cohorts.

### Impact and decision tools
- **Back early vs. wait for the IA** ([strategy.py](../src/queue_risk/strategy.py)): cost per built MW, plus a break-even hit rate across 243 assumption combinations.
  - The break-even rate is **13–34%** (middle half; median 21%). Backing every project blindly gets 5.9% of capacity built within 5 years.
  - On validation, gradient boosting's top 25 projects per year got **17–84%** built, clearing the median bar in 4 of 5 years. LASSO's got 7–41%, clearing it in 3 of 5.
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
| 9 | ~~**Demo page** ([demo/README.md](../demo/README.md), the "Deal Desk") plus `export_data.py` with checkpoints~~ **Done Nov 20** (simple version: no map, cost dial or SHAP) | Charles | Wed–Thu |
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
