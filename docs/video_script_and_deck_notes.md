# Two-minute video script, and fact-check notes for the deck

*Draft, Nov 24. The script follows Sara's deck ("Queue Risk AI", 10 slides, Google Slides) and is addressed to an infrastructure fund. Test-set numbers are pending and marked [TEST]. Every other number is traced in the notes below.*

## Video script (~290 words, about 2 minutes at a steady pace)

| Time | On screen | Narration |
|---|---|---|
| 0:00–0:15 | Title slide | Infrastructure funds want into clean energy early. But of all the capacity that joined US grid queues from 2000 to 2020, only 13% was running by the end of 2025. |
| 0:15–0:35 | Slides 2–3 | So funds wait for the interconnection agreement, about two and a half years in, and then pay 7 to 20 cents a watt for the rights. On day one, the same rights cost 2 to 5 cents. Backing early is cheaper, *if* you can pick the projects that will actually get built. |
| 0:35–0:55 | Slide 4 | Our model scores every new project the day it joins the queue, using only public data: its size and technology, and how earlier projects fared in the same county and at the same grid connection point. We trained it on 2005–2014 projects and tested it on 2015–2018 projects it had never seen. |
| 0:55–1:20 | Demo page: list → Run the model → Show what happened | Here are 50 projects from 2015 to 2018, in the order they entered. We run the model, then reveal what happened. Across all 3,490 projects, the model's top fifth got built **about twice as often** as average: 17.6% against 8.8%. Against the industry's rule of thumb, it picks the built project in 69 pairs out of 100, versus 59. [TEST: and on 2019–2020 projects, ___.] |
| 1:20–1:40 | Slides 8–9 | Ranked by expected megawatts, the model's top picks waste far less development money per megawatt built. Early backing beats waiting when about 28% of picks get built. The model cleared that in 2015 and 2016, but not in 2017 and 2018, when speculative requests flooded the queue. |
| 1:40–1:55 | Slide 7 | We also read local-opposition reports with an AI model. It's a useful warning sign to check in due diligence, but not a reason to drop a project. |
| 1:55–2:10 | Slide 10 | Our recommendation: score every new entrant, send the top fifth to due diligence, back early only when the picks clear the bar, and retrain every year. |

**Delivery notes:**
- Say "about twice as often," not "three times." Don't call AUC "accuracy." The "69 out of 100 pairs" wording is the AUC, said plainly.
- The demo's 50-project sample is luckier than typical (it says so on the page). Quote the full-cohort 2×, not the sample.
- Fill [TEST] only from `reports/final_run_log.md`. If the test result is weaker, say so; the script holds either way.

## Fact-check notes for Sara's deck (against the repo, Nov 24)

| Slide | Claim | Status |
|---|---|---|
| 2 | 13%; median 2.6 years to agreement; 7–20¢/W | ✓ Berkeley Lab; `strategy.py`; `docs/assumptions.md` |
| 3 | 2–5¢ vs. 7–20¢; 5.9% of MW built backing everything | ✓ `strategy.py` |
| 4 | "38,066 projects … across 6 regional grid operators" | ✗ **Fix.** 38,066 is the whole national file (all regions). The model uses about 10,700 labeled projects from the 6 operators (4,029 train / 3,490 validation / 3,184 test). |
| 4 | "25 facts known on day one"; "515 paragraphs" | ✓ 25 features; 515 resolved paragraphs |
| 5 | 50 / 59 / 69 | ✓ flat rate / historical rate 0.593 / LASSO 0.686 out of time. Test placeholder still to fill |
| 6 | 17.6% (123 of 698) vs. 8.8% (308 of 3,490) | ✓ demo export, LASSO out of time |
| 7 | 89% (80 of 90); 5.2% vs. 9.2%; 4 built projects pushed out | ✓ Paul's notebook 02 and 03 |
| 7 | "Adding opposition improved the overall ranking only slightly" | ⚠ **Soften.** With Paul's corrected paragraph dates, the gain on top of the tabular model is about zero (+0.001 pooled AUC); his fusion metric gives +0.011. Suggest: "Opposition marks lower-odds counties, but didn't improve the overall ranking", until the two numbers are reconciled |
| 8 | "About 85% less money per MW built"; "3 to 6 times as often as random" | ✓ Notebook 01 cell 86: LASSO × MW wastes 0.14× random out of time; cell 79: 56/37/24/23% vs. 20/10/8/4%. ⚠ **Add context:** the historical-rate rule ranked by MW already reaches 0.19× (81% less). The model's edge over the rule of thumb is 0.14 vs. 0.19, and most of the saving comes from ranking by expected MW. Worth one line |
| 9 | "Pays off when 28% of picks get built, as in 2015–16" | ✓ Break-even median 28% (`strategy.py`, sourced costs); LASSO top 25 hit 56% and 37% in 2015–16, 24% and 23% in 2017–18 (notebook 01 cell 79). Note: the hit rates are MW-weighted top-25 picks, the bar is a median across assumptions |
| 10 | Recommendations | ✓ Consistent with the analysis. "Retrain each year" is supported by the drift findings |
