# The Deal Desk (planned)

> **Built so far (Nov 20): a simpler version for the video.** Run `python demo/export_data.py`, then `python demo/serve.py` (http://localhost:8000).
>
> - **What it shows:** 100 projects drawn at random from the 3,464 entries of 2015–2018 with a capacity the page can show (20 PJM 5 kW batteries left out, none of them built), scored by the out-of-time LASSO (C = 0.1, trained on 2005–2014 only). Three states: the entry-day list in queue order → **Run the model** (scores, re-sort, top 25 highlighted) → **Show what actually happened** (✓ = built within 5 years, plus a scoreboard). Under the scoreboard: the same top-vs-random comparison over all 3,484 entries, and a check of how typical the sample is (20,000 redraws). After the reveal, a **"What does the ranking save?"** button opens a savings pop-up (only on click, never by itself) that gives `impact.py`'s metric, development money lost per built MW, for the model's top 25 vs. a random 25, and in dollars at the sourced `DEV_COST_RANGE_USD` ($2.40M / $4.95M / $8.65M per withdrawn project), plus the same for all 3,484 entries. `export_data.py` computes every figure; the page only prints them. It ranks by chance of being built, not chance × MW, so it shows 72% on the page (46% over the full cohort), not the deck's 85%.
> - **Current draw (Nov 25):** 6 of the 100 projects were built within 5 years, and the model's top 25 holds 5 of them (1.5 expected at random). Over 20,000 redraws the top 25 averages 4.2 builds vs. 2.2 at random, so this draw is luckier for the model than 83% of draws. Over the full cohort, the top 871 holds 142 builds vs. 77 at random and 308 with perfect hindsight.
> - **Tabular model only.** No Sabin text or opposition flag (team decision for the video; `sabin_labels_all.csv` also wasn't on the machine that built it).
> - **The sample seed is 15072, not `config.SEED`.** The first draw (15071) was luckier for the model than 95% of random draws. Before looking at any other draw we committed to the next seed and to keeping whatever it gave. Don't re-pick it; that would be hand-picking. On Nov 24 we dropped the 0 MW entries from the pool; with the same seed, that 50-project draw held a single build, so we moved to 100 projects / top 25 (same seed, same pool) and kept the result. On Nov 25 we merged main's frozen spec, which drops odd rows (0 or negative MW, fossil-paired) from the cohort; the same seed then drew a different 100 (11 builds became 6), and we kept that too. History in `export_data.py`.
> - **Self-checks:** `export_data.py` refuses to write `data.js` unless the out-of-time AUC is still 0.686 and the cohort is still 3,484 projects (both read from the frozen spec, `src/queue_risk/frozen.py`). It drops 2019–2020 entries before building features, so the test cohort is never scored.
> - **Files:** `index.html` (page, styles and script inline; the script only sorts, reveals and counts), `data.js` (generated; commit it so the page works from a clone), `export_data.py`, `serve.py`. There is no `app.js`.
> - **Not built:** the map, the ranking-rule picker, the cost dial and the SHAP drivers described below.

The interactive page for the pitch and the two-minute video. It's built the way the class demos are: one static `index.html`, a generated `data.js`, and no build step. It is modeled most closely on the Lecture 6 demo (*Fire Line*: rank fires on a map, then reveal what actually happened) and the Lecture 5 demo (*Care Management Worklist*: a capacity dial and a running dollar figure).

## The page

A map of the counties where the 2015–2017 queue entries sat. Each project is a dot sized by MW. The investor has room to back **K** projects this year.

- **Pick a ranking rule:**
  - *Flat rate*: today's rule of thumb. It gives every project the same chance, so it can't rank them; the page shows its expected value over a random pick, as the Lecture 5 demo does for tied scores.
  - *Historical rate* by region × technology
  - the fitted model
  - *Perfect hindsight* as the ceiling

  The top K get a ring.
- **Show what actually happened** marks the projects that were really built. When the two marks sit on different dots, the ranking got it wrong. Nothing on the page narrates the outcome; the marks make the argument.
- **Three running figures**, all the same arithmetic on the same projects: projects built, MW built, and **wasted development spend per built MW**. The spend is computed at the cost-per-project dial, which spans the sourced range.
- Click a project to see its top drivers (SHAP) and its county's opposition score.

## Rules the page keeps (borrowed from the class demos)

- Every number on screen comes from `data.js`, and `export_data.py` builds `data.js` from the raw file with the repo's own code.
- `export_data.py` checks itself against the numbers in the slides and refuses to write if anything moved.
- Every assumption, including cost per project and K, is printed in the footer so the room can argue with it.
- Say what the model can't do on the page itself: it ranks likelihood of being built. It doesn't say backing a project *causes* it to be built.

## Files (when built)

| File | What it is |
|---|---|
| `index.html` | The page: layout, styles, the map |
| `app.js` | Drawing and interaction; no numbers computed beyond sums |
| `data.js` | Generated. Test-cohort projects, scores per rule, assumptions |
| `export_data.py` | Builds `data.js`, with self-checks (currently a stub) |
| `serve.py` | Local web server, for browsers that block `file://` data files |
