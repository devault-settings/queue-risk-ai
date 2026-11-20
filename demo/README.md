# The Deal Desk (planned)

> **Built so far (Nov 20): a simpler version for the video.** Run `python demo/export_data.py`, then `python demo/serve.py` (http://localhost:8000).
>
> - **What it shows:** 50 projects drawn at random from the 3,490 entries of 2015–2018, scored by the out-of-time LASSO (C = 0.1, trained on 2005–2014 only). Three states: the entry-day list in queue order → **Run the model** (scores, re-sort, top 10 highlighted) → **Show what actually happened** (✓ = built within 5 years, plus a scoreboard). Under the scoreboard: the same top-vs-random comparison over all 3,490 entries, and a check of how typical the sample is (20,000 redraws).
> - **Tabular model only.** No Sabin text or opposition flag (team decision for the video; `sabin_labels_all.csv` also wasn't on the machine that built it).
> - **The sample seed is 15072, not `config.SEED`.** The first draw (15071) was luckier for the model than 95% of random draws. Before looking at any other draw we committed to the next seed and to keeping whatever it gave. Don't re-pick it; that would be hand-picking.
> - **Self-checks:** `export_data.py` refuses to write `data.js` unless the out-of-time AUC is still 0.686 and the cohort is still 3,490 projects (`docs/model_log.md`). It drops 2019–2020 entries before building features, so the test cohort is never scored.
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
