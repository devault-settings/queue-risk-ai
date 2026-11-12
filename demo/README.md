# The Deal Desk (planned)

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
