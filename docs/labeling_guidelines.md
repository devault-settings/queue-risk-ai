# Labeling guidelines: Sabin Center contested-project paragraphs

**Sheet:** `data/interim/sabin_label_sheet.csv`. It's 300 paragraphs sampled from the September 2026 edition and opens in Excel. Rebuild it with `python -m queue_risk.sabin`. It's gitignored because the paragraphs are the report's text, so share it with labelers directly, never through the repo.

**Time:** about 30 seconds a paragraph, roughly 2.5 hours for 300. Do the first 50, then pause so we can check the guidelines against real cases before doing the rest.

## Column `outcome`: one label, as of the *last* event the paragraph describes

| Label | Use when | Example cue |
|---|---|---|
| `blocked` | The project was stopped: permit denied with no successful appeal, developer withdrew or cancelled, a ban or moratorium killed it | "the Board denied the permit"; "the developer withdrew its application" |
| `delayed` | Still alive but materially set back: a lawsuit pending, a moratorium pausing it, or a denial under appeal or resubmission | "re-submitted the application with a modified plan" |
| `proceeded` | Approved, under construction, or operating, despite the opposition | "the commission approved the project 3-2" |
| `unresolved` | The paragraph describes opposition but no decision yet, or the outcome isn't stated | "residents voiced concerns at a public hearing" |

**Rules:**
- If a project was denied and then later approved, the label is `proceeded`. The last event wins.
- If a denial is followed by resubmission and the paragraph ends there, use `delayed`.
- The `rule_suggestion` column is a keyword guess (the baseline the trained models must beat). **Don't copy it.** Read the paragraph; the guess is often wrong when a paragraph mentions both a denial and an approval.

## Column `concerns`: every concern residents raised, separated by semicolons

`viewshed` (aesthetics, "industrialization") · `farmland` (agricultural land, rural character) · `property` (property values) · `wildlife` (birds, habitat, environment) · `health` (noise, shadow flicker, health effects) · `safety` (fire, especially batteries; glare for aviation) · `water` (runoff, wells, drainage) · `cultural` (tribal, historic, cultural sites) · `process` (lack of notice, out-of-state developer, fairness) · `other` (describe it in `notes`)

Leave it blank if the paragraph names no concern.

## Column `notes`

Anything ambiguous: two projects in one paragraph, an outcome that's unclear, a location that doesn't look like a county.

## Agreement check (Monday)

One teammate labels the same 50 paragraphs independently. We report Cohen's kappa for `outcome`. If kappa is below 0.6, the definitions above need tightening before the labels are used.
