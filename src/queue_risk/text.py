"""Text workstream: local opposition, from unstructured text to a county-level signal.

STATUS (Nov 12). Source chosen: the Sabin Center's opposition reports (sabin.py), not
GDELT news (whose free API returns article lists only for recent months; the historical
archive needs BigQuery). Parsed: 572 contested-project paragraphs and ~450 restricted
jurisdictions, matched to counties. Validated (stack.py): a county with an earlier
contested project has about half the odds of a 5-year build, beyond the tabular model;
AUC gain +0.015, 95% CI [-0.003, +0.031]. Suggestive, not conclusive.

NEXT, once the labeling sheet is done (docs/labeling_guidelines.md): read each paragraph
for its OUTCOME (blocked / delayed / proceeded / unresolved) and CONCERNS, comparing four
readers on held-out labeled paragraphs, as in the Lecture 12 demo:

  0. Keyword rules              sabin.rule_outcome()      (Lecture 10: the baseline to beat)
  1. Counting words             TF-IDF + logistic regression
  2. General English            DistilBERT, frozen, + small head      (Lecture 12)
  3. Domain English             DistilBERT, fine-tuned (small learning rate, few epochs)

All four run on a laptop CPU: ~300 labeled paragraphs fine-tune in minutes. Report
macro-F1 and per-class recall, especially for `blocked`, the class that matters.
The chosen reader then labels all 572 paragraphs, and stack.py tests whether
"opposition that succeeded" (prior blocked projects) predicts better than any opposition.

The interface below is kept for a later news source (GDELT via BigQuery, or Media Cloud).
Compliance: publish derived scores only, never the report's or articles' text.
"""
import pandas as pd

from . import config


def pull_articles(start_year: int, end_year: int) -> pd.DataFrame:
    """Columns: url, date, fips, title, snippet. TODO(NLP lead)."""
    raise NotImplementedError


def score_counties(articles: pd.DataFrame, p_opposition) -> pd.DataFrame:
    """Aggregate article-level P(opposition) to the shared interface table.

    share_opposition = share of a county-year's articles classified as opposition.
    """
    a = articles.assign(year=pd.to_datetime(articles["date"]).dt.year,
                        is_opp=(pd.Series(p_opposition, index=articles.index) >= 0.5).astype(int))
    out = (a.groupby(["fips", "year"])
             .agg(n_articles=("is_opp", "size"), share_opposition=("is_opp", "mean"))
             .reset_index())
    return out[config.OPPOSITION_COLS]


def load_scores() -> pd.DataFrame | None:
    """The delivered score table, or None if it has not landed yet."""
    if not config.OPPOSITION_CSV.exists():
        return None
    return pd.read_csv(config.OPPOSITION_CSV, dtype={"fips": str})
