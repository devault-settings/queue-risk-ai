"""Text workstream (NLP lead): local news -> county-year opposition score.

This module fixes the INTERFACE the modeling workstream builds against. The implementation
follows Lectures 10-12 and the Lecture 12 demo ("The Overnight Desk"), which compares three
readers of the same labeled sentences. We do the same with three readers of local news:

  1. Counting words      TF-IDF + logistic regression          (Lecture 10: the baseline to beat)
  2. General English     DistilBERT, frozen, + small head       (Lecture 12: transfer learning)
  3. Domain English      DistilBERT, fine-tuned on our labels  (Lecture 12: fine-tuning;
                                                                small learning rate, few epochs)

Lecture 12's lesson to plan for: the frozen model may buy nothing over counting words;
the gain comes from fine-tuning, and it concentrates in the rare class (there, bad news;
here, opposition). So report per-class recall, not just accuracy.

Deep models are selected on a single validation set, not cross-validation (Lecture 8/9:
too expensive). Fine-tune in Colab; never commit weights (.pt) - only the score table.

Pipeline:
  pull_articles()     GDELT, solar/wind/storage project news, tagged to a county FIPS
  label set           300-500 snippets: opposition / support / neutral, with written guidelines
                      and a second-labeler agreement check (LLM drafts allowed if a person checks)
  train readers       60/20/20 split, the three readers above, per-class recall on test
  score_counties()    apply the chosen reader, aggregate to (fips, year)

Compliance: publish article URLs and derived scores only, never article text.
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
