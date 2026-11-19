"""Text workstream: local opposition, from unstructured text to a per-project score.

Source: the Sabin Center's opposition reports (sabin.py), 572 contested-project
paragraphs matched to counties. All 572 are labeled (data/interim/sabin_labels_all.csv,
gitignored): 100 by two humans, 472 by Claude following docs/labeling_guidelines.md.

The text-only model has two stages (notebooks/02_paul_text_models.ipynb):

  stage 1  the READER, paragraph -> P(proceeded) vs blocked/delayed. A Lecture 10-12
           ladder: keyword rules, TF-IDF + logistic regression, frozen DistilBERT,
           fine-tuned DistilBERT. Trained on Claude's labels, tested on the human ones.
  stage 2  each queue project's county text history -> P(built within 5 years). The
           features below count the county's EARLIER contested paragraphs and what the
           reader says happened to them. Scored out-of-fold and written to
           data/processed/text_scores.csv, the input to late fusion with the tabular model.

Dating (Lecture 8, lags from the past only): a paragraph counts as "contested" for a
project entering in year t when its first year < t, and its OUTCOME counts only when
its last year < t. Until then it is "contested, outcome not yet known".

Compliance: publish derived scores only, never the report's or articles' text.
"""
import re

import numpy as np
import pandas as pd

from . import config

LABELS_CSV = config.INTERIM / "sabin_labels_all.csv"
HUMAN_CSV = config.INTERIM / "sabin_labels_human.csv"
CONTESTED_CSV = config.INTERIM / "sabin_contested.csv"
TEXT_SCORES_CSV = config.PROCESSED / "text_scores.csv"

# Binary target: did the contested project go ahead? (docs/labeling_guidelines.md)
TARGET = {"proceeded": 1, "blocked": 0, "delayed": 0}          # unresolved -> NaN
CONCERNS = ["viewshed", "farmland", "property", "wildlife", "health",
            "safety", "water", "cultural", "process", "other"]
TECH_PATTERNS = {"solar": r"solar|photovoltaic|\bpv\b",
                 "wind": r"\bwind|turbine",
                 "battery": r"batter|energy storage|\bbess\b"}
TYPE_TO_TECH = {"Solar": "solar", "Wind": "wind", "Offshore Wind": "wind", "Battery": "battery"}
_APPENDIX = re.compile(r"\s*\d*\s*Appendix: Other Resources.*$", re.S)   # pid 571 parse bug
_YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def load_paragraphs(labels_csv=LABELS_CSV, contested_csv=CONTESTED_CSV) -> pd.DataFrame:
    """The 572 labeled paragraphs, one row each, with the columns stage 1 and 2 need.

    `id` is the row index in sabin_contested.csv, which brings in the county list.
    `y` is 1 = proceeded, 0 = blocked or delayed, NaN = unresolved.
    `is_gold` marks the 100 human-labeled paragraphs (the stage-1 test set).
    """
    lab = pd.read_csv(labels_csv, encoding="utf-8-sig")
    con = pd.read_csv(contested_csv, dtype={"fips_list": str}, encoding="utf-8-sig")
    if len(con) != len(lab) or not (con.loc[lab["id"], "project"].to_numpy() == lab["project"].to_numpy()).all():
        raise ValueError("sabin_labels_all.csv and sabin_contested.csv disagree: was sabin.py rerun?")
    p = lab.copy()
    p["fips_list"] = con.loc[p["id"], "fips_list"].fillna("").to_numpy()
    p["text"] = p["text"].fillna("").map(lambda t: _APPENDIX.sub("", t).strip())
    years = p["text"].map(lambda t: [int(y) for y in _YEAR.findall(t)])
    p["first_year"] = years.map(lambda ys: min(ys) if ys else np.nan)
    p["last_year"] = years.map(lambda ys: max(ys) if ys else np.nan)
    p["y"] = p["outcome"].map(TARGET)
    p["is_gold"] = p["source"].str.startswith("human")
    p["n_words"] = p["text"].str.split().str.len()
    name_and_text = (p["project"].fillna("") + " " + p["text"]).str.lower()
    for tech, pat in TECH_PATTERNS.items():
        p[f"tech_{tech}"] = name_and_text.str.contains(pat, regex=True).astype(int)
    tags = p["concerns"].fillna("").str.split(";").map(lambda xs: {x.strip() for x in xs if x.strip()})
    for k in CONCERNS:
        p[f"concern_{k}"] = tags.map(lambda s, k=k: int(k in s))
    return p.set_index("id", drop=False).rename_axis(None)


def county_text_features(projects: pd.DataFrame, paras: pd.DataFrame, p_reader,
                         emb: np.ndarray | None = None) -> pd.DataFrame:
    """One row per queue project, built only from paragraphs dated before its entry year.

    projects  needs fips, state, q_year, type_1 (as data.built_within returns them)
    paras     load_paragraphs() output
    p_reader  P(proceeded) for each paragraph, aligned with `paras` (out-of-fold)
    emb       optional paragraph embeddings, aligned with `paras`; adds the mean
              embedding of the county's earlier paragraphs (zeros when there are none)

    County columns (c_*): contested, known (outcome dated before entry), pending,
    blocked and proceeded (expected counts from the reader, known outcomes only),
    recent3 (contested in the 3 years before entry), same_tech, one count per concern.
    State columns (s_*): contested, blocked, proceeded, as a fallback where the county
    itself has no history.
    """
    p = np.asarray(p_reader, dtype=float)
    fy, ly = paras["first_year"].to_numpy(float), paras["last_year"].to_numpy(float)
    tech = {t: paras[f"tech_{t}"].to_numpy() for t in TECH_PATTERNS}
    conc = paras[[f"concern_{k}" for k in CONCERNS]].to_numpy()
    out = pd.DataFrame(0.0, index=projects.index, columns=
                       ["c_contested", "c_known", "c_pending", "c_blocked", "c_proceeded", "c_recent3",
                        "c_same_tech"] + [f"c_concern_{k}" for k in CONCERNS]
                       + ["s_contested", "s_blocked", "s_proceeded"])
    E = None
    if emb is not None:
        E = pd.DataFrame(0.0, index=projects.index, columns=[f"emb_{i}" for i in range(emb.shape[1])])

    def _fill(rows, idx, prefix):
        """rows: projects sharing a county (or state); idx: paragraph positions there."""
        if len(idx) == 0 or len(rows) == 0:
            return
        q = rows["q_year"].to_numpy(float)[:, None]
        seen = fy[idx][None, :] < q                       # contested before entry
        known = seen & (ly[idx][None, :] < q)             # outcome dated before entry
        out.loc[rows.index, f"{prefix}_contested"] = seen.sum(1)
        out.loc[rows.index, f"{prefix}_blocked"] = (known * (1 - p[idx])).sum(1)
        out.loc[rows.index, f"{prefix}_proceeded"] = (known * p[idx]).sum(1)
        if prefix == "c":
            out.loc[rows.index, "c_known"] = known.sum(1)
            out.loc[rows.index, "c_pending"] = seen.sum(1) - known.sum(1)
            out.loc[rows.index, "c_recent3"] = (seen & (fy[idx][None, :] >= q - 3)).sum(1)
            pt = rows["type_1"].map(TYPE_TO_TECH).to_numpy()
            same = np.stack([tech[t][idx] if t in tech else np.zeros(len(idx), int) for t in pt])
            out.loc[rows.index, "c_same_tech"] = (seen & (same == 1)).sum(1)
            out.loc[rows.index, [f"c_concern_{k}" for k in CONCERNS]] = seen.astype(float) @ conc[idx]
            if E is not None:
                n = seen.sum(1, keepdims=True)
                E.loc[rows.index] = np.where(n > 0, (seen.astype(float) @ emb[idx]) / np.maximum(n, 1), 0.0)

    county_idx = {}
    for i, fl in enumerate(paras["fips_list"]):
        for f in (fl.split(";") if isinstance(fl, str) and fl else []):
            county_idx.setdefault(f, []).append(i)
    for f, rows in projects.groupby("fips"):
        _fill(rows, np.array(county_idx.get(f, []), int), "c")
    state_idx = {s: np.flatnonzero((paras["state"] == s).to_numpy()) for s in paras["state"].unique()}
    for s, rows in projects.groupby("state"):
        _fill(rows, state_idx.get(s, np.array([], int)), "s")
    return out if E is None else out.join(E)


def write_text_scores(df: pd.DataFrame, path=TEXT_SCORES_CSV):
    """The late-fusion interface: project_id, q_year, y, has_text, p_text (+ features)."""
    need = ["project_id", "q_year", "y", "has_text", "p_text"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"text_scores needs columns {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df[need + [c for c in df.columns if c not in need]].to_csv(path, index=False)
    return path


def load_text_scores() -> pd.DataFrame | None:
    """The stage-2 scores for fusion, or None if the notebook has not written them yet."""
    if not TEXT_SCORES_CSV.exists():
        return None
    return pd.read_csv(TEXT_SCORES_CSV)


# ------------------------------------------------------------------ news (not used yet)
# Kept for a later news source (GDELT via BigQuery, or Media Cloud).

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
