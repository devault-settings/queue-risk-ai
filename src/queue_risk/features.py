"""Entry-time features: only what an investor could know on the day a project enters the queue.

Two families, both from the lectures:

- Project attributes (Lecture 2/3): technology, size, grid region, state, service type,
  hybrid pairing. Categorical columns are one-hot encoded inside the model pipeline.
- Track-record "lag variables" (Lecture 8, CTR): how busy the county and region queues
  were in the 3 years before entry, and how earlier projects there had resolved *by the
  entry date*. Lecture 8's two caveats apply: lags must be built from past data only
  (an earlier project counts as built only if its on_date precedes this q_date), and
  new counties have no history (cold start), so missing history stays NaN and gets a flag.

q_year is deliberately NOT a feature. The split is temporal, and Lecture 5 warns that
trees cannot extrapolate: every test year would fall in the last training bin.
Time trends enter through the lag variables instead.
"""
import numpy as np
import pandas as pd

from . import config

CATEGORICAL = ["region", "type_clean", "service", "state"]
NUMERIC_BASE = ["log_mw", "is_hybrid", "q_month"]
WINDOW_YEARS = 3


def _ns(s: pd.Series) -> np.ndarray:
    """Datetimes as int64 nanoseconds; missing dates become +inf (never happened)."""
    out = pd.to_datetime(s).to_numpy(dtype="datetime64[ns]").astype("int64").astype("float64")
    out[pd.isna(s).to_numpy()] = np.inf
    return out


def _asof_group_stats(ref: pd.DataFrame, target: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """For each target row: activity in its `key` group strictly before its q_date.

    ref holds every project in the file (any technology, any status), because the
    whole queue is public information at entry time. Only dates are used to decide
    what counts as "known", never the current status.
    """
    window = np.timedelta64(365 * WINDOW_YEARS, "D").astype("timedelta64[ns]").astype("int64")
    cols = {f"{prefix}_entries_{WINDOW_YEARS}y": np.full(len(target), np.nan),
            f"{prefix}_mw_{WINDOW_YEARS}y": np.full(len(target), np.nan),
            f"{prefix}_built_before": np.full(len(target), np.nan),
            f"{prefix}_withdrawn_before": np.full(len(target), np.nan)}
    ref = ref[ref[key].notna()]
    groups = {k: g for k, g in ref.groupby(key, sort=False)}
    tpos = {k: idx for k, idx in pd.Series(np.arange(len(target))).groupby(target[key].to_numpy()).groups.items()}
    for k, positions in tpos.items():
        g = groups.get(k)
        if g is None or pd.isna(k):
            continue
        positions = np.asarray(positions)
        q = _ns(target["q_date"].iloc[positions])
        entry = _ns(g["q_date"])
        order = np.argsort(entry)
        entry_sorted = entry[order]
        mw_cum = np.concatenate([[0.0], np.nancumsum(g["mw_total"].to_numpy(dtype=float)[order])])
        hi = np.searchsorted(entry_sorted, q, side="left")
        lo = np.searchsorted(entry_sorted, q - window, side="left")
        cols[f"{prefix}_entries_{WINDOW_YEARS}y"][positions] = hi - lo
        cols[f"{prefix}_mw_{WINDOW_YEARS}y"][positions] = mw_cum[hi] - mw_cum[lo]
        built = np.sort(_ns(g["on_date"]))
        wd = np.sort(_ns(g["wd_date"]))
        cols[f"{prefix}_built_before"][positions] = np.searchsorted(built, q, side="left")
        cols[f"{prefix}_withdrawn_before"][positions] = np.searchsorted(wd, q, side="left")
    out = pd.DataFrame(cols, index=target.index)
    resolved = out[f"{prefix}_built_before"] + out[f"{prefix}_withdrawn_before"]
    out[f"{prefix}_built_share_before"] = out[f"{prefix}_built_before"] / resolved.where(resolved > 0)
    return out


def add_opposition(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Join the NLP lead's county-year opposition score, using the year BEFORE entry.

    Counties with no articles keep share_opposition = NaN and get news_coverage = 0
    (Risks: thin rural coverage), rather than being treated as zero opposition.
    """
    missing = set(config.OPPOSITION_COLS) - set(scores.columns)
    if missing:
        raise ValueError(f"opposition table is missing columns: {sorted(missing)}")
    s = scores[config.OPPOSITION_COLS].rename(columns={"year": "score_year"})
    out = df.assign(score_year=df["q_year"].astype("Int64") - 1).merge(
        s.assign(score_year=s["score_year"].astype("Int64")), on=["fips", "score_year"], how="left")
    out.index = df.index
    out["n_articles"] = out["n_articles"].fillna(0)
    out["news_coverage"] = (out["n_articles"] > 0).astype(int)
    return out.drop(columns="score_year")


def build_features(raw: pd.DataFrame, projects: pd.DataFrame, opposition: pd.DataFrame | None = None):
    """Return (X, y) for `projects` (a subset of `raw`, e.g. resolved clean-energy rows)."""
    p = projects.copy()
    p["log_mw"] = np.log1p(p["mw_total"])
    p["q_month"] = p["q_date"].dt.month
    p["is_hybrid"] = p["is_hybrid"].astype(int)
    p["region_tech"] = p["region"].astype(str) + "|" + p["type_1"].astype(str)
    ref = raw.assign(region_tech=raw["region"].astype(str) + "|" + raw["type_1"].astype(str))

    parts = [p[CATEGORICAL + NUMERIC_BASE],
             _asof_group_stats(ref, p, "fips", "county"),
             _asof_group_stats(ref, p, "region_tech", "region_tech")]
    X = pd.concat(parts, axis=1)
    X["county_has_history"] = X["county_built_share_before"].notna().astype(int)

    if opposition is not None:
        opp = add_opposition(p[["fips", "q_year"]], opposition)
        X = X.join(opp[["n_articles", "share_opposition", "news_coverage"]])

    leaked = set(X.columns) & set(config.LEAKAGE_COLS)
    assert not leaked, f"leakage columns in features: {leaked}"
    return X, p["built"].astype(int)
