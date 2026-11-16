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
import re

import numpy as np
import pandas as pd

from . import config

# "service" (ERIS/NRIS) was dropped: its blanks leak the outcome in CAISO and MISO (_hide_missingness).
CATEGORICAL = ["region", "type_clean", "state"]
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


# ------------------------------------------------------------------ point of interconnection
# The POI name is filed with the request ("Dotson 69kV Substation", "Lugo-Pisgah No. 2 230 kV
# tran line"). Its text gives the voltage and whether the project taps a line or connects at
# a substation; its value groups projects that share a connection point, whose track record
# (as of entry) is the finest-grained lag variable we have.

def poi_key(names: pd.Series, entity: pd.Series) -> pd.Series:
    k = names.astype(str).str.lower().str.replace(r"\d{2,3}\s*-?\s*kv", "", regex=True)
    k = k.str.replace(r"[^a-z]", "", regex=True)
    k = k.where(~k.isin(["", "nan", "none"]))
    return (k + "|" + entity.astype(str)).where(k.notna())


def poi_text_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["poi_name"].astype(str)
    kv = t.str.extract(r"(\d{2,3})\s*-?\s*kv", flags=re.IGNORECASE)[0].astype(float)
    sub = t.str.contains(r"substation|switching|station|\bsub\b", case=False)
    line = t.str.contains(r"-|\bline\b|\btap\b|\bto\b", case=False) & ~sub
    return pd.DataFrame({"poi_kv": kv, "poi_is_line": line.astype(int), "poi_is_sub": sub.astype(int)},
                        index=df.index)


# ------------------------------------------------------------------ county context (Census)
CENSUS_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/Gaz_counties_national.zip"


def census_counties() -> pd.DataFrame:
    """2010 Census county gazetteer: log population density, log land area, centroid lat/lon.
    Slow-moving county facts; 2010 values are used for every cohort (a mild anachronism for
    2005-2009 entries, noted in the README). Downloaded once to data/external/census/."""
    path = config.EXTERNAL / "census" / "Gaz_counties_national.txt"
    if not path.exists():
        import io, urllib.request, zipfile
        path.parent.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(CENSUS_URL).read())).extractall(path.parent)
    g = pd.read_csv(path, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    g.columns = g.columns.str.strip()
    return pd.DataFrame({"log_density": np.log1p(g["POP10"] / g["ALAND_SQMI"]),
                         "log_land": np.log(g["ALAND_SQMI"]),
                         "lat": g["INTPTLAT"], "lon": g["INTPTLONG"]}).set_index(g["GEOID"])


def _hide_missingness(X: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Keep blank fields from telling the model the outcome.

    In the end-2025 snapshot, RESOLVED projects have sparser records than active ones:
    in SPP the POI name is missing only for withdrawn projects (14%, 0% for every other
    status); in ERCOT the county code is missing for 0.6% of active, 36% of operational
    and 59% of withdrawn projects. A blank field is therefore written after the outcome,
    i.e. leakage (caught by C. Loriot's SHAP check). So every feature of a blank field is
    left missing, and the model pipeline imputes it WITHOUT a missing-value indicator.
    The one legitimate missingness signal, a county with no resolved projects before
    entry (cold start), keeps its explicit flag, but only where the county is known.
    """
    X = X.copy()
    no_county = p["fips"].isna().to_numpy()
    X.loc[no_county, "county_has_history"] = np.nan
    if "poi_is_line" in X:
        no_poi = p["poi_name"].isna().to_numpy()
        X.loc[no_poi, ["poi_is_line", "poi_is_sub"]] = np.nan
    return X


DEFAULT_EXTRAS = ("poi",)   # census: mixed out-of-time, see docs/model_log.md


def build_features(raw: pd.DataFrame, projects: pd.DataFrame, opposition: pd.DataFrame | None = None,
                   extras=DEFAULT_EXTRAS):
    """Return (X, y) for `projects` (a subset of `raw`, e.g. resolved clean-energy rows).

    extras: optional feature groups, kept switchable so their value can be measured
      "poi"     POI text (voltage, line tap vs. substation) + POI track record as of entry
      "census"  county density, land area, centroid lat/lon
    """
    p = projects.copy()
    p["log_mw"] = np.log1p(p["mw_total"])
    p["q_month"] = p["q_date"].dt.month
    p["is_hybrid"] = p["is_hybrid"].astype(int)
    p["region_tech"] = p["region"].astype(str) + "|" + p["type_1"].astype(str)
    ref = raw.assign(region_tech=raw["region"].astype(str) + "|" + raw["type_1"].astype(str))

    parts = [p[CATEGORICAL + NUMERIC_BASE],
             _asof_group_stats(ref, p, "fips", "county"),
             _asof_group_stats(ref, p, "region_tech", "region_tech")]
    if "poi" in extras and "poi_name" in p.columns:
        p["poi_key"] = poi_key(p["poi_name"], p["entity"])
        ref = ref.assign(poi_key=poi_key(raw["poi_name"], raw["entity"]))
        parts += [poi_text_features(p), _asof_group_stats(ref, p, "poi_key", "poi")]
    X = pd.concat(parts, axis=1)
    X["county_has_history"] = X["county_built_share_before"].notna().astype(int)
    X = _hide_missingness(X, p)
    if "census" in extras:
        X = X.join(p[["fips"]].join(census_counties(), on="fips").drop(columns="fips"))

    if opposition is not None:
        opp = add_opposition(p[["fips", "q_year"]], opposition)
        X = X.join(opp[["n_articles", "share_opposition", "news_coverage"]])

    leaked = set(X.columns) & set(config.LEAKAGE_COLS)
    assert not leaked, f"leakage columns in features: {leaked}"
    return X, p["built"].astype(int)
