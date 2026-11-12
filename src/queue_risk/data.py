"""Load and clean the Berkeley Lab Queued Up project-level file.

    python -m queue_risk.data        # prints the label counts by entry year
"""
import pandas as pd

from . import config


def load_raw_queue() -> pd.DataFrame:
    """Read the project-level sheet, with light type cleaning and de-duplication."""
    df = pd.read_excel(config.QUEUE_XLSX, sheet_name=config.QUEUE_SHEET, header=1)
    # 5-digit zero-padded county FIPS string: the join key for every table.
    # One row carries a garbled 9-digit code (603106019); anything over 5 digits is missing.
    df["fips"] = df["fips_code"].map(lambda x: f"{int(x):05d}" if pd.notna(x) and x < 1e5 else pd.NA)
    df["q_year"] = df["q_year"].astype("Int64")
    # The codebook says entity + q_id is unique. 135 IDs appear twice: mostly 1997-2002
    # gas plants re-listed at the end of the file without county/state. Keep the
    # more complete row of each pair.
    df["project_id"] = df["entity"].astype(str) + "|" + df["q_id"].astype(str)
    completeness = df.notna().sum(axis=1)
    df = (df.assign(_complete=completeness)
            .sort_values("_complete", ascending=False, kind="stable")
            .drop_duplicates("project_id")
            .drop(columns="_complete")
            .sort_index())
    df["mw_total"] = df[["mw_1", "mw_2", "mw_3"]].sum(axis=1, min_count=1)
    df["is_hybrid"] = df["type_2"].notna()
    return df


def resolved_clean_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Clean-energy projects with a final outcome since FIRST_YEAR, plus the label `built`."""
    keep = (df["type_1"].isin(config.CLEAN_TYPES)
            & df["q_status"].isin(config.RESOLVED_STATUSES)
            & (df["q_year"] >= config.FIRST_YEAR))
    out = df[keep].copy()
    out["built"] = (out["q_status"] == "operational").astype(int)
    return out


def built_within(df: pd.DataFrame, horizon: int, cutoff=config.DATA_CUTOFF) -> pd.DataFrame:
    """Clean-energy projects in the date-reporting regions whose `horizon`-year window has
    closed by `cutoff`, with the label `built` = reached operation within `horizon` years.

    Every status is kept: a project still active, suspended or withdrawn when its window
    closed is labeled 0. That is what removes the censoring of the "ever built" outcome.

    The 4% of built projects without an operation date take the proposed online date
    instead (flagged `on_date_imputed`); on projects with both dates, that stand-in gives
    the same 5-year label 90% of the time. It is used for the LABEL only, never as a feature.
    Built projects with neither date are dropped (their label is unknown).
    """
    end = pd.Timestamp(cutoff)
    keep = (df["type_1"].isin(config.CLEAN_TYPES)
            & df["region"].isin(config.DATE_REGIONS)
            & (df["q_year"] >= config.FIRST_YEAR)
            & (df["q_date"] + pd.DateOffset(years=horizon) <= end))
    out = df[keep].copy()
    is_op = out["q_status"] == "operational"
    out["on_date_imputed"] = is_op & out["on_date"].isna() & out["prop_date"].notna()
    done = out["on_date"].where(out["on_date"].notna(), out["prop_date"].where(is_op))
    out = out[~(is_op & done.isna())]
    done = done.loc[out.index]
    out["built"] = (done.notna() & (done <= out["q_date"] + pd.DateOffset(years=horizon))).astype(int)
    return out


def modeling_table(raw: pd.DataFrame, outcome: str = config.OUTCOME) -> pd.DataFrame:
    """The labeled projects for the chosen outcome definition."""
    if outcome == "within":
        return built_within(raw, config.HORIZON)
    if outcome == "ever":
        return resolved_clean_energy(raw)
    raise ValueError(f"unknown outcome {outcome!r}")


if __name__ == "__main__":
    raw = load_raw_queue()
    for outcome in ["within", "ever"]:
        res = modeling_table(raw, outcome)
        print()
        print(f"{outcome}: {len(res):,} labeled projects")
        print(res.groupby("q_year")["built"].agg(["size", "mean"]).round(3).T.to_string())
