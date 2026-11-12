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


if __name__ == "__main__":
    raw = load_raw_queue()
    res = resolved_clean_energy(raw)
    print(f"raw rows: {len(raw):,}  |  resolved clean-energy rows since {config.FIRST_YEAR}: {len(res):,}")
    print(res.groupby("q_year")["built"].agg(["size", "mean"]).round(3).to_string())
