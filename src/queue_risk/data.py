"""Loading and basic cleaning of the Berkeley Lab Queued Up project-level file."""
import pandas as pd

from . import config


def load_raw_queue() -> pd.DataFrame:
    """Read the project-level sheet with light type cleaning. No rows are dropped."""
    df = pd.read_excel(config.QUEUE_XLSX, sheet_name=config.QUEUE_SHEET, header=1)
    # 5-digit zero-padded county FIPS string: the join key for every table.
    df["fips"] = df["fips_code"].map(lambda x: f"{int(x):05d}" if pd.notna(x) else pd.NA)
    df["q_year"] = df["q_year"].astype("Int64")
    # The codebook says entity + q_id is unique, but ~135 rows repeat (e.g. q_id "not assigned").
    # TODO(data lead): decide how to dedupe or disambiguate these.
    df["project_id"] = df["entity"].astype(str) + "|" + df["q_id"].astype(str)
    df["mw_total"] = df[["mw_1", "mw_2", "mw_3"]].sum(axis=1, min_count=1)
    df["is_hybrid"] = df["type_2"].notna()
    return df


def resolved_clean_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Clean-energy projects with a final outcome, plus a binary label `built`."""
    out = df[df["type_1"].isin(config.CLEAN_TYPES) & df["q_status"].isin(config.RESOLVED_STATUSES)].copy()
    out["built"] = (out["q_status"] == "operational").astype(int)
    return out


if __name__ == "__main__":
    raw = load_raw_queue()
    res = resolved_clean_energy(raw)
    print(f"raw rows: {len(raw):,}  |  resolved clean-energy rows: {len(res):,}")
    print(f"duplicate project_id: {raw['project_id'].duplicated().sum()}")
    print(res.groupby("q_year")["built"].agg(["size", "mean"]).loc[2005:].round(3).to_string())
