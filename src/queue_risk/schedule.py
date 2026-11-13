"""Are projects coming online before, on, or after their projected date?

    python -m queue_risk.schedule

Two measures, both descriptive (not model features, because the proposed date is
revised after entry):

1. Schedule slip of COMPLETED projects: operation date minus the recorded proposed
   online date -> early / on time (within SCHEDULE_TOLERANCE_DAYS) / late.
   Only in operators where the proposed date was not overwritten with the actual one:
   in MISO 100% and in SPP 88% of completed projects carry a proposed date EXACTLY equal
   to their operation date, versus 0-10% in CAISO, ERCOT and PJM.
   Even there, the recorded date is the LAST projection, not the one filed at entry, so
   measured slip is a LOWER BOUND on the delay against the original plan.

2. Overdue ACTIVE projects: still active or suspended at the file's as-of date although
   their proposed online date has passed. Valid in every operator (a project that is
   active after its proposed date is late whatever revisions happened), and also a lower
   bound (a project whose date was pushed back is not counted).

FINDING (end-2025 file): measured slip mostly reflects how often each operator revises
the proposed date, not real differences in delay. ERCOT: 84% "on time", median slip 0;
ERCOT republishes projected dates monthly, so they converge on the actual date.
CAISO: 96% late, median slip 2.9 years; its dates look like the original filings.
Compare operators only with that caveat, or not at all.

The original projections need earlier snapshots of the queue: the previous Queued Up
edition (data through 2024), older editions via the Wayback Machine, or the operators'
own monthly queue reports.
"""
import pandas as pd

from . import config, data

SCHEDULE_REGIONS = ["CAISO", "ERCOT", "PJM"]
SCHEDULE_TOLERANCE_DAYS = 90


def slip(df: pd.DataFrame, tolerance_days=SCHEDULE_TOLERANCE_DAYS) -> pd.DataFrame:
    """Completed projects with both dates in SCHEDULE_REGIONS, with slip and status."""
    c = df[(df["q_status"] == "operational") & df["region"].isin(SCHEDULE_REGIONS)
           & df["on_date"].notna() & df["prop_date"].notna()].copy()
    days = (c["on_date"] - c["prop_date"]).dt.days
    c["slip_years"] = days / 365.25
    c["schedule"] = pd.cut(days, [-float("inf"), -tolerance_days - 1, tolerance_days, float("inf")],
                           labels=["early", "on time", "late"])
    return c


def overdue(df: pd.DataFrame, cutoff=config.DATA_CUTOFF) -> pd.DataFrame:
    """Active/suspended projects with a proposed date, flagged if it has already passed."""
    a = df[df["q_status"].isin(["active", "suspended"]) & df["prop_date"].notna()].copy()
    a["overdue"] = a["prop_date"] < pd.Timestamp(cutoff)
    a["overdue_years"] = ((pd.Timestamp(cutoff) - a["prop_date"]).dt.days / 365.25).clip(lower=0)
    return a


def summary(raw: pd.DataFrame):
    scope = raw[data.in_scope(raw)]
    s, o = slip(scope), overdue(scope[scope["region"].isin(config.DATE_REGIONS)])
    def shares(key):
        t = pd.crosstab(s[key], s["schedule"], normalize="index").round(2)
        t["n"] = s.groupby(key).size()
        t["median slip (yrs)"] = s.groupby(key)["slip_years"].median().round(2)
        return t

    by_region, by_tech = shares("region"), shares("type_1")
    od = o.groupby("region").agg(n=("overdue", "size"), overdue=("overdue", "mean"),
                                 median_years_overdue=("overdue_years", lambda x: x[x > 0].median()))
    return s, by_region, by_tech, od.round(2)


def main():
    raw = data.load_raw_queue()
    s, by_region, by_tech, od = summary(raw)
    tol = SCHEDULE_TOLERANCE_DAYS
    print(f"1. Completed projects vs. their last recorded proposed date (on time = within {tol} days)")
    print(f"   {len(s):,} projects in {', '.join(SCHEDULE_REGIONS)}; slip is a lower bound\n")
    print(by_region.to_string(), "\n")
    print(by_tech.to_string(), "\n")
    print("2. Active/suspended projects already past their proposed online date (end-2025)")
    print(od.to_string())


if __name__ == "__main__":
    main()
