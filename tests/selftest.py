"""
Check the pipeline's promises without training anything big.

    python tests/selftest.py

Modeled on the class demos' self-tests: each check prints ok/FAIL, and the script
exits non-zero if anything failed. Most checks run on tiny hand-built tables whose
right answers can be worked out on paper. The last block reads the real Queued Up
file, if it is present, and refuses to pass if the numbers the README quotes move.

What this covers: leakage (no post-entry columns; lag features see only the past),
the temporal split and rolling folds, the opposition-score join, the benchmark
rules, the ranking metrics and the impact arithmetic. What it cannot cover: whether
the model is any good. That is the pipeline's validation table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from queue_risk import config, evaluate, features, impact  # noqa: E402
from queue_risk.models import FlatRate, HistoricalRate  # noqa: E402
from queue_risk.split import rolling_folds, temporal_split  # noqa: E402

FAILURES = []


def check(label, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def tiny_queue():
    """Four projects in one county. A is built in 2012, B withdraws in 2016,
    C enters in 2014 (after A was built, before B withdrew), D is elsewhere."""
    t = pd.Timestamp
    return pd.DataFrame({
        "project_id": ["A", "B", "C", "D"],
        "fips": ["01001", "01001", "01001", "02002"],
        "region": ["PJM", "PJM", "PJM", "ERCOT"],
        "type_1": ["Solar"] * 4, "type_clean": ["Solar"] * 4,
        "state": ["AL", "AL", "AL", "TX"], "service": ["NRIS"] * 4,
        "q_date": [t("2010-01-01"), t("2011-06-01"), t("2014-01-01"), t("2014-01-01")],
        "on_date": [t("2012-01-01"), pd.NaT, pd.NaT, pd.NaT],
        "wd_date": [pd.NaT, t("2016-01-01"), t("2017-01-01"), t("2015-01-01")],
        "q_year": [2010, 2011, 2014, 2014],
        "mw_total": [100.0, 50.0, 20.0, 10.0], "is_hybrid": [False] * 4,
        "q_status": ["operational", "withdrawn", "withdrawn", "withdrawn"],
        "built": [1, 0, 0, 0],
    })


def test_lag_features_see_only_the_past():
    print("\nLag features (Lecture 8: built from past data only)")
    q = tiny_queue()
    X, y = features.build_features(q, q)
    c = X.iloc[2]
    check("C sees A's build, which happened before C entered", c["county_built_before"] == 1)
    check("C does not see B's withdrawal, which happened after", c["county_withdrawn_before"] == 0)
    check("C's 3-year window counts B (2011) but not A (2010)", c["county_entries_3y"] == 1,
          f"got {c['county_entries_3y']}")
    check("A, the county's first project, has no history (cold start)",
          np.isnan(X.iloc[0]["county_built_share_before"]) and X.iloc[0]["county_has_history"] == 0)
    check("no post-entry column is a feature", not set(X.columns) & set(config.LEAKAGE_COLS))
    check("entry year is not a feature (trees cannot extrapolate)", "q_year" not in X.columns)


def test_opposition_join():
    print("\nOpposition score interface")
    q = tiny_queue()
    scores = pd.DataFrame({"fips": ["01001", "01001"], "year": [2013, 2014],
                           "n_articles": [4, 9], "share_opposition": [0.25, 0.9]})
    X, _ = features.build_features(q, q, opposition=scores)
    check("a 2014 entry gets its county's 2013 score, not 2014's",
          X.iloc[2]["share_opposition"] == 0.25 and X.iloc[2]["n_articles"] == 4)
    check("no articles -> coverage flag 0 and share left missing, not zero",
          X.iloc[3]["news_coverage"] == 0 and np.isnan(X.iloc[3]["share_opposition"]))
    try:
        features.add_opposition(q, scores.drop(columns="n_articles"))
        check("a table missing an interface column is refused", False)
    except ValueError:
        check("a table missing an interface column is refused", True)


def test_split():
    print("\nTemporal split (Lectures 2, 4, 6)")
    years = pd.Series(range(2005, 2020))
    train, test = temporal_split(years)
    check("train and test years do not overlap", not (train & test).any())
    check("every test year is after every train year", years[train].max() < years[test].min())
    folds = rolling_folds(years[train])
    ok = all(years[train].iloc[f].max() < years[train].iloc[v].min() for f, v in folds)
    check("each rolling fold validates on a year after everything it fits on", ok and len(folds) > 0)


def test_benchmarks():
    print("\nBenchmark rules")
    X = pd.DataFrame({"region": ["A"] * 30 + ["B"] * 5, "type_clean": ["Solar"] * 35})
    y = np.array([1] * 6 + [0] * 24 + [1] * 5)
    flat = FlatRate().fit(X, y).predict_proba(X)[:, 1]
    check("flat rate gives everyone the base rate", np.allclose(flat, y.mean()))
    hist = HistoricalRate(min_count=20).fit(X, y).predict_proba(X)[:, 1]
    check("a large cell uses its own rate", np.isclose(hist[0], 6 / 30))
    check("a small cell (5 projects) falls back to the technology rate", np.isclose(hist[-1], 11 / 35))


def test_metrics_and_impact():
    print("\nRanking metrics and impact arithmetic (Lecture 5)")
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    check("precision@3 of a perfect ranking is 1", evaluate.precision_at_k(y, y.astype(float), 3) == 1)
    check("tied scores earn the base rate, not the sort order's luck",
          np.isclose(evaluate.precision_at_k(y, np.ones(10), 3), y.mean()))
    mw = np.full(10, 100.0)
    p = np.array([0.9, 0.8, 0.7] + [0.1] * 7)          # backs projects 0, 1, 2
    r = impact.portfolio(y, mw, p, k=3, cost_per_project=2e6)
    check("backing 3 with 1 withdrawal wastes one project's spend", r["wasted_usd"] == 2e6)
    check("... spread over the 200 MW that got built", np.isclose(r["wasted_per_built_mw"], 1e4))


def test_built_within_label():
    print("\nFixed-horizon label: built within N years")
    from queue_risk import data
    t = pd.Timestamp
    q = pd.DataFrame({
        "project_id": list("ABCDEF"), "type_1": ["Solar"] * 6, "region": ["PJM"] * 5 + ["ISO-NE"],
        "q_year": [2010, 2010, 2010, 2010, 2022, 2010],
        "q_date": [t("2010-01-01")] * 4 + [t("2022-01-01"), t("2010-01-01")],
        "q_status": ["operational", "operational", "active", "operational", "operational", "operational"],
        "on_date": [t("2013-01-01"), t("2016-06-01"), pd.NaT, pd.NaT, t("2024-01-01"), t("2012-01-01")],
        "prop_date": [pd.NaT, pd.NaT, pd.NaT, t("2012-01-01"), pd.NaT, pd.NaT],
    })
    lab = data.built_within(q, horizon=5, cutoff="2025-12-31").set_index("project_id")["built"]
    check("built in year 3 -> 1", lab.get("A") == 1)
    check("built in year 6 -> 0 for a 5-year horizon", lab.get("B") == 0)
    check("still active when the window closed -> 0, not dropped", lab.get("C") == 0)
    check("built, no operation date: proposed date stands in", lab.get("D") == 1)
    check("window not closed by the cutoff -> excluded", "E" not in lab.index)
    check("regions that don't report dates -> excluded", "F" not in lab.index)


def test_real_data_checkpoints():
    print("\nReal data checkpoints (the numbers the README quotes)")
    if not config.QUEUE_XLSX.exists():
        print("  [skip] Queued Up file not found")
        return
    from queue_risk import data
    raw = data.load_raw_queue()
    res = data.resolved_clean_energy(raw)
    check("38,066 rows after de-duplicating 135 repeated IDs", len(raw) == 38066, f"got {len(raw):,}")
    check("project_id is unique", raw["project_id"].is_unique)
    check("every FIPS key is a 5-character string",
          raw["fips"].dropna().map(lambda s: isinstance(s, str) and len(s) == 5).all())
    ever = config.SPLITS["ever"]
    train, test = temporal_split(res["q_year"], ever["train"], ever["test"])
    check("ever built: 5,872 training projects, 2005-2014", train.sum() == 5872, f"got {train.sum():,}")
    check("ever built: 3,559 test projects, 2015-2017", test.sum() == 3559, f"got {test.sum():,}")
    w = data.built_within(raw, config.HORIZON)
    within = config.SPLITS["within"]
    train, test = temporal_split(w["q_year"], within["train"], within["test"])
    check("within 5y: 4,039 training projects, 2005-2014", train.sum() == 4039, f"got {train.sum():,}")
    check("within 5y: 3,188 test projects, 2019-2020", test.sum() == 3188, f"got {test.sum():,}")
    check("within 5y: train/test gap is at least the horizon",
          within["test"][0] - within["train"][1] >= config.HORIZON)
    check("only resolved outcomes are labeled", set(res["q_status"]) <= set(config.RESOLVED_STATUSES))


def main():
    for t in [test_lag_features_see_only_the_past, test_opposition_join, test_split,
              test_benchmarks, test_metrics_and_impact, test_built_within_label,
              test_real_data_checkpoints]:
        t()
    print(f"\n{'All checks passed.' if not FAILURES else f'{len(FAILURES)} FAILED: ' + '; '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
