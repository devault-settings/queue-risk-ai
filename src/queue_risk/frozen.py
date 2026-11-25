"""The frozen specification for the ONE-TIME test run (docs/FINAL_RUN.md).

Model selection happens wherever the pipeline runs, and it is environment-dependent: on
Colab, LASSO selected C = 0.01, on the laptop C = 0.1. So the final run must not
re-select anything. `--final` (pipeline, stack, timeline) uses ONLY what is written here,
and `preflight()` refuses to start if the data, the configuration or a reproduced
validation number disagrees with it, or if a team decision is still pending.

Values are the laptop run of Nov 15 (reports/results_within_validation.csv, after the
blank-field leakage fix), the run every README and model-log number refers to.
Change anything here only by team decision, before the final run, and say so in the commit.
"""
FROZEN = {
    "frozen_on": "2026-11-24",
    "status": "SIGNED OFF Nov 24, 2026: Audrey DeVault, Paul Bustarret, Charles Loriot, Sara Alafifi",

    # ---- task and data
    "outcome": "within",
    "horizon": 5,
    "regions": ["CAISO", "ERCOT", "MISO", "NYISO", "PJM", "SPP"],
    "train": (2005, 2014),
    "out_of_time": (2015, 2018),
    "test": (2019, 2020),

    # TEAM DECISION (Charles's Nov 20 question): drop projects with 0 or missing
    # capacity and solar/wind projects paired with fossil units (e.g. "Wind+Gas") as out of
    # scope? Effect on validation is nil (LASSO out-of-time 0.686 either way).
    # Team decision, Nov 24: EXCLUDE them (out of scope).
    "exclude_odd_rows": True,
    # Team decision, Nov 24: TabPFN is reported on validation only; the test covers the
    # models that run on the laptop (it is skipped there for lack of a GPU).
    "tabpfn_in_test": False,

    # ---- features
    "feature_extras": ("poi",),
    "feature_columns": [
        "region", "type_clean", "state", "log_mw", "is_hybrid", "q_month",
        "county_entries_3y", "county_mw_3y", "county_built_before", "county_withdrawn_before",
        "county_built_share_before",
        "region_tech_entries_3y", "region_tech_mw_3y", "region_tech_built_before",
        "region_tech_withdrawn_before", "region_tech_built_share_before",
        "poi_kv", "poi_is_line", "poi_is_sub", "poi_entries_3y", "poi_mw_3y", "poi_built_before",
        "poi_withdrawn_before", "poi_built_share_before", "county_has_history",
    ],

    # ---- models: settings as selected by rolling CV on the laptop, Nov 15
    "headline_model": "LASSO logistic",
    "models": {
        "Flat rate (rule of thumb)": {},
        "Historical rate (region x tech)": {},
        "Logistic regression": {},
        "LASSO logistic": {"C": 0.1},
        "CART": {"ccp_alpha": 0.0003},
        "Random forest": {"max_depth": 6, "max_features": 0.25},
        "Gradient boosting": {"learning_rate": 0.03, "max_depth": 4, "max_iter": 500},
        "TabPFN (pre-trained transformer)": {},          # no settings; needs a GPU
    },
    # The time-to-build curve uses its own fixed gradient-boosting settings (timeline.py).
    "timeline_gb": {"learning_rate": 0.03, "max_iter": 200},
    # Opposition second stage (stack.py): first stage = headline model, feature below.
    "stack_opposition_cols": ["any_prior_contested"],

    # ---- checkpoints preflight() reproduces before the test is touched
    "counts": {   # projects per window, by the exclude_odd_rows decision
        False: {"train": 4029, "out_of_time": 3490, "test": 3184},
        True: {"train": 4020, "out_of_time": 3484, "test": 3181},
    },
    "headline_out_of_time_auc": 0.686,   # LASSO, fit 2005-2014, scored 2015-2018
    "auc_tolerance": 0.003,              # library versions move this slightly; more = stop
}
