"""Shared paths, constants and assumptions. Change things here, in one place, for everyone.

Every business assumption the impact analysis uses lives at the bottom of this file,
the way the class demos keep theirs in one block, so the room can argue with them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
DEMO = ROOT / "demo"

QUEUE_XLSX = RAW / "LBNL_Ix_Queue_Data_File_thru2025.xlsx"
QUEUE_SHEET = "03. Complete Queue Data"
OPPOSITION_CSV = PROCESSED / "opposition_scores.csv"

SEED = 15071

# ------------------------------------------------------------------ the task
# Technologies in scope: any project whose primary type (type_1) is one of these.
CLEAN_TYPES = ["Solar", "Wind", "Battery", "Offshore Wind"]

# Outcome: operational = 1 (built), withdrawn = 0. Active/suspended are unresolved.
RESOLVED_STATUSES = ["operational", "withdrawn"]

# Two ways to define the outcome (README "Two outcomes"):
#   "within"  built within HORIZON years of entry. Every project gets a label once its
#             window has closed (still active or withdrawn = not built in time), so
#             nothing is censored and the test cohort can be recent. DEFAULT.
#   "ever"    ever built, among resolved projects only. Recent cohorts are censored
#             (withdrawals resolve faster than builds), so the test window stops at 2017.
OUTCOME = "within"
HORIZON = 5                       # years from queue entry
CURVE_HORIZONS = [2, 3, 4, 5, 7]  # for the time-to-build curve (timeline.py)
DATA_CUTOFF = "2025-12-31"        # the file's "as of" date

# The "within" outcome needs a completion date. These six operators report one for
# 96% of built projects; ISO-NE reports none, the non-ISO West and Southeast few.
DATE_REGIONS = ["CAISO", "ERCOT", "MISO", "NYISO", "PJM", "SPP"]

# ------------------------------------------------------------------ the split
# Temporal split, as in Lecture 6 (wildfires: 2015-2022 train / 2023-2024 test).
FIRST_YEAR = 2005
SPLITS = {
    # Training labels for 2014 entries close in 2019, so a fund deciding on 2019 entries
    # could have built exactly this model: the train/test gap equals HORIZON.
    "within": {"train": (2005, 2014), "test": (2019, 2020)},
    "ever":   {"train": (2005, 2014), "test": (2015, 2017)},
}
TRAIN_YEARS = SPLITS[OUTCOME]["train"]
TEST_YEARS = SPLITS[OUTCOME]["test"]

# Rolling-origin validation inside the training window (Lecture 4 cross-validation,
# made temporal): for each year t below, fit on entries before t, validate on t.
CV_VAL_YEARS = [2010, 2011, 2012, 2013, 2014]

# ------------------------------------------------------------------ leakage
# Columns only known AFTER queue entry. Never use them as features (Lecture 8:
# lag variables "must always be built using past data").
LEAKAGE_COLS = [
    "q_status",        # the label itself
    "on_date",
    "wd_date",
    "ia_date",
    "IA_phase_raw",
    "IA_phase_clean",
    "prop_date",       # revisable after entry
    "prop_year",
]

# Interface: opposition score table (NLP lead -> modeling lead).
OPPOSITION_COLS = ["fips", "year", "n_articles", "share_opposition"]

# ------------------------------------------------------------------ the decision
# The investor backs the top-K projects in a cohort (Lecture 5: a fixed-capacity
# worklist). These K values are the ones the evaluation reports precision at.
TOP_K = [25, 50, 100]

# Pre-construction development spend per project that withdraws, in USD.
# PLACEHOLDER RANGE, NOT SOURCED. The impact lead replaces these with cited figures;
# every chart is drawn across the range, never at one number.
DEV_COST_RANGE_USD = (1_000_000, 3_000_000, 10_000_000)

# ------------------------------------------------------------------ back early vs. wait
# Indicative prices for development rights, USD per watt (low, mid, high). Industry
# benchmarks, not observed trades: Ascend Analytics, "Increasing Early-Stage Project
# Success in a Highly Competitive Renewable Energy M&A Landscape" (solar+storage, US ISOs):
# early stage $0.02-0.05/W, late stage $0.07-0.20/W. Cross-check (unsourced banker guide):
# <$50k/MW early, $80-150k/MW in interconnection, $150-350k/MW with IA + PPA.
# Early floor (SEC filing): Bimergen Energy Corp, Form S-1/A (Aug 7, 2025), 2,425 MW of
# greenfield solar sold at $5,000/MW + $3,000/MW milestone payments = $0.008/W.
# Greenfield is earlier than queue entry, so this is a floor, below our low.
PRICE_EARLY_USD_PER_W = (0.02, 0.035, 0.05)
# Late support (SEC filings): a 2015 purchase agreement, 72.9 MW at $0.08/W; Dominion
# Energy 2016 10-K, $32M for ~221 MW (~$0.145/W). Queue stage of both TO CONFIRM.
PRICE_LATE_USD_PER_W = (0.07, 0.135, 0.20)
# Development spend the early backer funds between entry and the IA, per 100 MW project
# (low, mid, high) = studies + permitting + land:
#   studies ~$150k: FERC Order 2023 study deposit tier for 80-200 MW (LGIP 3.1.1.1);
#   permitting $250k / $300k / $500k: NREL benchmarks 2016/2018; mid is NREL 2018:
#     $295k, rounded (TO UPDATE from the 2025 Q1 benchmark); CA is 9.5% of 2015-2020
#     projects;
#   land $0 / ~$1M / $3M: low and mid are ASSUMPTIONS (mid = land option), high is
#     NREL ($0.03/W).
# Deposits (commercial readiness) are tied-up money, not spend, so they are excluded.
SPEND_TO_IA_USD = (400_000, 1_450_000, 3_650_000)
# Annual cost of capital, to carry the early outlay to the IA date (median 2.6 years
# after entry in our six operators). NREL ATB 2025, Financial Cases & Methods: nominal
# after-tax WACC 5.6-6.5% for renewables (low, 6%); construction equity return 10.0%
# (mid). High (15%) is an ASSUMPTION for day-one queue risk, above both.
COST_OF_CAPITAL = (0.06, 0.10, 0.15)
