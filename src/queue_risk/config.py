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

# ------------------------------------------------------------------ the split
# Temporal split, as in Lecture 6 (wildfires: 2015-2022 train / 2023-2024 test).
# PROPOSAL, pending the data lead: recent cohorts are right-censored (builds take
# longer to resolve than withdrawals), so the test window stops at 2017, where
# >= 84% of entries are resolved. See README "Known issues".
FIRST_YEAR = 2005
TRAIN_YEARS = (2005, 2014)
TEST_YEARS = (2015, 2017)

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
