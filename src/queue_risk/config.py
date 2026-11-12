"""Shared paths and constants. Change split years here, in one place, for everyone."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

QUEUE_XLSX = RAW / "LBNL_Ix_Queue_Data_File_thru2025.xlsx"
QUEUE_SHEET = "03. Complete Queue Data"

SEED = 15071

# Technologies in scope: any project whose primary type (type_1) is one of these.
CLEAN_TYPES = ["Solar", "Wind", "Battery", "Offshore Wind"]

# Outcome classes: operational = 1, withdrawn = 0; everything else is unresolved.
RESOLVED_STATUSES = ["operational", "withdrawn"]

# Time split. Placeholder until the data lead fixes it (see README "Known issues").
TRAIN_YEARS = (2005, 2016)
TEST_YEARS = (2017, 2019)

# Columns that are only known AFTER queue entry. Never use them as features.
LEAKAGE_COLS = [
    "q_status",        # the label itself
    "on_date",
    "wd_date",
    "ia_date",
    "IA_phase_raw",
    "IA_phase_clean",
    "prop_date",       # can be revised after entry; use with care, if at all
    "prop_year",
]

# Interface: opposition score table (NLP lead -> modeling lead)
OPPOSITION_COLS = ["fips", "year", "n_articles", "share_opposition"]
