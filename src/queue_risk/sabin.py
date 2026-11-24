"""Parse the Sabin Center's "Opposition to Renewable Energy Facilities in the United States".

    python -m queue_risk.sabin          # download if missing, parse, match, write label sheet

Source: the annual editions (2022-03 ... 2026-09), downloaded as PDFs to
data/external/sabin/ (gitignored) from the Internet Archive's copies of
scholarship.law.columbia.edu, whose own download links sit behind a bot check.
Editions are cumulative, so the text comes from the newest one; older editions are used
only to date WHEN an entry first became public (a restriction listed in the March 2022
edition was known by end-2021).

Each state section has two parts, parsed into two tables (data/interim/, gitignored):

  restrictions  jurisdiction, rule number, rule type, technology   (structured list)
  contested     project name, counties, the narrative paragraph    (unstructured text)

The contested-project paragraphs are the text the NLP model reads (Lectures 10-12).
Only derived fields are published; the paragraphs stay in data/interim.
"""
import re
from pathlib import Path

import pandas as pd

from . import config

SABIN = config.EXTERNAL / "sabin"
EDITIONS = {"2022-03": 2021, "2023-05": 2022, "2024-06": 2023, "2025-06": 2024, "2026-09": 2025}
# edition -> last calendar year it covers

# Internet Archive copies (raw file, "id_" suffix) of the Columbia scholarship archive PDFs,
# whose own links sit behind a bot check. edition -> (timestamp, article number).
WAYBACK = {"2022-03": ("20250702231136", 1186), "2023-05": ("20230629063110", 1201),
           "2024-06": ("20260708024633", 1227), "2025-06": ("20260910175436", 1252),
           "2026-09": ("20260910070217", 1281)}


def download():
    """Fetch any missing edition PDFs into data/external/sabin/."""
    import urllib.request
    SABIN.mkdir(parents=True, exist_ok=True)
    for ed, (ts, art) in WAYBACK.items():
        out = SABIN / f"sabin_{ed}.pdf"
        if out.exists():
            continue
        url = (f"https://web.archive.org/web/{ts}id_/https://scholarship.law.columbia.edu/"
               f"cgi/viewcontent.cgi?article={art}&context=sabin_climate_change")
        urllib.request.urlretrieve(url, out)
        if not out.read_bytes()[:5] == b"%PDF-":
            out.unlink()
            raise RuntimeError(f"{ed}: the archive returned something other than a PDF")
        print(f"downloaded {out.name}")


STATES = {"ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
          "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
          "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
          "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA",
          "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT",
          "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
          "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
          "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
          "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT",
          "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY"}

STATE_HEADER = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][A-Z ]+?)\s*$")
NOISE = re.compile(r"Sabin Center for Climate Change Law \||^\s*\*|oppositionreport\.org|^\s*\d{1,3}\s*$")
JURISDICTION = re.compile(r"^\s*([A-Z][\w.'’ -]{1,80}?):\s*(Rule\s.*)$")
RULE = re.compile(r"Rule\s*(\d+[a-z]?)\s*:\s*([^|(]+?)\s*\(([^)]+)\)")
ENTRY = re.compile(r"^\s{2,12}([A-Z0-9“\"][^():]{1,160}?)\s\(([^()]{2,250})\):\s*(.*)$")
FOOTNOTE = re.compile(r"(?<=[A-Za-z.,;\"”’)])\d{1,4}(?=[\s,.;]|$)")
YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def edition_text(edition: str) -> list[str]:
    """Lines of an edition's text, extracted once with pdftotext -layout and cached."""
    pdf, txt = SABIN / f"sabin_{edition}.pdf", SABIN / f"sabin_{edition}.txt"
    if not txt.exists():
        import subprocess
        subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True, capture_output=True)
    text = txt.read_text(encoding="utf-8", errors="replace")
    # Some poppler builds keep the list bullets; blank them so the patterns see the same
    # indented text either way (a space per bullet keeps the column positions).
    return text.translate(BULLETS).splitlines()


BULLETS = str.maketrans({"●": " ", "○": " ", "■": " ", "▪": " ", "•": " "})


def state_sections(lines):
    """{state abbreviation: [lines]} for the body (the table of contents is skipped)."""
    sections, current = {}, None
    for line in lines:
        m = STATE_HEADER.match(line)
        if m and m.group(2) in STATES and "...." not in line:
            current = STATES[m.group(2)]
            sections[current] = []
        elif m and m.group(2).startswith("APPENDIX"):
            current = None
        elif current and not NOISE.search(line) and line.strip() not in ("", *[s.title() for s in STATES]):
            sections[current].append(line)
    return sections


def _block(lines, start_pat, end_pat=None):
    out, on = [], False
    for line in lines:
        if re.search(start_pat, line):
            on = True
            continue
        if on and end_pat and re.search(end_pat, line):
            break
        if on:
            out.append(line)
    return out


def parse_restrictions(state, lines):
    rows, cur = [], None
    for line in _block(lines, r"(?i)Type of restrictions:", r"^\s*\d+\.3\s+Contested"):
        m = JURISDICTION.match(line)
        if m:
            cur = [m.group(1).strip(), m.group(2)]
            rows.append(cur)
        elif cur and line.strip():
            cur[1] += " " + line.strip()
    out = []
    for juris, rules in rows:
        for num, kind, tech in RULE.findall(rules):
            for t in re.split(r"\s*(?:,|and|/)\s*", tech):
                if t:
                    out.append({"state": state, "jurisdiction": juris, "rule": num,
                                "rule_type": kind.strip(), "technology": t.strip().lower()})
    return out


def parse_contested(state, lines):
    entries = []
    for line in _block(lines, r"^\s*\d+\.3\s+Contested"):
        m = ENTRY.match(line)
        if m and len(line) - len(line.lstrip()) <= 10:  # 8 in most poppler builds; 10 in some sections of others
            entries.append({"state": state, "project": m.group(1).strip(" “\"”"),
                            "location": m.group(2).strip(), "text": m.group(3).strip()})
        elif entries and line.strip():
            entries[-1]["text"] += " " + line.strip()
    for e in entries:
        e["text"] = FOOTNOTE.sub("", re.sub(r"\s+", " ", e["text"])).strip()
        years = [int(y) for y in YEAR.findall(e["text"])]
        e["first_year"], e["last_year"] = (min(years), max(years)) if years else (None, None)
    return entries


def parse_edition(edition):
    secs = state_sections(edition_text(edition))
    r = [row for s, ls in secs.items() for row in parse_restrictions(s, ls)]
    c = [row for s, ls in secs.items() for row in parse_contested(s, ls)]
    return pd.DataFrame(r), pd.DataFrame(c)


# ------------------------------------------------------------------ county matching

def _norm(name: str) -> str:
    name = re.sub(r"(?i)\b(county|counties|parish|parishes|borough)\b", "", str(name))
    return re.sub(r"[^a-z]", "", name.lower().replace("saint", "st"))


def county_lookup(raw: pd.DataFrame) -> dict:
    """(state, normalized county name) -> FIPS, built from the queue file itself."""
    c = raw.dropna(subset=["fips", "county", "state"])
    pairs = c.groupby(["state", "county"])["fips"].agg(lambda s: s.mode().iat[0])
    return {(st, _norm(co)): f for (st, co), f in pairs.items()}


def counties_in(location: str, state: str, lookup: dict) -> list[str]:
    """FIPS codes for every county named in a location string such as
    'Kern and Kings Counties' or 'City of Madison, Jefferson County'."""
    loc = re.sub(r"(?i)\b(counties)\b", "County", location)
    names = re.findall(r"([A-Z][\w.'’ -]*?)\s+(?:County|Parish)", loc)
    if not names and re.search(r"(?i)counties", location):
        names = re.split(r",|\band\b", re.sub(r"(?i)counties", "", location))
    out = []
    for n in names:
        for part in re.split(r",|\band\b", n):
            f = lookup.get((state, _norm(part)))
            if f and f not in out:
                out.append(f)
    return out


def build(raw: pd.DataFrame):
    """Parse the newest edition, date each entry by the first edition that mentions it,
    and match locations to counties.

    Dating searches each older edition's section for the same state for the entry's
    name. Older editions use other layouts, so a name search is sturdier than a parser
    per layout; the cost is that an entry whose name is garbled in the 2025 PDF (its
    "ti"/"tt"/"ft" ligatures) may be dated a year late.
    """
    lookup = county_lookup(raw)
    eds = [ed for ed in EDITIONS if (SABIN / f"sabin_{ed}.pdf").exists()]
    latest = eds[-1]
    r, c = parse_edition(latest)
    texts = {ed: {st: _norm(" ".join(ls)) for st, ls in state_sections(edition_text(ed)).items()} for ed in eds}

    def known_by(state, name):
        key = _norm(name)
        for ed in eds:
            if key and key in texts[ed].get(state, ""):
                return EDITIONS[ed]
        return EDITIONS[latest]

    r["known_by_year"] = [known_by(s, j) for s, j in zip(r["state"], r["jurisdiction"])]
    r["fips"] = [lookup.get((s, _norm(j))) if re.search(r"(?i)county|parish", j) else None
                 for s, j in zip(r["state"], r["jurisdiction"])]
    is_state = r["jurisdiction"].str.contains(r"(?i)\b(bill|statute|law|code|act|order|regulation|state)\b")
    is_county = r["jurisdiction"].str.contains(r"(?i)county|parish")
    r["level"] = ["state" if st else "county" if co else "municipal" for st, co in zip(is_state, is_county)]
    c["known_by_year"] = [known_by(s, n) for s, n in zip(c["state"], c["project"])]
    c["fips_list"] = [counties_in(l, s, lookup) for l, s in zip(c["location"], c["state"])]
    return r, c


def county_year_signal(r: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """One row per (county FIPS, opposition event): the earliest year the event can be
    dated from the text (contested) or from the edition that first listed it (restrictions)."""
    ev = [{"fips": f, "year": y, "kind": "contested"}
          for fl, y in zip(c["fips_list"], c["first_year"]) if pd.notna(y) for f in fl]
    rr = r.dropna(subset=["fips"]).drop_duplicates(["fips", "jurisdiction"])
    ev += [{"fips": f, "year": y, "kind": "restriction"} for f, y in zip(rr["fips"], rr["known_by_year"])]
    return pd.DataFrame(ev)


def main():
    from . import data
    download()
    raw = data.load_raw_queue()
    r, c = build(raw)
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    # The committed parse is canonical: sabin_labels_all.csv is keyed to it, and pdftotext
    # builds differ slightly across machines (curly vs straight quotes), which would break
    # text.load_paragraphs(). Keep existing files unless explicitly rebuilding.
    import sys
    rebuild = "--rebuild" in sys.argv
    out_r, out_c = config.INTERIM / "sabin_restrictions.csv", config.INTERIM / "sabin_contested.csv"
    if out_c.exists() and not rebuild:
        print(f"{out_c.name} and {out_r.name} already exist (committed parse, keyed to the labels); "
              "kept as is. Pass --rebuild to overwrite, then re-check text.load_paragraphs().")
    else:
        r.to_csv(out_r, index=False)
        c.assign(fips_list=c["fips_list"].map(";".join)).to_csv(out_c, index=False)
    print(f"Restrictions: {len(r):,} rules in {r['jurisdiction'].nunique():,} jurisdictions, "
          f"{r['state'].nunique()} states; county-level {(r['level'] == 'county').mean():.0%}, "
          f"matched to a FIPS {r.loc[r['level'] == 'county', 'fips'].notna().mean():.0%}")
    print(f"Contested projects: {len(c):,} in {c['state'].nunique()} states; matched to >=1 county "
          f"{c['fips_list'].map(bool).mean():.0%}; text dated {c['first_year'].notna().mean():.0%}; "
          f"median {c['text'].str.split().str.len().median():.0f} words")
    print("First listed (restrictions):", r.drop_duplicates(["state", "jurisdiction"])["known_by_year"].value_counts().sort_index().to_dict())
    print("Earliest year in contested text:", c["first_year"].value_counts().sort_index().to_dict())
    sheet = write_label_sheet(c)
    print(f"Rule-based outcome on all {len(c)} paragraphs:", c["text"].map(rule_outcome).value_counts().to_dict())
    print(f"Labeling sheet: {sheet.relative_to(config.ROOT)} (see docs/labeling_guidelines.md)")



# ------------------------------------------------------------------ features and labeling

def contested_events(c: pd.DataFrame) -> pd.DataFrame:
    """(fips, year) for every county a contested project touches, dated by the earliest
    year in its paragraph. That year comes from the text itself, so it reaches back before
    the first edition (2021); restrictions are only dated by edition and start in 2021."""
    rows = []
    for fl, yr in zip(c["fips_list"], c["first_year"]):
        fl = fl.split(";") if isinstance(fl, str) else (fl if isinstance(fl, list) else [])
        rows += [(f, int(yr)) for f in fl if f and pd.notna(yr)]
    return pd.DataFrame(rows, columns=["fips", "year"])


def prior_opposition(projects: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """For each project: contested projects in its county dated STRICTLY before its entry
    year. Strictly before keeps it an entry-time feature (Lecture 8: lags from the past)."""
    by_fips = {f: g["year"].to_numpy() for f, g in events.groupby("fips")}
    n = [int((by_fips.get(f, []) < y).sum()) if isinstance(f, str) else 0
         for f, y in zip(projects["fips"], projects["q_year"])]
    out = pd.DataFrame({"prior_contested": n}, index=projects.index)
    out["any_prior_contested"] = (out["prior_contested"] > 0).astype(int)
    return out


OUTCOME_RULES = [  # (label, keywords) -- the Lecture 10 baseline the trained readers must beat
    ("blocked", r"\b(denied|rejected|withdr[ae]w|cancel+ed|abandon|terminated|scrapped|voted against|blocked)"),
    ("proceeded", r"\b(approved|granted|operational|began operat|under construction|completed|in service)"),
]


def rule_outcome(text: str) -> str:
    """Last outcome keyword in the paragraph wins (paragraphs run chronologically)."""
    hits = [(m.start(), label) for label, pat in OUTCOME_RULES for m in re.finditer(pat, text, re.I)]
    return max(hits)[1] if hits else "unresolved"


def write_label_sheet(c: pd.DataFrame, n=300, seed=config.SEED):
    """A labeling sheet of n contested-project paragraphs, sampled across years.
    Stays in data/interim (gitignored): the paragraphs are the report's text.
    Never overwrites an existing sheet, which may hold hand labels; delete it to rebuild."""
    out = config.INTERIM / "sabin_label_sheet.csv"
    if out.exists():
        print(f"{out.name} already exists; kept as is (delete it to rebuild)")
        return out
    s = c.sample(n=min(n, len(c)), random_state=seed).sort_values(["state", "project"])
    sheet = s[["state", "project", "location", "first_year", "text"]].copy()
    sheet.insert(0, "id", s.index)
    sheet["rule_suggestion"] = sheet["text"].map(rule_outcome)
    sheet["outcome"] = ""          # blocked / delayed / proceeded / unresolved
    sheet["concerns"] = ""         # semicolon list from the guidelines
    sheet["notes"] = ""
    sheet.to_csv(out, index=False, encoding="utf-8-sig")   # utf-8-sig opens cleanly in Excel
    return out


if __name__ == "__main__":
    main()
