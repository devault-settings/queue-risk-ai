"""The project sponsor's view: which choices are associated with better odds?

    python -m queue_risk.sponsor --region ERCOT --tech Solar --mw 200

Two tools for a developer planning a new queue request:

- what_if(): one planned project, varied one choice at a time (size, technology,
  hybrid pairing), each scored by the 5-year model.
- siting_screen(): the same planned project placed in every county of a region,
  ranked by predicted P(operating within 5 years). County effects enter only through
  the track-record features (past queue activity and outcomes there, as of the entry date).

READ THIS BEFORE QUOTING ANY OUTPUT. These are ASSOCIATIONS learned from history,
not causal effects (Lecture 6: "correlation vs. causation"; Lecture 5 demo: "nothing here
says enrolling a patient reduces their cost"). A county where past projects got built
may differ in transmission headroom, land, or permitting in ways the model never sees;
moving a project there does not import those conditions. Use the screen to decide
where to look harder, not where to build.
"""
import argparse

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from . import config, data, features
from .models import _preprocess


def fit_deployment_model(raw):
    """5-year model on every cohort whose window has closed (the most recent possible)."""
    lab = data.built_within(raw, config.HORIZON)
    X, y = features.build_features(raw, lab)
    model = Pipeline([("prep", _preprocess(X, scale=False)),
                      ("clf", HistGradientBoostingClassifier(learning_rate=0.03, max_iter=200,
                                                             random_state=config.SEED))]).fit(X, y)
    return model, lab


def _planned(region, tech, mw, fips, state, q_date, hybrid=False, service="NRIS"):
    return {"project_id": "planned", "region": region, "type_1": tech,
            "type_clean": f"{tech}+Battery" if hybrid else tech, "service": service,
            "state": state, "fips": fips, "q_date": pd.Timestamp(q_date),
            "q_year": pd.Timestamp(q_date).year, "mw_total": float(mw), "is_hybrid": hybrid, "built": 0}


def score(raw, model, rows):
    X, _ = features.build_features(raw, pd.DataFrame(rows))
    return model.predict_proba(X)[:, 1]


def siting_screen(raw, model, region, tech, mw, q_date=config.DATA_CUTOFF, min_history=3):
    """Rank counties in `region` for a planned project. Only counties with at least
    `min_history` past queue projects are scored (no history = no basis for a score)."""
    r = raw[(raw["region"] == region) & raw["fips"].notna()]
    counties = (r.groupby("fips").agg(county=("county", "first"), state=("state", "first"),
                                     past_projects=("project_id", "size"))
                 .query("past_projects >= @min_history").reset_index())
    rows = [_planned(region, tech, mw, c.fips, c.state, q_date) for c in counties.itertuples()]
    counties["p_within_5y"] = score(raw, model, rows)
    return counties.sort_values("p_within_5y", ascending=False)


def what_if(raw, model, base: dict, options: dict) -> pd.DataFrame:
    """Score `base` and each single-choice variant: options = {"mw_total": [50, 200], ...}."""
    rows, labels = [base], ["as planned"]
    for key, values in options.items():
        for v in values:
            row = dict(base, **{key: v})
            if key == "is_hybrid":
                row["type_clean"] = f"{base['type_1']}+Battery" if v else base["type_1"]
            rows.append(row)
            labels.append(f"{key} = {v}")
    return pd.DataFrame({"variant": labels, "p_within_5y": score(raw, model, rows)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="ERCOT")
    ap.add_argument("--tech", default="Solar")
    ap.add_argument("--mw", type=float, default=200)
    args = ap.parse_args()
    raw = data.load_raw_queue()
    model, lab = fit_deployment_model(raw)
    print(f"Deployment model: {len(lab):,} projects, cohorts {lab['q_year'].min()}-{lab['q_year'].max()}")
    screen = siting_screen(raw, model, args.region, args.tech, args.mw)
    print(f"\nSiting screen: {args.mw:.0f} MW {args.tech} in {args.region}, {len(screen)} counties scored "
          "(associations, not causal effects)")
    with pd.option_context("display.width", 160, "display.precision", 3):
        print(screen.head(8).to_string(index=False))
        top = screen.iloc[0]
        base = _planned(args.region, args.tech, args.mw, top.fips, top.state, config.DATA_CUTOFF)
        print("\nWhat-if, in the top county:")
        print(what_if(raw, model, base, {"mw_total": [50, 100, 400], "is_hybrid": [True]}).to_string(index=False))


if __name__ == "__main__":
    main()
