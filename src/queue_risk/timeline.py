"""Time-to-build curve: P(operating within n years of entry) for several horizons n.

    python -m queue_risk.timeline            # validation: decide on 2013-2014 entries
    python -m queue_risk.timeline --final    # test: decide on 2019-2020 entries (use once)

One classifier per horizon, each a plain Lecture 6 gradient-boosting model. The rule that
keeps it honest: a fund deciding in year D can only learn the n-year outcome from cohorts
that entered by D - n, so each horizon trains on a different, earlier window. The curve
is then made non-decreasing in n (a project cannot be "less likely" to be built within 7
years than within 5).

Output for the investor, per project: "x% chance of operating within 3 years, y% within 5,
z% within 7". Most projects never reach 50%, so a single "expected completion date" would
mislead; the curve is the honest summary. Horizons whose window has not yet closed for
the evaluation cohort are predicted but cannot be scored, and are reported as such.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import config, data, features, text
from .models import _preprocess
from sklearn.pipeline import Pipeline

# Fixed gradient-boosting settings for every horizon, recorded in the frozen spec.
from .frozen import FROZEN
GB_SETTINGS = dict(**FROZEN["timeline_gb"], random_state=config.SEED)


def _model(X):
    return Pipeline([("prep", _preprocess(X, scale=False)),
                     ("clf", HistGradientBoostingClassifier(**GB_SETTINGS))])


def curve(raw, decide_years, horizons=config.CURVE_HORIZONS):
    """Fit one model per horizon and score the cohorts in `decide_years` (inclusive range).

    Returns (curves, scores): curves has one P(within n) column per horizon for every
    project in the decision cohort; scores reports AUC and predicted vs. actual build
    rate wherever the horizon's window has closed by the data cutoff.
    """
    first, last = decide_years
    cohort = raw[data.in_scope(raw) & raw["region"].isin(config.DATE_REGIONS)
                 & raw["q_year"].between(first, last)]
    opp = text.load_scores()
    Xc, _ = features.build_features(raw, cohort.assign(built=0), opposition=opp)
    probs, rows = {}, []
    for n in horizons:
        lab = data.built_within(raw, n)
        train = lab[lab["q_year"].between(config.FIRST_YEAR, first - n)]
        Xt, yt = features.build_features(raw, train, opposition=opp)
        model = _model(Xt).fit(Xt, yt)
        probs[n] = model.predict_proba(Xc)[:, 1]
        row = {"horizon": n, "trained on": f"{config.FIRST_YEAR}-{first - n} ({len(yt):,})"}
        observed = lab[lab["q_year"].between(first, last)]
        if len(observed) and observed["q_year"].max() == last:
            p = pd.Series(probs[n], index=cohort.index).loc[observed.index]
            row.update({"scored on": f"{len(observed):,}", "AUC": roc_auc_score(observed["built"], p),
                        "predicted rate": p.mean(), "actual rate": observed["built"].mean()})
        else:
            row["scored on"] = "window not closed yet"
        rows.append(row)
    curves = pd.DataFrame(np.maximum.accumulate(np.column_stack([probs[n] for n in horizons]), axis=1),
                          index=cohort.index, columns=[f"p_within_{n}y" for n in horizons])
    return curves, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="decide on the test cohort (use once)")
    ap.add_argument("--signoff", default=None, help='required with --final: "names, date"')
    args = ap.parse_args()
    if args.final:
        from .final import preflight
        preflight(args.signoff)
    years = config.SPLITS["within"]["test"] if args.final else (2013, 2014)
    raw = data.load_raw_queue()
    curves, scores = curve(raw, years)
    print(f"Decision cohort: {years[0]}-{years[1]} entries, {len(curves):,} projects")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(scores.to_string(index=False))
        print("\nAverage predicted curve:", curves.mean().round(3).to_dict())
    if args.final:
        from .final import log_run
        log_run("timeline --final", args.signoff,
                [f"decision cohort {years[0]}-{years[1]}: {len(curves):,} projects"]
                + [", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items())
                   for r in scores.to_dict("records")])


if __name__ == "__main__":
    main()
