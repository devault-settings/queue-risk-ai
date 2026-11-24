"""Does local opposition add anything beyond the tabular model? A two-stage (stacked) test.

    python -m queue_risk.stack                       # validation: cohorts 2015-2018, both stage-1 models
    python -m queue_risk.stack --stage1 lasso        # one stage-1 model only (lasso, gb, tabpfn)
    python -m queue_risk.stack --final               # test: fit stage 2 on 2015-2018, score 2019-2020 (once)

Why two stages: the Sabin opposition record reaches back to ~2010 only thinly, so it can't
be a feature of the stage-1 model trained on 2005-2014 entries. Instead (Lecture 7,
stacking with out-of-fold predictions):

  stage 1  a tabular model at its pipeline-selected settings (LASSO or gradient boosting),
           fit on 2005-2014 entries, tabular features only (frozen)
  stage 2  logistic regression on [stage-1 log-odds, opposition features], fit on the
           2015-2018 cohorts, scored leave-one-year-out so every validation prediction
           comes from a stage-2 model that never saw that year

The comparison is stage 2 with vs. without the opposition features, so the only thing
that differs is the text-derived signal. Bootstrap gives an interval on the AUC gain.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config, data, features, sabin
from .models import selected

STAGE2_YEARS = (2015, 2018)
OPP_COLS = ["any_prior_contested"]


STAGE1 = {"lasso": "LASSO logistic", "gb": "Gradient boosting",
          "tabpfn": "TabPFN (pre-trained transformer)"}   # tabpfn: GPU (Colab) only


def _stage1(raw, projects, model="gb"):
    X, y = features.build_features(raw, projects)
    tr = projects["q_year"].between(*config.SPLITS["within"]["train"]).to_numpy()
    est = selected(STAGE1[model], X[tr])
    est.fit(X[tr], y[tr])
    p = np.clip(est.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return pd.Series(np.log(p / (1 - p)), index=projects.index), y


def leave_one_year_out(V, cols):
    oof = np.zeros(len(V))
    for yr in sorted(V["q_year"].unique()):
        fit, held = (V["q_year"] != yr).to_numpy(), (V["q_year"] == yr).to_numpy()
        m = LogisticRegression(max_iter=1000).fit(V.loc[fit, cols], V["y"][fit])
        oof[held] = m.predict_proba(V.loc[held, cols])[:, 1]
    return oof


def bootstrap_gain(y, a, b, n=1000, seed=0):
    rng, d = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    d = np.array(d)
    return d.mean(), np.quantile(d, 0.025), np.quantile(d, 0.975), (d > 0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--signoff", default=None, help='required with --final: "names, date"')
    ap.add_argument("--stage1", choices=["lasso", "gb", "tabpfn", "both"], default="both",
                    help="both = lasso and gb; tabpfn needs the tabpfn package and a GPU")
    args = ap.parse_args()
    if args.final:
        from .final import preflight
        from .frozen import FROZEN
        preflight(args.signoff)
        # The frozen first stage is the headline model; the other stage-1 models stay validation-only.
        args.stage1 = {name: key for key, name in STAGE1.items()}[FROZEN["headline_model"]]
    raw = data.load_raw_queue()
    P = data.built_within(raw, config.HORIZON)
    # Paragraph dates from text.load_paragraphs() (Paul, Nov 18): re-dated from the cleaned
    # labeling text, without the footnote citations sabin_contested.csv had glued on
    # (first_year changes for 123 paragraphs). Same county lists.
    from . import text
    opp = sabin.prior_opposition(P, sabin.contested_events(text.load_paragraphs()))
    for s1 in (["lasso", "gb"] if args.stage1 == "both" else [args.stage1]):
        logit, y = _stage1(raw, P, s1)
        V = P[["q_year", "fips"]].assign(logit=logit, y=y).join(opp)
        fit = V[V["q_year"].between(*STAGE2_YEARS)]
        print(f"\n=== stage 1: {STAGE1[s1]} (pipeline-selected settings)")
        if not args.final:
            a, b = leave_one_year_out(fit, ["logit"]), leave_one_year_out(fit, ["logit"] + OPP_COLS)
            yy = fit["y"].to_numpy()
            coef = LogisticRegression(max_iter=1000).fit(fit[["logit"] + OPP_COLS], fit["y"]).coef_[0]
            print(f"Validation, cohorts {STAGE2_YEARS[0]}-{STAGE2_YEARS[1]} ({len(fit):,} projects, "
                  f"{fit['any_prior_contested'].mean():.0%} in a county with earlier contested projects)")
            print(f"  stage-1 model alone, AUC {roc_auc_score(yy, fit['logit']):.3f}")
            print(f"  stage 2 out-of-fold AUC: tabular only {roc_auc_score(yy, a):.3f} | + opposition {roc_auc_score(yy, b):.3f}")
            print("  stage-2 coefficients:", dict(zip(["logit"] + OPP_COLS, coef.round(2))),
                  f"-> odds x{np.exp(coef[1]):.2f} for a county with earlier contested projects")
            g = bootstrap_gain(yy, a, b)
            print(f"  AUC gain {g[0]:+.3f}, 95% CI [{g[1]:+.3f}, {g[2]:+.3f}], positive in {g[3]:.0%} of draws")
            print("  by year:", {yr: (round(roc_auc_score(yy[m], a[m]), 3), round(roc_auc_score(yy[m], b[m]), 3))
                                 for yr in range(STAGE2_YEARS[0], STAGE2_YEARS[1] + 1)
                                 for m in [(fit["q_year"] == yr).to_numpy()]})
        else:
            test = V[V["q_year"].between(*config.SPLITS["within"]["test"])]
            out = {}
            for name, cols in [("tabular only", ["logit"]), ("+ opposition", ["logit"] + OPP_COLS)]:
                m = LogisticRegression(max_iter=1000).fit(fit[cols], fit["y"])
                out[name] = m.predict_proba(test[cols])[:, 1]
            yy = test["y"].to_numpy()
            g = bootstrap_gain(yy, out["tabular only"], out["+ opposition"])
            print({k: round(roc_auc_score(yy, v), 3) for k, v in out.items()},
                  f"gain {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}]")


if __name__ == "__main__":
    main()
