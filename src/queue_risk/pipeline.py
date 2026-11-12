"""Run the whole tabular pipeline and print the benchmark table.

    python -m queue_risk.pipeline            # model selection on rolling validation only
    python -m queue_risk.pipeline --final    # + one-time test-set "showtime" (Lecture 4)
    python -m queue_risk.pipeline --fast     # small grids, for a quick end-to-end check
    python -m queue_risk.pipeline --outcome ever   # the "ever built" definition instead

Model selection never touches the test cohort. --final refits each model's selected
settings on the whole training window and scores the test cohort once, with bootstrap
95% intervals and the impact sweep.
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from . import config, data, evaluate, features, impact, text
from .models import model_zoo
from .split import rolling_folds, temporal_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="score the held-out test cohort")
    ap.add_argument("--fast", action="store_true", help="first value of every grid only")
    ap.add_argument("--outcome", choices=["within", "ever"], default=config.OUTCOME)
    args = ap.parse_args()
    split = config.SPLITS[args.outcome]

    raw = data.load_raw_queue()
    projects = data.modeling_table(raw, args.outcome)
    X, y = features.build_features(raw, projects, opposition=text.load_scores())
    train, test = temporal_split(projects["q_year"], split["train"], split["test"])
    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    folds = rolling_folds(projects["q_year"][train])
    label = f"built within {config.HORIZON}y" if args.outcome == "within" else "ever built"
    print(f"outcome: {label} | train {split['train']}: {len(ytr):,} projects, {ytr.mean():.1%} | "
          f"test {split['test']}: {len(yte):,} projects, {yte.mean():.1%} | "
          f"{X.shape[1]} features, {len(folds)} rolling folds")

    rows, test_scores = [], {}
    for name, (est, grid) in model_zoo(Xtr).items():
        if args.fast:
            grid = {k: v[:1] for k, v in grid.items()}
        t0 = time.time()
        search = GridSearchCV(est, grid or {}, cv=folds, scoring="roc_auc", n_jobs=1).fit(Xtr, ytr)
        row = {"model": name, "CV AUC": search.best_score_,
               "settings": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
               "secs": round(time.time() - t0, 1)}
        if args.final:
            p = search.best_estimator_.predict_proba(Xte)[:, 1]
            test_scores[name] = p
            row.update({f"test {k}": v for k, v in evaluate.scorecard(yte, p, config.TOP_K).items()})
            lo, hi = evaluate.bootstrap_ci(yte.to_numpy(), p, evaluate.roc_auc_score, n=300)
            row["test AUC 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
        print(f"  {name:32s} CV AUC {search.best_score_:.3f}  ({row['secs']}s)")

    table = pd.DataFrame(rows)
    config.FIGURES.parent.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES.parent / f"results_{args.outcome}_{'test' if args.final else 'validation'}.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(table.drop(columns="secs").to_string(index=False))
    print(f"wrote {out.relative_to(config.ROOT)}")

    if args.final:
        mw = projects.loc[test, "mw_total"].to_numpy()
        test_scores["Perfect hindsight"] = yte.to_numpy().astype(float)
        cost = config.DEV_COST_RANGE_USD[1]
        curve = impact.sweep(yte, mw, test_scores, ks=config.TOP_K, cost_per_project=cost)
        print(f"\nWasted development spend per built MW, at ${cost:,.0f}/project (placeholder):")
        print(curve.pivot(index="rule", columns="k", values="wasted_per_built_mw").round(0).to_string())


if __name__ == "__main__":
    main()
