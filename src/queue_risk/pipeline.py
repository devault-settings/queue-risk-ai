"""Run the whole tabular pipeline and print the benchmark table.

    python -m queue_risk.pipeline            # model selection on rolling validation only
    python -m queue_risk.pipeline --final --signoff "names, date"   # one-time test (docs/FINAL_RUN.md)
    python -m queue_risk.pipeline --fast     # small grids, for a quick end-to-end check
    python -m queue_risk.pipeline --outcome ever   # the "ever built" definition instead
    python -m queue_risk.pipeline --models "Gradient,TabPFN" --ablation
        # selected models only, each with and without the news-opposition features

Model selection never touches the test cohort. --final does NOT re-select anything: after
final.preflight() passes, it fits each model at its FROZEN settings (frozen.py) on the
training window and scores the test cohort once, with bootstrap 95% intervals and the
impact sweep across the sourced cost range. The run is logged to reports/final_run_log.md.
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from . import config, data, evaluate, features, impact, text
from .models import model_zoo
from .split import rolling_folds, temporal_split

TEXT_COLS = ["n_articles", "share_opposition", "news_coverage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="score the held-out test cohort (frozen spec)")
    ap.add_argument("--signoff", default=None, help='required with --final: "names, date"')
    ap.add_argument("--fast", action="store_true", help="first value of every grid only")
    ap.add_argument("--outcome", choices=["within", "ever"], default=config.OUTCOME)
    ap.add_argument("--models", default="", help="comma-separated name fragments, e.g. 'Gradient,TabPFN'")
    ap.add_argument("--ablation", action="store_true",
                    help="also run every model WITHOUT the text features (what does the text model add?)")
    args = ap.parse_args()
    if args.final:
        return run_final(args)
    split = config.SPLITS[args.outcome]

    raw = data.load_raw_queue()
    projects = data.modeling_table(raw, args.outcome)
    X, y = features.build_features(raw, projects, opposition=text.load_scores())
    train, test = temporal_split(projects["q_year"], split["train"], split["test"])
    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    folds = rolling_folds(projects["q_year"][train])
    # Out-of-time band between the training window and the test cohort (validation, not test):
    # each SELECTED model is refit on the training window and scored here, so the validation
    # and out-of-time numbers always describe the same configuration.
    oot = projects["q_year"].between(split["train"][1] + 1, split["test"][0] - 1).to_numpy()
    Xoot, yoot = X[oot], y[oot]
    label = f"built within {config.HORIZON}y" if args.outcome == "within" else "ever built"
    print(f"outcome: {label} | train {split['train']}: {len(ytr):,} projects, {ytr.mean():.1%} | "
          f"test {split['test']}: {len(yte):,} projects, {yte.mean():.1%} | "
          f"{X.shape[1]} features, {len(folds)} rolling folds")

    feature_sets = {"all features": list(X.columns)}
    if args.ablation:
        if not set(TEXT_COLS) & set(X.columns):
            print("--ablation: no opposition score table yet (data/processed/opposition_scores.csv); "
                  "running without text features only")
        else:
            feature_sets["without text"] = [c for c in X.columns if c not in TEXT_COLS]
    wanted = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    jobs = [(fs, cols, name, est, grid)
            for fs, cols in feature_sets.items()
            for name, (est, grid) in model_zoo(Xtr[cols]).items()
            if not wanted or any(w in name.lower() for w in wanted)]

    rows, test_scores = [], {}
    for fs, cols, name, est, grid in jobs:
        if args.fast:
            grid = {k: v[:1] for k, v in grid.items()}
        t0 = time.time()
        search = GridSearchCV(est, grid or {}, cv=folds, scoring="roc_auc", n_jobs=1).fit(Xtr[cols], ytr)
        label = name if len(feature_sets) == 1 else f"{name} [{fs}]"
        row = {"model": label, "CV AUC": search.best_score_,
               "settings": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
               "secs": round(time.time() - t0, 1)}
        if oot.any() and yoot.nunique() > 1:
            p_oot = search.best_estimator_.predict_proba(Xoot[cols])[:, 1]
            row[f"out-of-time AUC {split['train'][1] + 1}-{split['test'][0] - 1}"] = evaluate.roc_auc_score(yoot, p_oot)
        rows.append(row)
        oot_txt = next((f" | out-of-time {v:.3f}" for k, v in row.items() if k.startswith("out-of-time")), "")
        print(f"  {label:48s} CV AUC {search.best_score_:.3f}{oot_txt}  ({row['secs']}s)")

    table = pd.DataFrame(rows)
    config.FIGURES.parent.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES.parent / f"results_{args.outcome}_validation.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(table.drop(columns="secs").to_string(index=False))
    print(f"wrote {out.relative_to(config.ROOT)}")



def run_final(args):
    """The one-time test: frozen models, preflight first, every model scored once."""
    from .final import log_run, preflight
    from .frozen import FROZEN
    from .models import selected
    if args.outcome != "within" or args.ablation or args.fast or args.models:
        raise SystemExit("--final runs the frozen spec only: no --outcome ever, --ablation, --fast or --models.")
    raw, P, X, y = preflight(args.signoff)
    tr = P["q_year"].between(*FROZEN["train"]).to_numpy()
    te = P["q_year"].between(*FROZEN["test"]).to_numpy()
    zoo = model_zoo(X[tr])
    rows, scores, lines = [], {}, []
    for name in FROZEN["models"]:
        if name not in zoo:
            print(f"  {name}: not available here (e.g. TabPFN needs its package and a GPU), skipped")
            lines.append(f"{name}: skipped (not available in this environment)")
            continue
        p = selected(name, X[tr]).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        scores[name] = p
        card = evaluate.scorecard(y[te], p, config.TOP_K)
        lo, hi = evaluate.bootstrap_ci(y[te].to_numpy(), p, evaluate.roc_auc_score, n=1000, seed=config.SEED)
        rows.append({"model": name, **{f"test {k}": v for k, v in card.items()}, "test AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]"})
        lines.append(f"{name}: test AUC {card['AUC']:.3f} [{lo:.3f}, {hi:.3f}], PR-AUC {card['PR-AUC']:.3f}")
        print(f"  {name:36s} test AUC {card['AUC']:.3f} [{lo:.3f}, {hi:.3f}]")
    table = pd.DataFrame(rows)
    out = config.ROOT / "reports" / "results_within_test.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(table.to_string(index=False))
    mw = P.loc[te, "mw_total"].fillna(0).to_numpy()
    scores["Perfect hindsight"] = y[te].to_numpy().astype(float)
    for cost in config.DEV_COST_RANGE_USD:
        curve = impact.sweep(y[te], mw, scores, ks=config.TOP_K, cost_per_project=cost)
        print(f"\nWasted development spend per built MW at ${cost:,.0f} per withdrawn project:")
        print(curve.pivot(index="rule", columns="k", values="wasted_per_built_mw").round(0).to_string())
    log_run("pipeline --final", args.signoff, lines + [f"table: {out.relative_to(config.ROOT)}"])


if __name__ == "__main__":
    main()
