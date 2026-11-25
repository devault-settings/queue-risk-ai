"""POST-HOC test scores for models the team left out of the signed-off test run.

    python -m queue_risk.posthoc                       # TabPFN + LASSO, same environment
    python -m queue_risk.posthoc --models "TabPFN,Gradient"

The one-time test (Nov 24, reports/final_run_log.md) was signed off WITHOUT TabPFN,
which needs a GPU. This script answers "how would it have done?" after the fact:

- the frozen task, data and settings only (frozen.py): no tuning, no re-selection;
- LASSO is scored alongside, in the SAME environment, because library versions move
  numbers slightly between machines (on Colab, compare with this LASSO, not the laptop's);
- results go to reports/posthoc_log.md, never to the final-run log, and are labeled
  post-hoc. They don't change the headline model or the reported test results.
"""
import argparse
import subprocess
from datetime import datetime

import numpy as np

from . import config, data, evaluate, features
from .frozen import FROZEN
from .models import model_zoo, selected

LOG = config.ROOT / "reports" / "posthoc_log.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="TabPFN,LASSO", help="comma-separated name fragments")
    args = ap.parse_args()
    wanted = [w.strip().lower() for w in args.models.split(",") if w.strip()]

    raw = data.load_raw_queue()
    P = data.built_within(raw, config.HORIZON)            # frozen odd-row rule applies
    X, y = features.build_features(raw, P)
    assert list(X.columns) == FROZEN["feature_columns"], "features differ from the frozen spec"
    tr = P["q_year"].between(*FROZEN["train"]).to_numpy()
    oot = P["q_year"].between(*FROZEN["out_of_time"]).to_numpy()
    te = P["q_year"].between(*FROZEN["test"]).to_numpy()
    want_n = FROZEN["counts"][bool(FROZEN["exclude_odd_rows"])]
    assert (tr.sum(), te.sum()) == (want_n["train"], want_n["test"]), "cohort counts differ from the frozen spec"
    print(f"POST-HOC (not part of the signed-off test): train {tr.sum():,}, test {te.sum():,} "
          f"({y[te].mean():.1%} built)")

    zoo = model_zoo(X[tr])
    lines = []
    for name in zoo:
        if not any(w in name.lower() for w in wanted):
            continue
        est = selected(name, X[tr]).fit(X[tr], y[tr])
        p_oot, p_te = est.predict_proba(X[oot])[:, 1], est.predict_proba(X[te])[:, 1]
        auc_oot = evaluate.roc_auc_score(y[oot], p_oot)
        auc = evaluate.roc_auc_score(y[te], p_te)
        lo, hi = evaluate.bootstrap_ci(y[te].to_numpy(), p_te, evaluate.roc_auc_score, n=1000, seed=config.SEED)
        yy = y[te].to_numpy()
        k = round(len(yy) * 0.2)
        top = np.argsort(-p_te, kind="stable")[:k]
        lift = yy[top].mean() / yy.mean()
        line = (f"{name}: test AUC {auc:.3f} [{lo:.3f}, {hi:.3f}] | out-of-time 2015-18 {auc_oot:.3f} "
                f"| top 20% built {yy[top].mean():.1%} ({lift:.2f}x average)")
        print("  " + line)
        lines.append(line)
    if not lines:
        raise SystemExit(f"none of {wanted} is available here (TabPFN needs its package and a GPU)")

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=config.ROOT).stdout.strip()
    except OSError:
        commit = "unknown"
    new = not LOG.exists()
    with LOG.open("a", encoding="utf-8") as f:
        if new:
            f.write("# Post-hoc test scores\n\nAfter the signed-off test run of Nov 24 (reports/final_run_log.md). "
                    "These do NOT change the reported results or the headline model.\n")
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | commit {commit} | environment: "
                f"{'GPU' if _has_gpu() else 'CPU'}\n\n")
        f.writelines(f"- {line}\n" for line in lines)
    print(f"logged to {LOG.relative_to(config.ROOT)}")


def _has_gpu():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


if __name__ == "__main__":
    main()
