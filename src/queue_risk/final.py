"""Preflight and logging for the ONE-TIME test run (docs/FINAL_RUN.md).

    python -m queue_risk.final            # dry run: every preflight check, no test scoring

`pipeline --final`, `stack --final` and `timeline --final` call `preflight()` first. It
refuses to start unless:
  1. the frozen spec (frozen.py) has no pending team decision, and a sign-off is given;
  2. the live configuration matches the spec (outcome, horizon, regions, windows, features);
  3. the data reproduce the frozen counts for each window;
  4. the headline model reproduces its frozen out-of-time AUC within tolerance, so the
     environment (library versions) is the one the numbers came from.
Only then is the test cohort scored. Every final run is appended to reports/final_run_log.md.
"""
import subprocess
import sys
from datetime import datetime

from . import config, data, evaluate, features
from .frozen import FROZEN
from .models import selected

LOG = config.ROOT / "reports" / "final_run_log.md"


def _check(label, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    return ok


def preflight(signoff: str | None, need_signoff: bool = True):
    """Run every check; exit with the list of problems if any fails.
    Returns (raw, projects, X, y) for the frozen task, test cohort included but unscored."""
    F = FROZEN
    print("Preflight for the one-time test run (frozen spec of", F["frozen_on"] + ")")
    ok = [
        _check("team decision on odd rows recorded", F["exclude_odd_rows"] is not None,
               "set frozen.FROZEN['exclude_odd_rows'] to True or False first"),
        _check("sign-off given", bool(signoff) or not need_signoff,
               "pass --signoff \"names, date\" once the team has agreed"),
        _check("outcome, horizon and regions match", config.OUTCOME == F["outcome"]
               and config.HORIZON == F["horizon"] and list(config.DATE_REGIONS) == F["regions"]),
        _check("train / test windows match", tuple(config.SPLITS["within"]["train"]) == F["train"]
               and tuple(config.SPLITS["within"]["test"]) == F["test"]),
        _check("feature groups match", tuple(features.DEFAULT_EXTRAS) == tuple(F["feature_extras"])),
    ]
    raw = data.load_raw_queue()
    P = data.built_within(raw, config.HORIZON)
    X, y = features.build_features(raw, P)
    ok.append(_check("feature columns match the frozen list", list(X.columns) == F["feature_columns"]))
    want = F["counts"][bool(F["exclude_odd_rows"])]
    for window in ["train", "out_of_time", "test"]:
        n = int(P["q_year"].between(*F[window]).sum())
        ok.append(_check(f"{window} cohort: {n:,} projects", n == want[window], f"expected {want[window]:,}"))
    tr = P["q_year"].between(*F["train"]).to_numpy()
    oot = P["q_year"].between(*F["out_of_time"]).to_numpy()
    est = selected(F["headline_model"], X[tr]).fit(X[tr], y[tr])
    auc = evaluate.roc_auc_score(y[oot], est.predict_proba(X[oot])[:, 1])
    ok.append(_check(f"{F['headline_model']} reproduces out-of-time AUC {auc:.3f}",
                     abs(auc - F["headline_out_of_time_auc"]) <= F["auc_tolerance"],
                     f"frozen {F['headline_out_of_time_auc']:.3f} +/- {F['auc_tolerance']}"))
    if not all(ok):
        sys.exit("Preflight failed: the test cohort was NOT touched. Fix the items marked FAIL.")
    print("Preflight passed.")
    return raw, P, X, y


def log_run(script: str, signoff: str, lines: list[str]):
    """Append one final run to reports/final_run_log.md (committed, so the record is shared)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=config.ROOT).stdout.strip()
    except OSError:
        commit = "unknown"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", encoding="utf-8") as f:
        if new:
            f.write("# Final (test-set) runs\n\nAppended automatically by `--final`. Do not edit.\n")
        f.write(f"\n## {script}: {datetime.now():%Y-%m-%d %H:%M} | commit {commit} | sign-off: {signoff}\n\n")
        f.writelines(f"- {line}\n" for line in lines)
    print(f"logged to {LOG.relative_to(config.ROOT)}")


if __name__ == "__main__":
    preflight(signoff=None, need_signoff=False)
