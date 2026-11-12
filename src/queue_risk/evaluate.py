"""Metrics, from the lectures' evaluation toolkit.

Lecture 3: ROC/AUC is threshold-free; the confusion matrix depends on a threshold.
Lecture 5: a ranking is judged at the decision's capacity (precision in the top k),
           and predictions should add up (calibration: predicted total vs actual total).
Lecture 10-12: "How well does it know when it is wrong?" -> Brier score, reliability.
Bootstrap confidence intervals put an error bar on every test-set number.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def precision_at_k(y, p, k):
    """Share of the k highest-scored projects that were actually built.

    Ties at the cut-off are scored pro rata (Lecture 5 demo: a model that scores a
    block of projects identically cannot rank inside the block), so a flat-rate rule
    returns the base rate instead of whatever order the sort happened to leave.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    k = min(k, len(p))
    cut = np.sort(p)[::-1][k - 1]
    above, tied = p > cut, p == cut
    take_tied = k - above.sum()
    return (y[above].sum() + take_tied * y[tied].mean()) / k


def calibration_ratio(y, p):
    """Predicted number of builds over actual number (1.0 = adds up)."""
    return float(np.sum(p) / max(np.sum(y), 1))


def scorecard(y, p, top_k=(25, 50, 100)) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    row = {"AUC": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
           "PR-AUC": average_precision_score(y, p),
           "Brier": brier_score_loss(y, p),
           "Pred/actual builds": calibration_ratio(y, p)}
    for k in top_k:
        row[f"Prec@{k}"] = precision_at_k(y, p, k)
    return row


def bootstrap_ci(y, p, metric, n=1000, seed=0, alpha=0.05):
    """Percentile bootstrap interval for any metric(y, p)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            stats.append(metric(y[i], p[i]))
    return tuple(np.quantile(stats, [alpha / 2, 1 - alpha / 2]))


def reliability_table(y, p, bins=10) -> pd.DataFrame:
    """Mean predicted vs observed completion rate by predicted-probability decile."""
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
