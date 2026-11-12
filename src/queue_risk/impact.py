"""From predictions to decisions: what a ranking is worth to the investor.

The decision (Lecture 5's worklist): from a cohort of new queue entries, an infrastructure
fund backs K projects, each of which costs development capital before it is known whether
it will be built. Projects that withdraw waste that spend.

Headline metric: wasted development spend per built MW
    = (cost per project x number of backed projects that withdrew) / (MW of backed projects built)

Lecture 2/3/5 pattern: compare the same arithmetic across ranking rules (a rule of thumb,
a historical baseline, the model, and perfect hindsight as the ceiling), and sweep the
decision variable (K) and the uncertain assumption (cost per project) instead of
quoting one number.
"""
import numpy as np
import pandas as pd


def backed(p, k):
    """Indices of the k highest-scored projects; ties broken at random but reproducibly."""
    p = np.asarray(p, dtype=float)
    jitter = np.random.default_rng(0).random(len(p)) * 1e-12
    return np.argsort(-(p + jitter))[:k]


def portfolio(y, mw, p, k, cost_per_project):
    """Outcome of backing the top-k projects by score p."""
    y, mw = np.asarray(y), np.asarray(mw, dtype=float)
    idx = backed(p, k)
    built = y[idx] == 1
    mw_built = mw[idx][built].sum()
    n_withdrawn = int((~built).sum())
    wasted = cost_per_project * n_withdrawn
    return {"k": k, "built": int(built.sum()), "withdrawn": n_withdrawn, "mw_built": mw_built,
            "wasted_usd": wasted,
            "wasted_per_built_mw": wasted / mw_built if mw_built > 0 else np.inf}


def sweep(y, mw, scores: dict, ks, cost_per_project) -> pd.DataFrame:
    """One row per (rule, k): the curve of wasted spend per built MW vs. projects backed."""
    rows = [{"rule": name, **portfolio(y, mw, p, k, cost_per_project)}
            for name, p in scores.items() for k in ks]
    return pd.DataFrame(rows)
