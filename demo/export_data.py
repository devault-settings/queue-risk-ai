"""Build `demo/data.js` for the demo page (a simpler first version of the Deal Desk).

    python demo/export_data.py

Same contract as the class demos' export scripts (Lectures 5, 6, 12):

1. Rebuild everything from the raw file with the pipeline's own code (no copied numbers).
2. Fit the headline model, LASSO at the settings model selection chose (`models.selected`),
   on 2005-2014 entries only, and score the 2015-2018 entries out of time.
3. Check itself against the numbers in docs/model_log.md and REFUSE to write data.js if
   anything has moved (CHECKPOINTS below).
4. Write only what the page reads: N_SAMPLE projects drawn at random (seed SAMPLE_SEED) from the
   2015-2018 entries, the full-cohort comparison, and the facts the header and footer print.

Tabular model only: no Sabin text or opposition signal on this page.
The 2019-2020 test cohort is never loaded into the feature table or scored here.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from queue_risk import config, data, evaluate, features  # noqa: E402
from queue_risk.models import selected  # noqa: E402

MODEL = "LASSO logistic"
SCORE_YEARS = (2015, 2018)          # the out-of-time band, between training and test
N_SAMPLE, TOP = 100, 25
# History of the draw, so nobody re-picks it:
# - The first 50-project draw (seed config.SEED) was luckier for the model than 95% of random
#   draws (top 10 held 4 builds vs 1.2 at random; typical is 1.7 vs 0.9). Before looking at any
#   other draw, we committed to the next seed and to keeping whatever it gives. Do not re-pick it.
# - Nov 24: the pool dropped projects with 0 or negative MW (MIN_MW below). With the same seed,
#   that 50-project draw had only 1 build (ranked 14th). A 50-project sample is too noisy, so
#   we moved to 100 projects / top 25 (same 25% share), same seed and pool, and kept the result.
# - Nov 25: merged main's frozen spec, which drops odd rows (0 or negative MW, fossil-paired) from
#   the cohort: 3,490 -> 3,484. Same seed, same rules; the 100-project draw changed with the data
#   (11 builds -> 6) and we kept it. Not a re-pick.
SAMPLE_SEED = config.SEED + 1
# Projects that would print as "0.0 MW" or less on the page are left out of the pool the sample
# is drawn from: 20 PJM 5 kW (0.005 MW) batteries, none built (the 3 negative-MW ERCOT entries
# are now dropped earlier, as odd rows). Rule fixed before looking at the new draw; the seed above
# is unchanged. The model, AUC check and full-cohort comparison use the whole cohort (3,484).
MIN_MW = 0.05
SIM_DRAWS = 20_000                  # redraws for the "how typical is this sample" line
OUT = Path(__file__).resolve().parent / "data.js"

# Read from the frozen spec (src/queue_risk/frozen.py), so the demo follows the team's decisions.
# Nov 24: odd rows (0 MW, fossil-paired) are excluded, so 2015-2018 has 3,484 projects, not
# the 3,490 of Nov 20. LASSO's out-of-time AUC is 0.686 either way. Because the cohort changed,
# the seed-15072 draw of 100 projects changes too; that follows from the data decision, not a re-pick.
# If a rerun disagrees, nothing is written.
from queue_risk.frozen import FROZEN  # noqa: E402

CHECKPOINTS = {
    "LASSO out-of-time AUC 2015-2018": FROZEN["headline_out_of_time_auc"],
    "out-of-time projects 2015-2018": FROZEN["counts"][bool(FROZEN["exclude_odd_rows"])]["out_of_time"],
}


def check(name, got):
    want = CHECKPOINTS[name]
    ok = round(got, 3) == want if isinstance(want, float) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got if isinstance(got, int) else round(got, 4)} (expected {want})")
    return ok


def savings(df, top, costs):
    """What the ranking saves, with impact.py's metric (development spend lost on backed
    projects that withdrew, per MW that got built). The model's top `top` by score, in the same
    stable order the page uses, against the expected value of a random pick of the same size.
    "Saved" = what the random rate would have lost to build the MW the model's picks built.
    Cost per withdrawn project is flat, so the reduction doesn't depend on it; dollars scale with it."""
    mw = df["mw_total"].fillna(0).clip(lower=0).to_numpy()
    yv = df["y"].to_numpy()
    pick = np.argsort(-df["p"].to_numpy(), kind="stable")[:top]
    m_withdrawn = int(top - yv[pick].sum())
    m_mw = float(mw[pick][yv[pick] == 1].sum())
    r_withdrawn = top * (1 - yv.mean())
    r_mw = top * float((mw * yv).sum()) / len(df)
    m_rate, r_rate = m_withdrawn / m_mw, r_withdrawn / r_mw     # withdrawn projects per built MW
    return {
        "top": int(top), "built": int(yv[pick].sum()),
        "mw_built": round(m_mw), "random_mw_built": round(r_mw),
        "per_mw": [round(c * m_rate) for c in costs],
        "random_per_mw": [round(c * r_rate) for c in costs],
        "reduction": round(1 - m_rate / r_rate, 3),
        "saved": [round(c * (r_rate - m_rate) * m_mw) for c in costs],
    }


def main():
    train_years = config.SPLITS["within"]["train"]
    test_years = config.SPLITS["within"]["test"]
    assert SCORE_YEARS[1] < test_years[0], "the demo must not reach the test cohort"

    raw = data.load_raw_queue()
    projects = data.built_within(raw, config.HORIZON)
    # Drop the test cohort before anything else touches the table.
    projects = projects[projects["q_year"] <= SCORE_YEARS[1]]
    X, y = features.build_features(raw, projects)

    tr = projects["q_year"].between(*train_years).to_numpy()
    oot = projects["q_year"].between(*SCORE_YEARS).to_numpy()
    est = selected(MODEL, X[tr])
    est.fit(X[tr], y[tr])
    C = est.get_params()["clf__C"]

    cohort = projects[oot].assign(p=est.predict_proba(X[oot])[:, 1], y=y[oot].to_numpy())
    auc = evaluate.roc_auc_score(cohort["y"], cohort["p"])
    lo, hi = evaluate.bootstrap_ci(cohort["y"].to_numpy(), cohort["p"].to_numpy(),
                                   evaluate.roc_auc_score, n=1000, seed=config.SEED)

    print("Checkpoints (docs/model_log.md):")
    ok = [check("LASSO out-of-time AUC 2015-2018", auc),
          check("out-of-time projects 2015-2018", int(len(cohort)))]
    if not all(ok):
        sys.exit("A checkpoint moved: data.js NOT written. Reconcile with docs/model_log.md first.")

    # ---- the N_SAMPLE projects on the page: a random draw, not hand-picked, from projects with
    # a capacity the page can show (MIN_MW)
    pool = cohort[cohort["mw_total"] >= MIN_MW]
    sample = pool.sample(n=N_SAMPLE, random_state=SAMPLE_SEED).sort_values(["q_date", "q_id"])
    rows = [{
        "id": str(r.q_id),
        "name": r.project_name if isinstance(r.project_name, str) and r.project_name.strip() else None,
        "operator": str(r.region),
        "state": r.state if isinstance(r.state, str) else None,
        "tech": str(r.type_clean),
        "mw": round(float(r.mw_total), 1) if np.isfinite(r.mw_total) else None,
        "entered": r.q_date.strftime("%Y-%m-%d"),
        "p": round(float(r.p), 4),
        "built": int(r.y),
    } for r in sample.itertuples()]
    s_built = int(sample["y"].sum())

    # ---- the same comparison over the whole cohort: the model's top TOP/N_SAMPLE share (25 of 100
    # on the page = top 25%) vs. a random share that size (expected = k x base rate) vs. perfect hindsight
    k = round(len(cohort) * TOP / N_SAMPLE)
    c_built = int(cohort["y"].sum())
    top = cohort.sort_values("p", ascending=False, kind="stable").head(k)
    full = {
        "n": int(len(cohort)), "built": c_built, "k": int(k),
        "top_built": int(top["y"].sum()),
        "random_expected": round(k * c_built / len(cohort), 1),
        "perfect": int(min(k, c_built)),
    }

    # ---- how typical is this draw? Redraw N_SAMPLE projects many times from the same pool
    # and compare the model's top TOP with a random TOP each time.
    p_all, y_all = pool["p"].to_numpy(), pool["y"].to_numpy()
    rng = np.random.default_rng(config.SEED)
    sim_top, sim_rand = np.empty(SIM_DRAWS), np.empty(SIM_DRAWS)
    for d in range(SIM_DRAWS):
        i = rng.choice(len(pool), N_SAMPLE, replace=False)
        sim_top[d] = y_all[i[np.argsort(-p_all[i], kind="stable")[:TOP]]].sum()
        sim_rand[d] = TOP * y_all[i].sum() / N_SAMPLE
    s_top = int(sample.sort_values("p", ascending=False, kind="stable").head(TOP)["y"].sum())
    s_excess = s_top - TOP * s_built / N_SAMPLE
    sim = {
        "draws": SIM_DRAWS,
        "mean_top": round(float(sim_top.mean()), 1),
        "mean_random": round(float(sim_rand.mean()), 1),
        # share of draws in which the model beat random by less than it does on this page
        "luckier_than": round(float((sim_top - sim_rand < s_excess - 1e-9).mean()), 2),
    }

    # ---- what the ranking saves, at the sourced cost per withdrawn project (low, mid, high)
    costs = config.DEV_COST_RANGE_USD
    save = {"cost": list(costs), "reference_mw": config.REFERENCE_MW,
            "sample": savings(sample, TOP, costs), "full": savings(cohort, k, costs)}
    # the pop-up must count the same picks as the scoreboard
    assert save["sample"]["built"] == s_top and save["full"]["built"] == full["top_built"]

    payload = {
        "meta": {
            "model": "LASSO", "C": C, "auc": round(auc, 3),
            "auc_lo": round(float(lo), 3), "auc_hi": round(float(hi), 3),
            "score_years": list(SCORE_YEARS), "train_years": list(train_years),
            "base_rate": round(c_built / len(cohort), 4),
            "horizon": config.HORIZON, "regions": config.DATE_REGIONS,
            "seed": SAMPLE_SEED, "generated": date.today().isoformat(),
            "pool_n": int(len(pool)), "pool_excluded": int(len(cohort) - len(pool)),
        },
        "sample": {
            "n": N_SAMPLE, "top": TOP, "built": s_built,
            "random_expected": round(TOP * s_built / N_SAMPLE, 1),
            "perfect": min(TOP, s_built),
        },
        "full": full,
        "sim": sim,
        "savings": save,
        "projects": rows,
    }
    OUT.write_text("// Generated by demo/export_data.py. Do not edit by hand.\n"
                   f"window.DEMO = {json.dumps(payload, indent=1)};\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}: {N_SAMPLE} projects ({s_built} built within "
          f"{config.HORIZON}y), AUC {auc:.3f} [{lo:.3f}, {hi:.3f}], full cohort top {k}: "
          f"{full['top_built']} built vs {full['random_expected']} random vs {full['perfect']} perfect")
    for name, s in (("page top", save["sample"]), ("full cohort top", save["full"])):
        print(f"Savings, {name} {s['top']}: {100 * s['reduction']:.0f}% less lost per built MW; saved "
              + " / ".join(f"${v / 1e6:,.0f}M" for v in s["saved"])
              + f" at ${costs[0] / 1e6}M / ${costs[1] / 1e6}M / ${costs[2] / 1e6}M per withdrawn project")


if __name__ == "__main__":
    main()
