"""Back early, or wait for the interconnection agreement? An expected-cost comparison.

    python -m queue_risk.strategy      # measured inputs + the assumption sweep

Today's rule of thumb: don't take a queue project seriously until it signs an
interconnection agreement (IA). Waiting is safer but the price of the rights is higher.
Backing at entry is cheaper but many more projects fail. The model's job is to make
the early bet less of a lottery.

The common yardstick is **cost per MW that actually gets built**:

    wait:   price_late                                    / s_wait
    early:  (price_early + spend_to_IA / MW) x (1+r)^t    / s_early

where s is the MW-weighted share of backed capacity that reaches operation within
HORIZON years of the investment (entry for "early", IA signing for "wait").
Backing early wins when s_early clears the break-even hit rate

    s* = s_wait x (price_early + spend_to_IA / MW) x (1+r)^t / price_late

WHAT THIS IS NOT. Prices are published industry ranges, not observed trades, and the
post-IA build rate comes from the minority of projects whose IA date is recorded
(mostly CAISO and ERCOT) and swings from 17% to 82% by signing year. So every output
is a band across the assumption grid, never a single number. The early label (built
within 5 years of ENTRY) is stricter than the wait label (within 5 years of IA, i.e.
up to ~7.6 years after entry), which biases the comparison against backing early.
"""
import itertools

import numpy as np
import pandas as pd

from . import config, data


def post_ia_build_rates(raw: pd.DataFrame, horizon=config.HORIZON, cutoff=config.DATA_CUTOFF):
    """MW-weighted share of projects built within `horizon` years of IA signing, by
    signing year, for clean-energy projects in the date-reporting operators."""
    end = pd.Timestamp(cutoff)
    c = raw[data.in_scope(raw) & raw["region"].isin(config.DATE_REGIONS)
            & raw["ia_date"].notna() & (raw["ia_date"] + pd.DateOffset(years=horizon) <= end)].copy()
    done = data.completion_date(c)
    c["built"] = (done.notna() & (done <= c["ia_date"] + pd.DateOffset(years=horizon))).astype(int)
    c["mw_built"] = c["mw_total"] * c["built"]
    g = c.groupby(c["ia_date"].dt.year).agg(n=("built", "size"), share=("built", "mean"),
                                           mw=("mw_total", "sum"), mw_built=("mw_built", "sum"))
    g["mw_share"] = g["mw_built"] / g["mw"]
    return g


def years_entry_to_ia(raw: pd.DataFrame) -> float:
    c = raw[raw["type_1"].isin(config.CLEAN_TYPES) & raw["region"].isin(config.DATE_REGIONS)
            & raw["ia_date"].notna()]
    return float(((c["ia_date"] - c["q_date"]).dt.days / 365.25).median())


def breakeven_hit_rate(s_wait, price_early, price_late, spend_to_ia=0.0, avg_mw=100.0,
                       r=0.0, years_to_ia=0.0):
    """MW-weighted hit rate an early backer needs to match waiting for the IA."""
    early_per_w = price_early + spend_to_ia / (avg_mw * 1e6)
    return s_wait * early_per_w * (1 + r) ** years_to_ia / price_late


def cost_per_built_mw(price_per_w, hit_rate, spend_per_project=0.0, avg_mw=100.0, carry=1.0):
    """USD paid per MW that reaches operation."""
    per_mw = price_per_w * 1e6 + spend_per_project / avg_mw
    return per_mw * carry / hit_rate if hit_rate > 0 else np.inf


def sweep_breakeven(s_wait_range, avg_mw, years_to_ia) -> pd.DataFrame:
    """Break-even hit rate over the whole assumption grid (one row per combination)."""
    rows = []
    for s, pe, pl, sp, r in itertools.product(s_wait_range, config.PRICE_EARLY_USD_PER_W,
                                              config.PRICE_LATE_USD_PER_W, config.SPEND_TO_IA_USD,
                                              config.COST_OF_CAPITAL):
        rows.append({"s_wait": s, "price_early": pe, "price_late": pl, "spend_to_ia": sp, "r": r,
                     "breakeven": breakeven_hit_rate(s, pe, pl, sp, avg_mw, r, years_to_ia)})
    return pd.DataFrame(rows)


def compare(s_early, s_wait_range, avg_mw, years_to_ia) -> pd.DataFrame:
    """Early vs. wait cost per built MW, for a measured early hit rate (e.g. the model's
    top-K on the test cohort), across the assumption grid. Returns the grid with the
    saving per built MW (positive = backing early is cheaper)."""
    g = sweep_breakeven(s_wait_range, avg_mw, years_to_ia)
    carry = (1 + g["r"]) ** years_to_ia
    g["early_cost_per_built_mw"] = [cost_per_built_mw(pe, s_early, sp, avg_mw, c)
                                    for pe, sp, c in zip(g["price_early"], g["spend_to_ia"], carry)]
    g["wait_cost_per_built_mw"] = g["price_late"] * 1e6 / g["s_wait"]
    g["saving_per_built_mw"] = g["wait_cost_per_built_mw"] - g["early_cost_per_built_mw"]
    return g


def main():
    raw = data.load_raw_queue()
    rates = post_ia_build_rates(raw)
    recent = rates.loc[2015:2020]
    s_wait_range = (recent["mw_share"].min(), float(np.average(recent["mw_share"], weights=recent["mw"])),
                    recent["mw_share"].max())
    t = years_entry_to_ia(raw)
    labeled = data.built_within(raw, config.HORIZON)
    avg_mw = float(labeled["mw_total"].median())
    base = labeled[labeled["q_year"].between(2015, 2020)]
    base_mw = float((base["mw_total"] * base["built"]).sum() / base["mw_total"].sum())

    print(f"Post-IA {config.HORIZON}-year build rate (MW-weighted), IA signed 2015-2020: "
          f"low {s_wait_range[0]:.0%}, pooled {s_wait_range[1]:.0%}, high {s_wait_range[2]:.0%} "
          f"(n={int(recent['n'].sum()):,} projects with a recorded IA date)")
    print(f"Median years from entry to IA: {t:.1f} | median project size: {avg_mw:.0f} MW")
    print(f"Backing every entry blind, 2015-2020 cohorts: {base_mw:.1%} of MW built within "
          f"{config.HORIZON} years of entry")
    g = sweep_breakeven(s_wait_range, avg_mw, t)
    q = g["breakeven"].quantile([0, 0.25, 0.5, 0.75, 1])
    print(f"\nBreak-even early hit rate across {len(g)} assumption combinations:")
    print(f"  min {q[0]:.0%} | 25th pct {q[0.25]:.0%} | median {q[0.5]:.0%} | 75th pct {q[0.75]:.0%} | max {q[1]:.0%}")
    # Computed directly, not filtered from the grid: the sourced ranges no longer include
    # zero spend or a zero cost of capital, so a "price only" row isn't in the grid.
    pe, pl = config.PRICE_EARLY_USD_PER_W[1], config.PRICE_LATE_USD_PER_W[1]
    sp, r = config.SPEND_TO_IA_USD[1], config.COST_OF_CAPITAL[1]
    print("  price effect alone (mid prices, no spend, no carry):",
          ", ".join(f"s_wait {s:.0%} -> {breakeven_hit_rate(s, pe, pl):.0%}" for s in s_wait_range))
    print("  all mid assumptions (mid prices, mid spend, mid cost of capital):",
          ", ".join(f"s_wait {s:.0%} -> {breakeven_hit_rate(s, pe, pl, sp, avg_mw, r, t):.0%}"
                    for s in s_wait_range))


if __name__ == "__main__":
    main()
