# Assumptions log

Every number the impact analysis uses that does *not* come from the Queued Up file. Each one lives in [config.py](../src/queue_risk/config.py) as a (low, mid, high) range, is swept, and is never quoted as a single figure.

**Rules for adding a number:**
- Record the URL, the date it was retrieved, the exact table or page, the units, and the dollar-year.
- Convert to USD per watt (or USD per project) in 2025 dollars.
- Find a second source to cross-check before marking it **verified**.
- Mark single-source industry figures **indicative**, and say so on the slide.

## 1. Price of development rights, early vs. late stage

| | Value | Source | Status |
|---|---|---|---|
| Early stage (pre-IA) | $0.02–0.05/W | [Ascend Analytics](https://www.ascendanalytics.com/blog/increasing-early-stage-project-success-in-a-highly-competitive-renewable-energy-m-a-landscape): solar+storage, US ISOs | Indicative (industry blog) |
| Late stage (IA / ready-to-build) | $0.07–0.20/W | same | Indicative |
| Cross-check | <$50k/MW early; $80–150k/MW in interconnection; $150–350k/MW with IA + PPA | [IB guide](https://ibinterviewquestions.com/guides/energy-investment-banking/renewable-energy-valuation-methods) (unsourced) | Consistent with the row above, weak source |
| Regional spread | ERCOT solar $30–75k/MW; NYISO ready-to-build solar ~$200k/MW | [Enerdatics H1 2026](https://www.enerdatics.com/insights/enerdatics-releases-h1-2026-renewables-m-a-outlook-tracking-64b-of-global-deal-value) | Indicative. Could support region-specific prices |
| Early floor (observed trade) | $5,000/MW + $3,000/MW in milestone payments for 2,425 MW of greenfield solar = $0.008/W | Bimergen Energy Corp, Form S-1/A, SEC, Aug 7, 2025 | Observed. Greenfield is earlier than queue entry, so this is a **floor**, not an estimate of the early price |
| Late support (observed trades) | 72.9 MW at $0.08/W (2015 purchase agreement filed with the SEC); $32M for ~221 MW, ~$0.145/W (Dominion Energy 2016 10-K) | SEC filings | Supports the late range. **To confirm** the queue stage of both projects at sale |

## 2. Development spend from queue entry to the IA, per project (`SPEND_TO_IA_USD`)

What an early backer actually pays to carry a project to its agreement, built from parts for a 100 MW reference project. Range used: **$400k / $1.45M / $3.65M** (low / mid / high).

| Part | Value (low / mid / high) | Source | Status |
|---|---|---|---|
| Interconnection studies | ~$150k | FERC Order 2023 study deposit tier for 80–200 MW (LGIP section 3.1.1.1); see deposit rules below | Verified against the order text |
| Permitting | $250k / $300k / $500k | NREL PV cost benchmarks, 2016 and 2018 editions; mid is NREL 2018: $295k, rounded. CA is 9.5% of 2015–2020 projects, so the high-cost CA case is a minority | Indicative. **To update** from the 2025 Q1 benchmark ZIP once the NLR data site is back online |
| Land | ~$0 / ~$1M / $3M | Low and mid are **assumptions** (mid = cost of a land option); high is NREL ($0.03/W) | Low/mid judgment; high **to update** with the NREL 2025 Q1 benchmark |
| Deposits (commercial readiness) | Excluded | — | Tied-up money, not spend: partially refundable (see below) |

**FERC Order No. 2023** (Docket RM22-14-000, eLibrary Accession 20230728-3060). Verified against the order text; page references are to the order PDF:
- Study deposit tiers (LGIP section 3.1.1.1): $35k + $1k/MW (20–80 MW); $150k (80–200 MW); $250k (200 MW+). The top tier starts at **200 MW** (one law-firm summary said 250 MW; that is wrong).
- Application fee: $5,000. The pre-2023 study deposit was $10,000.
- Commercial readiness deposits step up: 2× the study deposit at entry, then 5%, then 10% of estimated network upgrade costs (order text, PDF p. 458, printed p. 456), then 20% at the LGIA (PDF p. 470, printed p. 468). (The earlier version of this file was missing the 10% step.)
- Withdrawal penalties (LGIP section 3.7.1.1, PDF p. 1192) follow the same steps: 2× study costs / 5% / 10% / 20%.
- Deposits are partially refundable on withdrawal (LGIP section 3.7, PDF p. 1191): study costs and any withdrawal penalty come out first, and the remainder is returned.
- **Scope:** these rules apply to post-2023 projects only. Pre-2023 cohorts (most of our training data) paid each operator's older tariff rates.

Further reading:
- FERC Order 2023, secondary summaries, further reading: [Troutman Pepper](https://www.troutman.com/insights/troutman-pepper-summary-of-ferc-order-no-2023-on-generator-interconnection-reform.html), [Orrick](https://www.orrick.com/en/Insights/2023/08/FERC-Finalizes-Rule-Adopting-Interconnection-Reforms), [Orrick on 2023-A](https://www.orrick.com/en/Insights/2024/04/FERC-Order-No-2023-A-Provides-Further-Guidance-on-Interconnection-Queue-Reforms).
- NREL (now NLR) Q1 2025 PV system cost benchmarks, for the to-update rows above: [data.gov](https://catalog-beta.data.gov/dataset/q1-2025-solar-photovoltaic-system-cost-benchmarks), [DOE](https://www.energy.gov/node/4846387).

## 3. Development spend lost per withdrawn project (`DEV_COST_RANGE_USD`)

For the investor, this is the price paid for the rights (section 1) plus the spend carried to the point of withdrawal (section 2), less any refunded deposits. **Built Nov 24** from those sourced components, for a 100 MW reference project (the median size), replacing the unsourced $1M / $3M / $10M placeholder:

| | Price of rights at entry | + spend toward the IA | = lost per withdrawn project |
|---|---|---|---|
| Low | $0.02/W × 100 MW = $2.0M | $0.40M | **$2.40M** |
| Mid | $0.035/W × 100 MW = $3.5M | $1.45M | **$4.95M** |
| High | $0.05/W × 100 MW = $5.0M | $3.65M | **$8.65M** |

This is an **upper bound**: many projects withdraw before spending everything up to the IA, and part of the price can be recovered by reselling the rights. It inherits the confidence of its parts: medium for the spend, low to medium for the price.

## 4. Cost of capital (`COST_OF_CAPITAL`)

Range used: **6% / 10% / 15%** (low / mid / high).

| Value | Source | Status |
|---|---|---|
| Low 6% | NREL ATB 2025, Financial Cases & Methods: nominal after-tax WACC 5.6–6.5% for renewables | Sourced |
| Mid 10% | NREL ATB 2025: construction equity return 10.0% | Sourced |
| High 15% | **Assumption**: day-one queue risk | Judgment |
| Cross-check | Lazard LCOE+ 2025, financing assumptions (debt/equity split, cost of each) | To pull |

Development-stage equity is riskier than the operating-project WACC and construction equity these sources describe, so the high end sits above both.

## Numbers that look relevant but are NOT spend

Berkeley Lab's interconnection cost studies report the estimated network upgrade costs of **withdrawn** projects:
- MISO: $452/kW ([APPA](https://www.publicpower.org/periodical/article/interconnection-costs-have-risen-steeply-miso-berkeley-lab-report))
- SPP: $304/kW, early 2020s ([T&D World](https://www.tdworld.com/utility-business/article/21264862/spp-data-show-rising-network-upgrade-costs-especially-for-withdrawn-projects))
- ISO-NE: $613/kW ([Renewable Energy World](https://www.renewableenergyworld.com/solar/utility-integration/one-in-five-projects-survives-the-ne-iso-interconnection-process-report-says/))

These are bills the projects walked away from, not money spent, so they don't go in the wasted-spend figure. They do belong in the story about *why* projects withdraw. They can't be model features either: study results arrive after entry, which would be leakage.

Data: https://emp.lbl.gov/interconnection_costs
