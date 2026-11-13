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

## 2. Development spend from queue entry to the IA, per project (`SPEND_TO_IA_USD`)

What an early backer actually pays to carry a project to its agreement. Build it from parts rather than as one guess:

| Part | Where to get it | Status |
|---|---|---|
| Interconnection study deposit | FERC Order 2023 tiers, as summarized by law firms: $35k + $1k/MW (20–80 MW), $150k (80–200 MW), $250k (200 MW+) ([Troutman Pepper](https://www.troutman.com/insights/troutman-pepper-summary-of-ferc-order-no-2023-on-generator-interconnection-reform.html), [Orrick](https://www.orrick.com/en/Insights/2023/08/FERC-Finalizes-Rule-Adopting-Interconnection-Reforms)). One summary puts the top tier at 250 MW | **To verify** against the order text. Note that pre-2023 cohorts paid each operator's older tariff rates |
| Commercial readiness deposit | Initially 2× the study deposit, rising to 5% then 20% of estimated network upgrade costs by the IA; withdrawal penalties scale with it ([Orrick on 2023-A](https://www.orrick.com/en/Insights/2024/04/FERC-Order-No-2023-A-Provides-Further-Guidance-on-Interconnection-Queue-Reforms)) | To verify. Partly refundable, so it isn't all "spend" |
| Permitting, land, environmental, developer overhead | NREL (now NLR) Q1-2025 PV system cost benchmarks, soft-cost categories, 100 MWdc reference system ([data.gov](https://catalog-beta.data.gov/dataset/q1-2025-solar-photovoltaic-system-cost-benchmarks), [DOE](https://www.energy.gov/node/4846387)) | **To pull** from the Excel model. Only the pre-IA share counts |

## 3. Development spend lost per withdrawn project (`DEV_COST_RANGE_USD`)

For the investor, this is the price paid for the rights (section 1) plus the spend carried to the point of withdrawal (section 2), less any refunded deposits. **To build** from sections 1 and 2 once they're verified, replacing the current placeholder of $1M / $3M / $10M.

## 4. Cost of capital (`COST_OF_CAPITAL`)

| Source | Status |
|---|---|
| NREL ATB 2025, Financial Cases and Methods (WACC by technology) ([2024 edition](https://atb.nrel.gov/electricity/2024/financial_cases_&_methods)) | **To pull** (check the 2025 edition) |
| Lazard LCOE+ 2025, financing assumptions (debt/equity split, cost of each) | To pull, as a cross-check |

Development-stage equity is riskier than the operating-project WACC these sources describe, so the high end of the range should sit above them.

## Numbers that look relevant but are NOT spend

Berkeley Lab's interconnection cost studies report the estimated network upgrade costs of **withdrawn** projects:
- MISO: $452/kW ([APPA](https://www.publicpower.org/periodical/article/interconnection-costs-have-risen-steeply-miso-berkeley-lab-report))
- SPP: $304/kW, early 2020s ([T&D World](https://www.tdworld.com/utility-business/article/21264862/spp-data-show-rising-network-upgrade-costs-especially-for-withdrawn-projects))
- ISO-NE: $613/kW ([Renewable Energy World](https://www.renewableenergyworld.com/solar/utility-integration/one-in-five-projects-survives-the-ne-iso-interconnection-process-report-says/))

These are bills the projects walked away from, not money spent, so they don't go in the wasted-spend figure. They do belong in the story about *why* projects withdraw. They can't be model features either: study results arrive after entry, which would be leakage.

Data: https://emp.lbl.gov/interconnection_costs
