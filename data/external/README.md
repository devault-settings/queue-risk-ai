# External data

- `sabin/`: the Sabin Center for Climate Change Law's *Opposition to Renewable Energy Facilities in the United States* reports (2022–2026 editions; public, open access), with their text extracted by `python -m queue_risk.sabin`. Copies are kept for reproducibility; cite the Sabin Center.
- `queue_with_costs.csv`: Paul's join of Berkeley Lab's *Queued Up* project list with interconnection-cost study records (the `c_*` columns: study costs, type, date, URL, coordinates; about 4,900 of 38,201 projects matched). Kept for reference. **No model or result in this repo uses it**: interconnection costs are known only after the studies are done, so they are not entry-time features.
- `census/`, `eia/`: optional downloads, not committed (rebuild with the code in `src/`).
