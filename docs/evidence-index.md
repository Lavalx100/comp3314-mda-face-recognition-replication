# Evidence index

| Evidence | Location | What it supports |
|---|---|---|
| Initial five-split result chart | [`docs/assets/accuracy_vs_training_size.svg`](assets/accuracy_vs_training_size.svg) | Historical exploratory accuracy by gallery size and method |
| Initial unit-L2 raw rows and summary | [`unit_l2/`](../results/2026-10-24-partial-replication/unit_l2/) | Five-split historical scores and settings |
| Initial alternate-scale output | [`within_scatter/`](../results/2026-10-24-partial-replication/within_scatter/) | Five-split matched-seed normalization sensitivity |
| Person 3 handoff | [`dated handoff`](experiments/2026-10-24-person3-handoff.md) | Historical report-ready interpretation and caveat |
| 30-split primary raw rows and summaries | [`unit_l2/`](../results/2026-10-28-30-repeat-followup/unit_l2/) | 480 method-by-split rows, summary, chart, and MDA objective trace |
| 30-split normalization sensitivity | [`within_scatter/`](../results/2026-10-28-30-repeat-followup/within_scatter/) | Same-seed alternative setting and its artifacts |
| Paired comparison tables | [`primary paired CSV`](../results/2026-10-28-30-repeat-followup/unit_l2/paired_differences.csv), [`alternate paired CSV`](../results/2026-10-28-30-repeat-followup/within_scatter/paired_differences.csv) | Per-split MDA/2-2 differences and conditional bootstrap ranges |
| Experiment record | [`30-split follow-up`](experiments/2026-10-28-30-repeat-followup.md) | Configuration, commands, findings, and limitations |
| Current report and PDF | [`report source`](../report/main.tex), [`rendered report`](../output/pdf/COMP3314_Group27_MDA_Reproduction.pdf) | Original-resolution protocol, fixed-setting and full-grid results, leakage caveat, optimization validation, and citations |
| Canonical full dimension grids | [`within-scatter`](../results/2026-11-09-paper-proximate-full-grid/), [`unit-L2`](../results/2026-11-09-paper-proximate-full-grid-unit-l2/) | Complete five-repeat 10,130-pair MDA/2-2 grids used by current tables and figures |
| Optimization and targeted equivalence | [`validation record`](experiments/2026-11-10-optimized-default-and-equivalence.md), [`validation outputs`](../results/2026-11-10-optimization-boundary-validation/) | Interrupted-run discrepancy audit and corrected direct/cache checks under both normalizations |
| Research overview site | [`site source`](../site/index.html) | Browser-readable project plan, method visuals, canonical results, current performance, and task ownership |

Split-level outputs document the group's computational results; independent implementation review is still pending. Tests, reruns, source citations, and a teammate's audit provide stronger validation than activity screenshots.
