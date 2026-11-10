# 9-minute presentation outline + 3-minute Q&A

Use 8–9 concise slides. The story is the paper's idea, what the group actually tested, and what the evidence can support.

| Time | Slide | Main point and visual |
|---:|---|---|
| 0:00–0:45 | Question | Can keeping an image as a matrix help supervised face recognition? Show a face matrix beside a flattened vector. |
| 0:45–1:35 | Paper idea | MDA learns row and column projections; cite Yan et al. and state the small-sample motivation. |
| 1:35–2:35 | How MDA works | DTC between/within-class intuition, generalized eigenproblem, alternating row/column updates. Use the method-flow diagram in the report. |
| 2:35–3:20 | Course connections | PCA, LDA, generalized eigenvectors, and Euclidean nearest-neighbour classification. |
| 3:20–4:15 | Our protocol | Original 112×92 ORL images, 40 people, gallery sizes 2–5, gallery-only fitting, 30 matched fixed-setting splits. State the 64-level equalization assumption. |
| 4:15–5:20 | Fixed-setting results | Use the 30-split accuracy chart and table. MDA/2-1 leads in this implementation; MDA/2-2 depends strongly on axis scaling. |
| 5:20–6:20 | Full dimension search | Show the unit-L2 heatmap and the within-scatter comparison. Explain that the best dimensions were selected on the same probe scores, so maxima are optimistic. |
| 6:20–7:25 | Critical analysis | Discuss the gap to the paper, undocumented settings, and the gallery-size-3 dip. The dip occurs only in the selected within-scatter grid; a fixed nested 2-vs-3 diagnostic improves under both scalings. |
| 7:25–8:30 | What we learned | Explain that implementation speedups must preserve numerics: cache eigensystem factorizations, keep dimension-shaped projections, and validate the sensitive cases. Mention full grids remain canonical. |
| 8:30–9:00 | Takeaway | MDA's tensor structure is useful to study, but this implementation does not reproduce the paper's best scores; make the assumptions and selection bias explicit. |

## Visual and result files

- Fixed-setting chart: results/2026-11-09-paper-proximate-fixed/unit_l2/accuracy_vs_training_size.svg
- Unit-L2 full-grid heatmap: results/2026-11-09-paper-proximate-full-grid-unit-l2/mda22_dimension_heatmap.svg
- Within-scatter full-grid heatmap: results/2026-11-09-paper-proximate-full-grid/mda22_dimension_heatmap.svg
- Within-scatter grid summary: results/2026-11-09-paper-proximate-full-grid/best_by_gallery.csv
- Current report with the fixed-setting and full-grid tables: output/pdf/COMP3314_Group27_MDA_Reproduction.pdf
- Correctness, leakage, and implementation audit: docs/experiments/2026-11-09-paper-proximate-reproduction.md
- Corrected cache validation under both axis normalizations: results/2026-11-10-optimization-boundary-validation/summary.json and within_scatter_summary.json

Label every result as either reported by Yan et al. or produced by Group 27. Cite paper figures and reproduce no paper visual without attribution.

## Q&A rehearsal

- **Did test probes affect the learned projections?** No. Projections are fit on gallery images. However, the full-grid dimension maxima are selected using probe accuracy, so those selected values are optimistic.
- **Why do the results differ from the paper?** The article does not fully specify preprocessing defaults, scaling, singular-scatter handling, random splits, or dimension selection. Our choices are documented, but the numerical gap remains.
- **Why does gallery size 3 look worse than 2 in one plot?** Only the probe-selected within-scatter grid shows that dip. Unit-L2 and the fixed-dimension nested comparison improve from 2 to 3.
- **Is this an exact reproduction?** No. It is a paper-proximate study using original-resolution ORL and the stated MDA update structure; the published run details are insufficient to match Table I exactly.
- **Why keep the full tables from the direct implementation?** They are complete. The corrected optimized path matched direct outputs on all 17 configurations flagged by the partial comparison under both axis normalizations, so recomputing every fit would not improve the evidence.
- **What does the optimization change?** It reuses repeated eigensystem factorizations and was faster on the M3 Max benchmark. It preserves the requested projection shape and was checked against direct fits.

## Speaker handoff

Assign slide owners during rehearsal, but have everyone practice the Q&A. Keep slide text short, use the linked charts, and make the probe-selection limitation visible near the full-grid results.
