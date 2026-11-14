# COMP3314 Group 27 — final presentation run of show

The deck has 12 slides: 10 content slides, a cover, and a references/Q&A slide. The planned talk is 8:30, leaving about 30 seconds for speaker handoffs and pacing within the 9-minute limit. Allow 3 minutes for Q&A.

- [Editable PowerPoint](Group27_MDA_Final_Presentation.pptx)
- [PDF version](../output/pdf/Group27_MDA_Final_Presentation.pdf)

The submission copy contains no embedded slide comments or speaker notes; this run of show remains available separately for the presenters.

| Time | Slide | Speaker | Purpose |
|---|---:|---|---|
| 0:00–0:15 | 1 | Person 1 | State the question and introduce the paper |
| 0:15–0:55 | 2 | Person 1 | Explain vectorization and connect to PCA/LDA |
| 0:55–1:45 | 3 | Person 2 | Show the two-mode tensor projection |
| 1:45–2:30 | 4 | Person 2 | Explain the discriminant tensor criterion |
| 2:30–3:15 | 5 | Person 2 | Explain the alternating generalized eigenproblems |
| 3:15–4:00 | 6 | Person 3 | Describe original ORL data and matched protocol |
| 4:00–5:05 | 7 | Person 3 | Present fixed-setting results and scaling sensitivity |
| 5:05–6:00 | 8 | Person 4 | Compare the full-grid peaks with the paper, with the selection caveat |
| 6:00–6:45 | 9 | Person 4 | Resolve the gallery-size-three dip using nested splits |
| 6:45–7:35 | 10 | Person 4 | Discuss assumptions, limits, and a better next evaluation |
| 7:35–8:20 | 11 | Person 3 | Report the optimization and targeted equivalence checks |
| 8:20–8:30 | 12 | All | Leave references visible before Q&A |

## Handoffs

- Person 1 to Person 2: “MDA keeps those axes. Here is what its feature looks like.”
- Person 2 to Person 3: “The method is clear; now we’ll show the exact data split we tested.”
- Person 3 to Person 4: “Those are fixed-dimension results. The full search raises a selection question.”
- Person 4 to Person 3: “With that limitation clear, we’ll close with how we validated the faster implementation.”

## Sources checked

- Original paper: Yan et al. (2007), *IEEE Transactions on Image Processing*, 16(1), 212–220.
- Final group report: report/main.tex
- Paper-proximate protocol and leakage audit: docs/experiments/2026-11-09-paper-proximate-reproduction.md
- Optimized default and targeted equivalence: docs/experiments/2026-11-10-optimized-default-and-equivalence.md
- Fixed-setting data: results/2026-11-09-paper-proximate-fixed/{unit_l2,within_scatter}/summary.csv
- Full-grid values: results/2026-11-09-paper-proximate-full-grid*/best_by_gallery.csv
- Nested gallery diagnostic: results/2026-11-09-nested-gallery-diagnostic/summary.csv
- Performance benchmark: results/2026-11-10-optimized-performance/
- Official ORL database details: AT&T Laboratories Cambridge.

The paper’s Table I values are identified as the authors’ reported best scores. All other results are identified as Group 27 results. The full-grid maxima use probe scores for dimension selection, so they are optimistic and are not unbiased test estimates.

## Q&A rehearsal

- **Did probe images train the projections?** No. Projection fitting uses gallery images. Probe accuracy was reused to select the full-grid dimensions, so those selected maxima are optimistic.
- **Why does the within-scatter grid dip at gallery size three?** That dip appears in the selected full-grid values. With dimensions fixed at 6×6 on 30 nested splits, adding the third image improves accuracy under both scalings.
- **Is this an exact reproduction?** No. The article does not specify split seeds, all preprocessing defaults, eigenvector scaling, singular-scatter handling, or the complete dimension-selection protocol.
- **Why do the paper and group scores differ?** The published settings are incomplete, and the group’s fixed and grid-selected results use different selection protocols. The displayed values should not be treated as a matched comparison.
- **What did the optimized implementation change?** It reuses class grouping and eigensystem factorizations while forming each requested-size projection with the direct operation shape. The corrected cache matched the direct fits on the 17 flagged settings under both normalizations.
- **Did you rerun the complete grid with the optimized code?** No. The complete direct-run grids remain canonical; the optimized path is the default for future sweeps.
- **What would improve the evaluation?** Select dimensions using training-only validation, then evaluate once on an untouched outer test split and a more varied dataset.
