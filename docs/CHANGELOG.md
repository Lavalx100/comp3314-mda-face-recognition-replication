# Change log

## 2026-10-28 — Thirty-split paired follow-up

- Ran 30 matched splits per gallery size for both unit-L2 and within-scatter projection normalization.
- Added raw results, summaries, plots, objective traces, and paired split-difference summaries under a new dated results folder; the five-split study remains unchanged.
- Added a reusable paired-analysis script and tests; the latest recorded local run passed all nine unit tests.
- Updated research plan, protocol, limitations, report, presentation outline, and progress record to use the follow-up and qualify the conditional bootstrap ranges.
- Expanded the three-page LaTeX report with the DTC intuition, alternating-mode diagram, and paper-versus-group protocol comparison; compiled and visually reviewed its PDF.
- Added a responsive local project overview with the research question, method flow, 30-split chart/table, Person 3 checklist, team roles, and links to the evidence.

## 2026-10-28 — Research repository scaffold (0.1.0)

- Created a standalone COMP3314 MDA research package, separate from the previous static-site repository.
- Organized prior baseline and scaling-sensitivity outputs as dated, frozen artifacts.
- Added protocol, paper notes, data provenance, limitations, project plan, team plan, report/presentation scaffolds, and progress tracking.
- Added unit tests and a CI workflow; verify both before public release.
- Excluded downloaded face data and local environments from version control.
- Verified 7 unit tests and a two-gallery-size smoke run using the cached dataset at that stage.

## 2026-10-24 — Initial partial-reproduction work (recorded retrospectively)

- Compared Eigenfaces PCA, Fisherfaces LDA, MDA/2-1, and MDA/2-2 on scikit-learn's 64×64 Olivetti copy.
- Evaluated 2/3/4/5 training images per identity using five matched seeded splits and Euclidean 1-NN.
- Ran same-seed sensitivity analysis for projection-axis normalization.
- Produced split-level results, summary tables, chart, MDA objective trace, and Person 3 handoff.

The October 24 entry is a retrospective summary of earlier artifacts and work notes; the original individual Git commits were not available in this new repository.
