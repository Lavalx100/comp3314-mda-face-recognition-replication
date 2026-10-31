# Change log

## 2026-10-31 — Visualization layout fix

- Reserved separate vertical space for the x-axis title and two-row legend in every saved accuracy chart; regenerated all 13 result charts and the historical chart asset from their existing CSV summaries.
- Added space below the dimension heatmap's rotated axis labels and regenerated it so the explanatory note no longer crowds the labels.
- Chart values are unchanged; no experiments were rerun.

## 2026-10-31 — Repository cleanup

- Removed a historical editor screenshot that exposed a local filesystem path and unrelated workspace tabs, then removed it from all Git history.
- Removed the compiled PDF for the superseded 64×64 preliminary report; retained and labeled its LaTeX source as an old draft.
- Added Git ignore rules for local dataset folders and PGM face images, and updated the evidence index, progress record, and team handoff.
- Kept the GitHub repository private pending the group's release decision.

## 2026-10-28 — Original-resolution ORL reproduction follow-up

- Added direct loading of the official 112×92 Cambridge ORL PGM archive and recorded its checksum; no face images are committed.
- Added the Fig. 3 MDA stopping rule and expanded baseline dimensions for the original-resolution protocol.
- Completed matched 30-split fixed-setting runs under unit-L2 and within-scatter scaling, with paired comparisons and objective traces.
- Completed a five-repeat, 156-setting MDA/2-2 dimension sweep and full PCA/Fisherface/MDA/2-1 feature curves; retained all 8,920 split/configuration rows and generated the row×column heatmap.
- Replaced quadratic high-dimensional objective-trace construction with an equivalent scatter-trace calculation; vectorized mode-wise scatter accumulation and made the sweep checkpoint after every split with compatible resume support.
- Documented the remaining gap to Table I, warned that grid maxima are test-selected, and updated the research, progress, limitations, README, and Person 3 handoff materials.
- Added a reusable SVG heatmap renderer for the MDA/2-2 dimension surface.

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
