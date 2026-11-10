# Project progress

**Last updated:** 2026-11-10 (Hong Kong time)

**Project:** COMP3314 Group 27, MDA face-recognition reproduction study

**Current phase:** source-fidelity, leakage, and performance audits are complete. Both full original-resolution grids remain canonical. The dimension-sweep default now reuses generalized eigensystem factorizations with shape-preserving projections. A redundant exhaustive rerun was stopped after three completed diagnostic splits; all checkpoints and logs remain preserved. The initial cache differences were traced to projection-shape rounding in near-singular fits, corrected, and rechecked only for affected configurations.

## At a glance

| Workstream | Status | Evidence |
|---|---|---|
| Paper and ORL claim notes | Done for current study | [paper notes](paper-notes.md) |
| Experiment implementation | Runnable 64×64 and original-resolution ORL implementations | [`src/run_experiment.py`](../src/run_experiment.py), [`src/orl_data.py`](../src/orl_data.py) |
| Initial five-split baseline | Complete; exploratory and retained as historical | [`2026-10-24 results`](../results/2026-10-24-partial-replication/) |
| Follow-up, 30 matched splits | Complete for current fixed settings | [dated record](experiments/2026-10-28-30-repeat-followup.md) |
| Normalization sensitivity | Complete for both run sizes; caveat required | Follow-up record and paired CSV outputs |
| Original-resolution ORL data | Complete; official 112×92 PGM files read from local archive | [data provenance](data-and-ethics.md), archive hash in run metadata |
| Original-resolution fixed study | Corrected: 30 matched splits per gallery size, 64-level equalization approximation, Fig. 3 threshold with explicit ε, both projection scalings | [`corrected fixed runs`](../results/2026-11-09-paper-proximate-fixed/), [record](experiments/2026-11-09-paper-proximate-reproduction.md) |
| Dimension sweep | Complete and canonical for within-scatter and unit-L2: five repeats over all 10,130 integer pairs; tables/figures continue to use these complete direct-run datasets | [`within-scatter grid`](../results/2026-11-09-paper-proximate-full-grid/), [`unit-L2 grid`](../results/2026-11-09-paper-proximate-full-grid-unit-l2/) |
| Paired split analysis | Complete; conditional descriptive bootstrap ranges | `paired_differences.csv` in each follow-up configuration |
| Optimized sweep | Default path reuses split grouping and cached eigensystem factorizations, then forms dimension-shaped axes; all 17 flagged configurations matched direct fits under both normalizations | [validation record](experiments/2026-11-10-optimized-default-and-equivalence.md), [`unit-L2 validation`](../results/2026-11-10-optimization-boundary-validation/targeted_equivalence.csv), [`within-scatter validation`](../results/2026-11-10-optimization-boundary-validation/within_scatter_targeted_equivalence.csv) |
| Interrupted redundant run | Stopped after three complete gallery-size-2 splits; checkpoint/log retained as a diagnostic only, not used in canonical tables | [`partial checkpoint`](../results/2026-11-10-optimized-full-grid-unit-l2/) |
| Automated tests | 14/14 pass, including equation-reference scatter, identity-step preservation, cache regression, no-probe-fit dependence, and PGM reader tests | `tests/` |
| Smoke run | Pass; one repeat at gallery sizes 2 and 5, not evidence | [`smoke-check`](../results/smoke-check/) |
| Independent algorithm-fidelity review | Paper equations, Theorem 2 and Fig. 3 checked; implementation assumptions remain documented for group review | [dated audit](experiments/2026-11-09-paper-proximate-reproduction.md) |
| Original-resolution fidelity | Original ORL and classifier match the stated setup; epsilon, equalization defaults, eigenvector scaling, and author splits remain undocumented. October stopping-rule results are historical. | [corrected experiment record](experiments/2026-11-09-paper-proximate-reproduction.md) |
| Final report and presentation outline | Updated to the original-resolution results, grid selection caveat, scaling sensitivity, and optimized implementation | [`report source`](../report/main.tex), [`rendered PDF`](../output/pdf/COMP3314_Group27_MDA_Reproduction.pdf), [`slides`](../slides/) |
| Project overview website | Updated with canonical grids and latest measured performance | [`site/index.html`](../site/index.html) |
| Public GitHub release for optional bonus | Group decision pending; current remote is private | Review repository contents, attribution, course policy, and data exclusions first |

## Repository cleanup — 2026-10-31

- Removed a historical VS Code screenshot that exposed a local path and unrelated workspace tabs, including its evidence-index entry.
- Removed the compiled PDF for the older 64×64 draft so it could not be mistaken for the current report; `report/main.tex` was later replaced with the original-resolution study and rebuilt as the current PDF.
- Rewrote Git history to remove the screenshot from earlier commits. The GitHub remote remains private.

## Completed work

- Compared Eigenfaces PCA, Fisherfaces LDA, MDA/2-1, and MDA/2-2 with Euclidean 1-nearest-neighbour.
- Reused deterministic matched splits across methods and fit projections using training data only.
- Preserved the initial five-split study rather than replacing its outputs.
- Ran a 30-split follow-up at gallery sizes 2, 3, 4, and 5 under both unit-L2 and within-scatter projection normalization.
- Saved split-level CSVs, summary CSV/JSON, charts, MDA objective traces, and paired comparison CSVs for each follow-up configuration.
- Added paired-difference analysis and documented the scope of its percentile bootstrap ranges.
- Added an in-place loader for the original Cambridge ORL PGM archive and retained its checksum and data provenance without committing face images.
- Ran original-resolution 30-split matched studies using the stopping condition printed in the paper's Fig. 3, plus a matched projection-normalization sensitivity run.
- Completed the earlier coarse five-repeat original-resolution sweep with 156 feasible MDA/2-2 settings; retained it as historical evidence alongside the later full 10,130-pair grids.
- Added this standalone repository, protocol and attribution notes, report and presentation scaffolds, and tests; nine tests passed at the latest recorded run.
- Ran a short smoke check from a cached dataset; it is excluded from research conclusions.
- Audited Fig. 3 directly and corrected the stopping threshold from `m'_k` to `m'_k × m_k × epsilon`; added an explicit epsilon setting and retained the October results with a correction note.
- Added a 64-output-level histogram equalization option aligned with MATLAB's documented default, while preserving the former 256-level setting as an alternative.
- Reran the original-resolution 30-split fixed 6×6 comparison under the corrected rule and both projection scaling conventions. The MDA/2-2 scale sensitivity remains substantial.
- Completed five-repeat searches across all 10,130 row/column dimension pairs under both projection scaling conventions (202,600 MDA/2-2 fits per grid).
- Audited test-set leakage: training projections are gallery-only, but best-by-grid dimensions are chosen on the probe scores later reported. Added leave-one-repeat-out sensitivity and a separate 30-repeat nested-galleries diagnostic at fixed 6×6.
- Profiled the local M3 Max run and fit stages; implemented per-split label grouping and first-iteration eigensystem reuse across dimension pairs. Benchmarked workers and BLAS threads on representative dimensions.
- Compared 30,390 rows from the stopped cached diagnostic against the canonical unit-L2 grid. Found two accuracy and 16 iteration differences in the initial cache implementation, traced them to shape-dependent floating-point behavior in nearly singular eigensystems, and changed the cache to reuse factorizations while projecting the requested dimensions directly.
- Rechecked only 17 flagged configurations under both axis normalizations. The corrected cached and direct fits matched exactly in predictions, accuracy, iterations, projections, distances, and objectives; direct fits matched canonical accuracy/iteration rows, and full-grid outputs were not rerun or overwritten.
- Rebenchmarked the corrected cache: 47.48 fits/s at 14 workers and one Accelerate thread (60 representative fits), versus 40.50 fits/s for direct fits at the same worker/BLAS setting.
- Rejected a slower SciPy generalized eigensolver alternative and did not port the small matrix problems to GPU.
- Prepared an optional AMD CPU SLURM job and documented HKU access requirements. No remote access or workload transfer was assumed.

## Next actions

1. Share the updated report, slide outline, website, and result links with the group for final presentation ownership and wording.
2. Do not present the probe-selected full-grid maxima as unbiased test accuracy; the report and site now mark this limitation.
3. A new nested dimension-selection evaluation is not required for the completed reproduction; add one only if the group wants a separate generalization estimate.

## Progress update format

Add a dated entry to `docs/CHANGELOG.md` for each meaningful source/document/artifact change. Every substantive experiment needs a record under `docs/experiments/` with command, configuration, seed policy, output paths, interpretation, and limitations. Git commits are the detailed change ledger; the changelog is the readable summary.
