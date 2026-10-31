# Project progress

**Last updated:** 2026-10-31 (Hong Kong time)

**Project:** COMP3314 Group 27, MDA face-recognition reproduction study

**Current phase:** both original-resolution fixed-setting runs and the five-repeat dimension sweep are complete; independent fidelity review and final coursework outputs are pending.

## At a glance

| Workstream | Status | Evidence |
|---|---|---|
| Paper and ORL claim notes | Done for current study | [paper notes](paper-notes.md) |
| Experiment implementation | Runnable 64×64 and original-resolution ORL implementations | [`src/run_experiment.py`](../src/run_experiment.py), [`src/orl_data.py`](../src/orl_data.py) |
| Initial five-split baseline | Complete; exploratory and retained as historical | [`2026-10-24 results`](../results/2026-10-24-partial-replication/) |
| Follow-up, 30 matched splits | Complete for current fixed settings | [dated record](experiments/2026-10-28-30-repeat-followup.md) |
| Normalization sensitivity | Complete for both run sizes; caveat required | Follow-up record and paired CSV outputs |
| Original-resolution ORL data | Complete; official 112×92 PGM files read from local archive | [data provenance](data-and-ethics.md), archive hash in run metadata |
| Original-resolution fixed study | Complete: 30 matched splits per gallery size with the Fig. 3 stopping rule | [`paper-faithful fixed run`](../results/2026-10-28-original-resolution-30-repeat/paper_figure_unit_l2/) |
| Dimension sweep | Complete: 5 repeats, 156 feasible MDA/2-2 dimension pairs, all split-level scores retained | [`dimension sweep`](../results/2026-10-28-original-resolution-dimension-sweep/paper_figure_unit_l2/), [heatmap](../results/2026-10-28-original-resolution-dimension-sweep/paper_figure_unit_l2/mda22_dimension_heatmap.svg) |
| Paired split analysis | Complete; conditional descriptive bootstrap ranges | `paired_differences.csv` in each follow-up configuration |
| Automated tests | Earlier 9/9 run predates original-ORL code additions; not rerun in this follow-up | `tests/` |
| Smoke run | Pass; one repeat at gallery sizes 2 and 5, not evidence | [`smoke-check`](../results/smoke-check/) |
| Independent algorithm-fidelity review | Pending | Ask a teammate to audit Sections II and IV and log discrepancies |
| Original-resolution fidelity | Data load, fixed-split studies, and declared dimension sweep complete; exact author settings are unavailable | [experiment record](experiments/2026-10-28-original-resolution-reproduction.md), [ORL outputs](../results/2026-10-28-original-resolution-30-repeat/) |
| Final report and presentation | Preliminary report source covers the older 64×64 study; final report and presentation still need group review | [`report source`](../report/main.tex), [`slides`](../slides/), and current result records |
| Project overview website | Complete; local and unpublished | [`site/index.html`](../site/index.html) |
| Public GitHub release for optional bonus | Group decision pending; current remote is private | Review repository contents, attribution, course policy, and data exclusions first |

## Repository cleanup — 2026-10-31

- Removed a historical VS Code screenshot that exposed a local path and unrelated workspace tabs, including its evidence-index entry.
- Removed the compiled PDF for the older 64×64 draft so it cannot be mistaken for the current report; the LaTeX source remains labeled as preliminary.
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
- Completed a five-repeat original-resolution dimension sweep with 156 feasible MDA/2-2 row/column settings, PCA/Fisherface/MDA/2-1 dimension curves, split-level results, objective traces, and a heatmap. The highest probe-selected MDA/2-2 means remain below Table I, even at the maximum dimensions.
- Added this standalone repository, protocol and attribution notes, report and presentation scaffolds, and tests; nine tests passed at the latest recorded run.
- Ran a short smoke check from a cached dataset; it is excluded from research conclusions.

## Next actions

1. Ask Person 2 to independently compare the implementation with Sections II and IV of the paper and audit the remaining Table I gap.
2. Give Person 4 the fixed-setting table, full dimension heatmap, paired differences, and mismatch caveats for the report and talk.
3. Review code, citations, dataset exclusions, and course policy as a group before any public release.

## Progress update format

Add a dated entry to `docs/CHANGELOG.md` for each meaningful source/document/artifact change. Every substantive experiment needs a record under `docs/experiments/` with command, configuration, seed policy, output paths, interpretation, and limitations. Git commits are the detailed change ledger; the changelog is the readable summary.
