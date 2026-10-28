# Project progress

**Last updated:** 2026-10-28 (Hong Kong time)

**Project:** COMP3314 Group 27, MDA face-recognition partial reproduction

**Current phase:** initial baseline frozen; 30-split follow-up complete; independent fidelity review and final coursework outputs pending.

## At a glance

| Workstream | Status | Evidence |
|---|---|---|
| Paper and ORL claim notes | Done for current study | [paper notes](paper-notes.md) |
| Experiment implementation | Runnable group implementation | [`src/run_experiment.py`](../src/run_experiment.py) |
| Initial five-split baseline | Complete; exploratory and retained as historical | [`2026-10-24 results`](../results/2026-10-24-partial-replication/) |
| Follow-up, 30 matched splits | Complete for current fixed settings | [dated record](experiments/2026-10-28-30-repeat-followup.md) |
| Normalization sensitivity | Complete for both run sizes; caveat required | Follow-up record and paired CSV outputs |
| Paired split analysis | Complete; conditional descriptive bootstrap ranges | `paired_differences.csv` in each follow-up configuration |
| Automated tests | 9/9 passed at latest recorded run | `python -m unittest discover -s tests -v` |
| Smoke run | Pass; one repeat at gallery sizes 2 and 5, not evidence | [`smoke-check`](../results/smoke-check/) |
| Independent algorithm-fidelity review | Pending | Ask a teammate to audit Sections II and IV and log discrepancies |
| Original-resolution follow-up | Pending / data-access dependent | Check data terms and original protocol before use |
| Final report and presentation | Three-page report PDF and presentation outline are drafts; group review pending | [`report source`](../report/main.tex), [`compiled PDF`](../output/pdf/COMP3314_Group27_MDA_Partial_Reproduction.pdf), and [`slides`](../slides/) |
| Project overview website | Complete; local and unpublished | [`site/index.html`](../site/index.html) |
| Public GitHub release for optional bonus | Group decision pending; current remote is private | Review repository contents, attribution, course policy, and data exclusions first |

## Completed work

- Compared Eigenfaces PCA, Fisherfaces LDA, MDA/2-1, and MDA/2-2 with Euclidean 1-nearest-neighbour.
- Reused deterministic matched splits across methods and fit projections using training data only.
- Preserved the initial five-split study rather than replacing its outputs.
- Ran a 30-split follow-up at gallery sizes 2, 3, 4, and 5 under both unit-L2 and within-scatter projection normalization.
- Saved split-level CSVs, summary CSV/JSON, charts, MDA objective traces, and paired comparison CSVs for each follow-up configuration.
- Added paired-difference analysis and documented the scope of its percentile bootstrap ranges.
- Added this standalone repository, protocol and attribution notes, report and presentation scaffolds, and tests; nine tests passed at the latest recorded run.
- Ran a short smoke check from a cached dataset; it is excluded from research conclusions.

## Next actions

1. Ask a teammate to independently compare the implementation with Sections II and IV of the paper; record discrepancies and corrections.
2. Decide whether original-resolution ORL data may be used under its terms. If not, keep the current 64×64 study labeled partial.
3. Finish and review the 3–4-page LaTeX report and 9-minute talk; rehearse the 3-minute Q&A and make sure all members understand the caveats.
4. Review code, citations, dataset exclusions, and course policy as a group before any public release.

## Progress update format

Add a dated entry to `docs/CHANGELOG.md` for each meaningful source/document/artifact change. Every substantive experiment needs a record under `docs/experiments/` with command, configuration, seed policy, output paths, interpretation, and limitations. Git commits are the detailed change ledger; the changelog is the readable summary.
