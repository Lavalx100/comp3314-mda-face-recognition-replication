# Project progress

**Last updated:** 2026-10-28 (Hong Kong time)  
**Project:** COMP3314 Group 27, MDA face-recognition partial reproduction  
**Current phase:** baseline frozen; verification and higher-repeat follow-up pending.

## At a glance

| Workstream | Status | Evidence |
|---|---|---|
| Paper and ORL claim notes | Done for initial study | [paper notes](paper-notes.md) |
| Experiment implementation | Baseline complete | [`src/run_experiment.py`](../src/run_experiment.py) |
| Baseline results, five matched splits | Complete; exploratory | [`results/2026-10-24-partial-replication/`](../results/2026-10-24-partial-replication/) |
| Feature-scaling sensitivity | Complete; caveat required | Dated experiment record and paired outputs |
| Automated tests | 7/7 pass locally | `python -m unittest discover -s tests -v` |
| Smoke run | Pass; 1 split at gallery sizes 2 and 5, not evidence | [`results/smoke-check/`](../results/smoke-check/) |
| Independent algorithm-fidelity review | Pending | Assign reviewer and record discrepancies |
| 30+ repeat paired analysis | Pending | New output folder plus dated experiment note |
| Original-resolution follow-up | Pending / data-access dependent | Check data terms and protocol details first |
| Final report and presentation | Draft scaffold added | [`report/`](../report/) and [`slides/`](../slides/) |
| Public GitHub release for optional bonus | Not yet approved by group | Keep working copy private until review |

## Completed work

- Implemented a shared comparison of Eigenfaces PCA, Fisherfaces LDA, MDA/2-1, and MDA/2-2 with Euclidean 1-NN.
- Used the same deterministic stratified split per method; fit projections on training data only.
- Saved split-level CSV, mean/SD summaries, chart, JSON configuration, and MDA objective trace.
- Ran a second eigenvector-normalization setting with the same seed rule to expose 1-NN scale sensitivity.
- Wrote a report-ready interpretation that separates group results from the source paper's Table I.
- Added this standalone project structure, documentation, and unit-test scaffold; seven tests pass.
- Ran a short smoke check from the cached dataset; outputs are explicitly excluded from research conclusions.

## Next actions

1. Run unit tests and a one-repeat smoke test on a clean environment.
2. Have a teammate compare implementation steps with Sections II and IV of the paper; log any discrepancies.
3. Freeze a higher-repeat protocol (recommended: at least 30 matched splits) before running it; report paired differences and uncertainty, not only separate means.
4. Decide whether original-resolution ORL data can be used under its terms. If not, keep the 64×64 study labeled partial.
5. Finish the 3–4-page LaTeX report and 9-minute talk; rehearse Q&A and ensure all members understand the caveat.
6. Review code, citations, dataset exclusions, and course policy as a group before any public release.

## Progress update format

Add a dated entry to `docs/CHANGELOG.md` for each meaningful source/document/artifact change. Every substantive experiment also needs its own record under `docs/experiments/` with command, software, dataset, seeds, outputs, and interpretation. Git commits are the detailed change ledger; the changelog is the readable summary.
