# Research plan

## Purpose

Reproduce and critically examine the central ORL face-recognition experiment in Yan et al. (2007): compare tensor/matrix-based MDA against vector-based baselines under controlled gallery/probe splits, then explain where the group's evidence matches or diverges from the paper.

## Questions

**Primary question:** On matched small-sample ORL splits, how does MDA/2-2 perform relative to Eigenfaces, Fisherfaces, and MDA/2-1 when all methods use nearest-neighbour classification?

**Secondary questions:**

1. How does performance change as gallery size rises from two to five images per identity?
2. How much do implementation details—especially feature dimensionality and generalized-eigenvector scaling—affect Euclidean 1-NN?
3. Which differences between our protocol and the paper prevent a direct numerical reproduction?

## Claims and hypotheses

- H1 (paper claim to examine): MDA/2-2 can outperform vector-based methods, with a larger advantage in small-sample settings.
- H2 (data-size trend): more gallery images per person will usually improve identification accuracy.
- H3 (implementation sensitivity): changing projection-axis normalization can change 1-NN outcomes because Euclidean distance is scale-sensitive.

These are questions, not assumed conclusions. Report negative or mixed results as they occur.

## Current study design

The initial baseline and 30-split follow-up use scikit-learn's 400-image, 40-identity, 64×64 Olivetti copy. A fidelity extension now loads the official original-resolution 112×92 ORL archive, uses the same 2/3/4/5 gallery counts, per-image histogram equalization, matched seeds, and Euclidean 1-NN. It also sweeps feature dimensions, retaining every per-dimension result. Full details and deviations are in [protocol](protocol.md).

## Planned research stages

| Stage | Purpose | Status | Exit evidence |
|---|---|---|---|
| A. Secure baseline | Freeze code, settings, raw split rows, and scale-sensitivity run | Complete | Dated run record and saved CSV/JSON/SVG |
| B. Verification | Independent code review, paper Table I audit, protocol audit, and rerun review | In progress | Person 2's discrepancy checklist and confirmed table transcription |
| C. Stronger estimate | Increase repeated splits to 30 and analyze paired per-split differences | Complete for current fixed settings | Raw rows, summaries, paired bootstrap ranges, charts, dated log |
| D. Fidelity extension | Reproduce the ORL experiment on original-resolution data and compare paper-style best dimensions | Complete as a documented follow-up; exact author run unavailable | Original archive hash, loader, full raw sweep, published-table comparison, and caveats |
| E. Coursework outputs | Final 3–4-page LaTeX report and 9-minute slide deck with a 3-minute Q&A plan | In progress | Report and slides reviewed by all four members |

The stronger estimate and fidelity extension must not overwrite Stage A. Keep every setting in a separate named results folder.

## Analysis plan

- Report mean accuracy and sample SD over repeated splits, and retain every split-level score.
- Since methods share splits, report paired differences as well as separate means in the stronger-estimate stage.
- Treat the five-split SD as exploratory description only. The 30-split percentile bootstrap ranges describe split-generation variability conditional on this fixed dataset; they are not population-level confidence intervals or significance tests.
- Include the MDA training objective trace as an optimization diagnostic only; it is not test accuracy.
- Compare with the article's reported best accuracies only as context because data resolution, selected dimensions, and aggregation differ.

## Course assessment mapping

- **Paper understanding (40%):** formulation, motivation, DTC, alternating mode optimization, and links to PCA/LDA/nearest neighbours.
- **Critical analysis & insight (30%):** assumptions, evidence, scaling sensitivity, protocol mismatch, small-sample uncertainty, and proposed validation.
- **Written report (10%):** concise, cited, clearly labeled group-produced measurements.
- **Presentation & Q&A (20%):** 9 minutes plus 3 minutes, with all members ready to explain methods and caveats.
- **Optional reproduction bonus (up to 10):** the course brief requests a public GitHub repository with setup/run documentation and clear attribution if code is submitted. Review access and contents with the group before making this working repository public.
