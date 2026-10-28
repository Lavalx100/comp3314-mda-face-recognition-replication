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

The baseline uses scikit-learn's 400-image, 40-identity, 64×64 Olivetti copy; per-image 8-bit histogram equalization; 2/3/4/5 training images per identity; five deterministic repeated stratified splits per count; and a shared Euclidean 1-NN classifier. It compares Eigenfaces PCA, Fisherfaces LDA, MDA/2-1, and MDA/2-2. Full details and deviations are in [protocol](protocol.md).

## Planned research stages

| Stage | Purpose | Status | Exit evidence |
|---|---|---|---|
| A. Secure baseline | Freeze code, settings, raw split rows, and scale-sensitivity run | Complete | Dated run record and saved CSV/JSON/SVG |
| B. Verification | Unit tests, independent code review, table-value audit, rerun check | In progress | Passing tests and protocol audit |
| C. Stronger estimate | Increase repeated splits (target at least 30) and analyze paired per-split differences | Planned | Raw rows, uncertainty intervals, paired plots, dated log |
| D. Fidelity extension | If permitted source data can be obtained, test original-resolution ORL preprocessing and paper feature settings | Planned / access dependent | Provenance/terms record and configuration; otherwise document why unavailable |
| E. Coursework outputs | Final 3–4-page LaTeX report and 9-minute slide deck with a 3-minute Q&A plan | In progress | Report and slides reviewed by all four members |

The stronger estimate and fidelity extension must not overwrite Stage A. Keep every setting in a separate named results folder.

## Analysis plan

- Report mean accuracy and sample SD over repeated splits, and retain every split-level score.
- Since methods share splits, report paired differences as well as separate means in the stronger-estimate stage.
- Do not call a five-split SD a confidence interval or claim statistical significance from it.
- Include the MDA training objective trace as an optimization diagnostic only; it is not test accuracy.
- Compare with the article's reported best accuracies only as context because data resolution, selected dimensions, and aggregation differ.

## Course assessment mapping

- **Paper understanding (40%):** formulation, motivation, DTC, alternating mode optimization, and links to PCA/LDA/nearest neighbours.
- **Critical analysis & insight (30%):** assumptions, evidence, scaling sensitivity, protocol mismatch, small-sample uncertainty, and proposed validation.
- **Written report (10%):** concise, cited, clearly labeled group-produced measurements.
- **Presentation & Q&A (20%):** 9 minutes plus 3 minutes, with all members ready to explain methods and caveats.
- **Optional reproduction bonus (up to 10):** the course brief requests a public GitHub repository with setup/run documentation and clear attribution if code is submitted. Review access and contents with the group before making this working repository public.
