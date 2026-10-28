# Original-resolution ORL reproduction follow-up

**Date:** 2026-10-28 (Hong Kong time)
**Owner:** Person 3
**Status:** fixed-setting runs and five-repeat dimension sweep complete; independent code audit pending
**Paper:** Yan et al. (2007), [Multilinear Discriminant Analysis for Face Recognition](https://mmlab.ie.cuhk.edu.hk/archive/2007/IP07_face02.pdf)

## Purpose

The earlier group study used the 64×64 scikit-learn copy of Olivetti faces. This follow-up uses the original 112-row × 92-column PGM images and the paper's stated per-image histogram equalization, then compares Eigenfaces PCA, Fisherfaces LDA, MDA/2-1, and MDA/2-2 with the same gallery/probe split and Euclidean 1-nearest-neighbour classifier.

The paper describes random gallery/probe partitions and reports its best accuracy across feature dimensions. It does not publish the random seeds, number of random partitions, or complete ORL dimension search and selection procedure. Therefore, this is an original-resolution, paper-style reproduction study, not a bit-for-bit recreation of the authors' reported Table I.

## Data and integrity

- Source: official [Cambridge AT&T/ORL archive](https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip).
- Dataset: 400 PGM images, 40 identities, 10 images per identity; each image decodes to 112×92.
- Archive SHA-256: `509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec`.
- Archive path: `.cache/orl-original/att_faces.zip`; it is ignored by Git and no face images are committed.
- Loader: [`src/orl_data.py`](../../src/orl_data.py), which reads the archive without extracting image files.

## Fixed-setting protocol

- Gallery sizes: 2, 3, 4, and 5 images per identity; remaining images are probes.
- Repeats: 30 matched stratified splits for each gallery size.
- Seed rule: `3314 + 1000 * gallery_count + zero_based_repeat`.
- Preprocessing: independent 8-bit histogram equalization per image.
- Classifier: Euclidean 1-NN for all four methods.
- PCA: 36 features. Fisherfaces: up to 36 LDA outputs, with full `N-C` PCA pre-reduction. MDA/2-1: 6 row features. MDA/2-2: 6×6 features.
- MDA initialization and order: identity projections; update row then column.
- MDA stopping: use the Fig. 3 rule as printed in the paper—after iteration 2, stop if each projection's Frobenius change is below its requested output dimension. Since the article does not report a maximum iteration count, cap at 10.
- Primary generalized-eigenvector scaling: unit-L2 axes. A second run repeats the same split seeds with within-scatter normalized axes as a sensitivity check.
- Fitting: projection matrices use gallery/training images only.

Commands from the repository root:

```bash
python3 src/run_experiment.py --dataset orl-original \
  --projection-normalization unit_l2 --mda-stopping-rule paper_figure \
  --repeats 30 --max-iter 10 --fisher-pca-max 10400 \
  --output-dir results/2026-10-28-original-resolution-30-repeat/paper_figure_unit_l2

python3 src/run_experiment.py --dataset orl-original \
  --projection-normalization within_scatter --mda-stopping-rule paper_figure \
  --repeats 30 --max-iter 10 --fisher-pca-max 10400 \
  --output-dir results/2026-10-28-original-resolution-30-repeat/paper_figure_within_scatter

python3 scripts/analyze_paired_results.py \
  results/2026-10-28-original-resolution-30-repeat/paper_figure_unit_l2/raw_results.csv
```

## Fixed-setting results

Mean accuracy ± sample SD across the 30 splits (percent):

| Gallery images/person | Eigenfaces PCA | Fisherfaces LDA | MDA/2-1 | MDA/2-2 (6×6) |
|---:|---:|---:|---:|---:|
| 2 | 73.8 ± 2.5 | 73.1 ± 2.9 | 81.4 ± 2.9 | 42.7 ± 5.5 |
| 3 | 84.0 ± 1.8 | 84.9 ± 2.8 | 89.9 ± 2.2 | 72.0 ± 3.9 |
| 4 | 88.2 ± 3.1 | 88.2 ± 2.4 | 92.1 ± 2.1 | 85.4 ± 2.2 |
| 5 | 91.6 ± 2.0 | 90.4 ± 2.2 | 94.6 ± 1.6 | 91.4 ± 2.3 |

The article's Table I reports best accuracies of 81.6/87.7/91.3/97.5 for Eigenface, 79.7/87.5/85.8/92.0 for Fisherface/O, 88.1/94.3/96.3/98.5 for MDA/2-1, and 90.1/95.0/98.3/99.0 for MDA/2-2 at gallery sizes 2/3/4/5 respectively. These are context, not a like-for-like test: our fixed run uses preselected feature sizes and repeated new splits, while the paper reports best-over-dimension results and Fisherface/O.

The main mismatch is MDA/2-2 at small gallery sizes. Its fixed 6×6 representation scores 42.7% with two gallery images and 72.0% with three. MDA/2-1 does better in those settings. The same 6×6 result improves to 85.4% and 91.4% at four and five gallery images. The optimization trace for 6×6 MDA/2-2 rises across its three iterations, while probe accuracy remains low at small gallery sizes; this is consistent with a gap between the training Fisher objective and nearest-neighbour generalization, but does not by itself identify the cause.

The within-scatter normalization sensitivity run strongly changes both Fisherfaces and MDA/2-2 scores. This confirms that projection-axis scaling is a material implementation choice for Euclidean 1-NN. It must be presented as sensitivity, not selected after seeing which gives a preferred conclusion.

With the Fig. 3 stopping rule matched, within-scatter MDA/2-2 means are 5.3%, 30.0%, 69.1%, and 84.8% for gallery sizes 2/3/4/5, compared with 42.7%, 72.0%, 85.4%, and 91.4% under unit-L2 axes. Within-scatter Fisherfaces fall to 33.0%/39.2%/39.0%/42.5% versus 73.1%/84.9%/88.2%/90.4%. This large sensitivity underscores that the scaling convention and classifier geometry need review. Earlier folders `projection_change_unit_l2/` and `projection_change_within_scatter/` use a stricter projection-change threshold and are exploratory; do not mix them with the `paper_figure_*` primary pair.

## Dimension sweep

A second script retains per-split results over a coarse 2D MDA feature grid, and full feature-dimension curves for PCA, Fisherfaces, and MDA/2-1. The completed study uses 5 repeated splits per gallery size, unit-L2 projection axes, row-first updates, and the Fig. 3 stopping condition. MDA/2-2 searches 13 feasible row sizes and 12 feasible column sizes (156 combinations); the maximum feasible feature space is 112×92. It saves `raw_sweep.csv` (8,920 rows), `dimension_summary.csv`, `best_by_gallery.csv`, `mda_objectives.csv`, `metadata.csv`, and the paper's published Table I values in `published_table_i.csv`. Checkpoint files are written after every completed split; the script can resume with `--resume`.

The best-by-gallery file selects the highest mean probe accuracy over the grid and is deliberately marked as optimistic/test-selected. Use it to understand the reported best-score style and to diagnose dimension sensitivity, not as an unbiased final estimate. Always show the full grid or label the selection clearly.

| Gallery images/person | Paper MDA/2-2 best | Group fixed 6×6, 30 splits | Highest group grid mean, 5 splits | Selected row×column dims | Paper minus grid |
|---:|---:|---:|---:|---:|---:|
| 2 | 90.1% | 42.7 ± 5.5% | 62.9% | 112×92 | 27.2 pp |
| 3 | 95.0% | 72.0 ± 3.9% | 83.6% | 32×16 | 11.4 pp |
| 4 | 98.3% | 85.4 ± 2.2% | 91.8% | 32×24 | 6.6 pp |
| 5 | 99.0% | 91.4 ± 2.3% | 94.6% | 48×24 | 4.4 pp |

The best grid cell at two gallery images uses the full 112×92 dimension (10,304 features), yet its five-split mean is 62.9%, still far below the paper's 90.1%. Dimension choice explains part, but not all, of the fixed-setting gap. The grid's maxima use the same probe splits for selection and measurement, so the listed grid means are optimistic. A teammate should audit the eigenproblem, axis scaling, and published-table protocol before attributing the remaining gap to the paper or algorithm.

## Outputs

Fixed unit-L2 results: [`paper_figure_unit_l2`](../../results/2026-10-28-original-resolution-30-repeat/paper_figure_unit_l2/)

Matched normalization sensitivity: [`paper_figure_within_scatter`](../../results/2026-10-28-original-resolution-30-repeat/paper_figure_within_scatter/)

Dimension sweep: [`paper_figure_unit_l2`](../../results/2026-10-28-original-resolution-dimension-sweep/paper_figure_unit_l2/) · [MDA/2-2 heatmap](../../results/2026-10-28-original-resolution-dimension-sweep/paper_figure_unit_l2/mda22_dimension_heatmap.svg)
Implementation: [`run_experiment.py`](../../src/run_experiment.py) and [`run_orl_dimension_sweep.py`](../../scripts/run_orl_dimension_sweep.py)

## Interpretation and limitations

1. The same 40 identities, original-resolution data, preprocessing, gallery sizes, and classifier are used, but the authors' random splits and full hyperparameter search are unavailable.
2. The fixed 6×6 MDA/2-2 model does not reproduce Table I's strong small-sample result. The wider dimension sweep may identify more successful settings, but because probe scores guide selection, those maxima are optimistic.
3. The Fisherfaces implementation uses a full `N-C` PCA subspace in the dimension sweep; the fixed study uses up to 36 LDA outputs. The paper's reported Fisherface/O is a dimension-selected baseline, so fixed Fisherfaces should not be labeled Fisherface/O.
4. The paper's generalized-eigenvector scaling and regularization details are not fully specified. The group records unit-L2 and within-scatter choices explicitly; nearest-neighbour distances depend on feature scaling.
5. The bootstrap intervals from paired split differences are conditional descriptions of variability over this fixed dataset and split generator, not population confidence intervals or significance tests.

## Handoff

- Person 2: independently audit mode-wise scatter, eigenvector ordering/scaling, stopping condition, Table I transcription, and data protocol; report discrepancies before any result is called final.
- Person 4: use the fixed table and complete dimension curves, distinguish group measurements from the paper's best scores, and state the MDA/2-2 small-sample mismatch plainly.
- Person 1: confirm the explanation of DTC and the alternating update matches the method section.
