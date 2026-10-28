# Initial partial reproduction — Person 3 handoff

**Run record:** retrospective from saved artifacts; baseline performed 2026-10-24.  
**Status:** completed exploratory run; partial reproduction, not exact reproduction.

## Configuration

- Data: scikit-learn 64×64 Olivetti/AT&T ORL copy (400 images, 40 identities, 10 images each); per-image 8-bit histogram equalization.
- Gallery counts: 2, 3, 4, 5 examples per identity; remaining images used as probes.
- Repeats: five stratified random splits per count; seed rule `3314 + 1000 * train_count + repeat_index` (zero-based repeat index).
- Classifier: Euclidean 1-nearest-neighbour. Methods share each split; projection fitting uses training images only.
- Models: Eigenfaces PCA (36 dimensions); Fisherfaces LDA (PCA pre-reduction then regularized LDA); MDA/2-1 (6×64); MDA/2-2 (6×6).
- Primary projection-axis convention: unit-L2. Sensitivity run: within-scatter normalization on the same seed rule.

## Group-computed results

Mean accuracy ± sample SD across five splits, percent:

| Train/person | Eigenfaces PCA | Fisherfaces LDA | MDA/2-1 | MDA/2-2 |
|---:|---:|---:|---:|---:|
| 2 | 67.1 ± 1.9 | 71.8 ± 2.8 | 73.5 ± 2.9 | 43.3 ± 4.8 |
| 3 | 74.5 ± 3.0 | 85.1 ± 3.2 | 83.7 ± 1.0 | 67.4 ± 3.0 |
| 4 | 81.3 ± 0.8 | 90.8 ± 2.2 | 87.1 ± 2.1 | 83.5 ± 1.4 |
| 5 | 84.0 ± 3.6 | 93.2 ± 1.2 | 89.8 ± 2.3 | 88.3 ± 2.8 |

## Interpretation

Accuracy generally rises with more gallery images. Under the primary setting, MDA/2-1 leads narrowly at two images per person; Fisherfaces leads from three through five. MDA/2-2 improves from 43.3% to 88.3% but does not lead in this fixed 6×6 setup. Five repeats are exploratory and do not establish a decisive ranking.

The scale sensitivity is important: at two gallery images per identity, Fisherfaces changes from 71.8% with unit-L2 axes to 29.0% with within-scatter axes; MDA/2-2 changes from 43.3% to 19.9%. State the convention used and report this as a limitation. Euclidean nearest-neighbour distance changes when feature axes are rescaled.

## Comparison with the paper

Paper Table I reports best ORL accuracies for Eigenface / MDA/2-2 of 81.6/90.1%, 87.7/95.0%, 91.3/98.3%, and 97.5/99.0% for 2/3/4/5 gallery images per person. Those are source-paper values, not our repeated-split means. Our 64×64 copy, fixed feature dimensions, and five random splits differ from the paper's original image resolution and best-result protocol.

## Artifacts

- [Primary output](../../results/2026-10-24-partial-replication/unit_l2/): raw rows, summary CSV/JSON, chart, and objective trace.
- [Sensitivity output](../../results/2026-10-24-partial-replication/within_scatter/): matched-seed alternative normalization.
- [Experiment source](../../src/run_experiment.py).

Suggested oral summary: “We compared four projections on the same repeated ORL splits. MDA/2-1 had a small edge with two training photos per person, while Fisherfaces led with three to five. MDA/2-2 improved as data increased but did not win in this fixed setup. Results were sensitive to eigenvector scaling, so this is a partial reproduction rather than an exact match to the paper.”
