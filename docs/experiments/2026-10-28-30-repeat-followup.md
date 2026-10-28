# Experiment record: 30-split paired follow-up

**Run date:** 2026-10-28 (Hong Kong time)

**Status:** completed; group implementation, partial reproduction

**Question:** Are the initial five-split findings stable across more matched gallery/probe partitions, and how sensitive are they to MDA projection-axis normalization?

## Configuration

- Dataset: scikit-learn's 64×64 Olivetti copy; 400 grayscale images, 40 identities, 10 images per identity. No test data are used to fit projections.
- Preprocessing: per-image 8-bit histogram equalization.
- Gallery sizes: 2, 3, 4, and 5 training images per identity; remaining images are probes.
- Splits: 30 deterministic stratified splits per gallery size, shared across all methods. Seed formula: `3314 + 1000 * train_images_per_identity + repeat_index`.
- Methods: Eigenfaces PCA (36 dimensions), Fisherfaces LDA (PCA pre-reduction plus regularized LDA), MDA/2-1, and MDA/2-2 (6×6 output).
- Classifier: Euclidean 1-nearest-neighbour.
- MDA settings: identity initialization, at most 10 alternating updates, relative objective tolerance 1e-4, ridge 1e-6 for singular scatter matrices.
- Two configurations: `unit_l2` (primary) and `within_scatter` (normalization sensitivity).
- Paired analysis: MDA/2-2 minus each comparator on the exact same split; 10,000 split-resampling bootstrap draws, seed 881; percentile range [2.5%, 97.5%].

## Commands

```bash
python src/run_experiment.py --repeats 30 \
  --output-dir results/2026-10-28-30-repeat-followup/unit_l2
python src/run_experiment.py --projection-normalization within_scatter \
  --repeats 30 \
  --output-dir results/2026-10-28-30-repeat-followup/within_scatter
python scripts/analyze_paired_results.py \
  results/2026-10-28-30-repeat-followup/unit_l2/raw_results.csv
python scripts/analyze_paired_results.py \
  results/2026-10-28-30-repeat-followup/within_scatter/raw_results.csv
```

## Primary results

Mean accuracy ± sample SD across 30 splits, percent:

| Gallery images/person | PCA | Fisherfaces | MDA/2-1 | MDA/2-2 |
|---:|---:|---:|---:|---:|
| 2 | 65.7 ± 2.3 | 70.6 ± 2.6 | **72.5 ± 3.5** | 42.4 ± 4.6 |
| 3 | 75.3 ± 2.6 | **84.7 ± 2.9** | 83.2 ± 2.4 | 66.7 ± 3.7 |
| 4 | 81.0 ± 3.6 | **90.0 ± 2.2** | 87.4 ± 2.5 | 82.2 ± 3.3 |
| 5 | 85.0 ± 2.4 | **93.3 ± 1.3** | 90.2 ± 2.3 | 88.1 ± 2.6 |

These are group-computed results, not the paper's Table I. MDA/2-1 leads at two gallery images; Fisherfaces leads at three to five. MDA/2-2 accuracy rises with gallery size but does not lead in this fixed 6×6 configuration.

## Normalization sensitivity

MDA/2-2 mean accuracy (percent) under the two conventions:

| Gallery images/person | Unit-L2 | Within-scatter |
|---:|---:|---:|
| 2 | 42.4 ± 4.6 | 20.1 ± 7.5 |
| 3 | 66.7 ± 3.7 | 62.6 ± 3.9 |
| 4 | 82.2 ± 3.3 | 80.3 ± 3.8 |
| 5 | 88.1 ± 2.6 | 87.5 ± 2.8 |

The largest difference appears with only two training examples per identity. Euclidean distance changes when feature axes are rescaled, so this sensitivity is a property of the chosen representation/metric/normalization combination. It is not evidence that MDA in general fails.

## Paired comparison examples

Paired difference is MDA/2-2 minus the named method, in percentage points. Ranges below are percentile bootstrap ranges across matched splits, conditional on this fixed dataset and split-generation procedure.

| Gallery | Comparator | Mean paired difference | 95% conditional bootstrap range | MDA/2-2 wins |
|---:|---|---:|---:|---:|
| 2 | MDA/2-1 | −30.14 | [−31.96, −28.34] | 0/30 |
| 2 | Fisherfaces | −28.17 | [−29.93, −26.43] | 0/30 |
| 5 | PCA | +3.08 | [+2.05, +4.23] | 24/30 |
| 5 | Fisherfaces | −5.15 | [−6.20, −4.17] | 0/30 |
| 5 | MDA/2-1 | −2.13 | [−3.02, −1.27] | 5/30 |

These intervals are descriptive resampling ranges, not population-level confidence intervals or significance tests. Thirty partitions of the same 400 images do not create independent new datasets.

## Interpretation and limits

- The larger repeat count makes split-to-split variation more visible than the original five-split exploratory run.
- The qualitative ranking is consistent with the initial run: MDA/2-1 is strongest at the smallest gallery, and Fisherfaces leads at larger gallery sizes under the primary normalization.
- MDA/2-2 changes substantially under an alternate axis normalization, especially at gallery size two; report both settings and do not select one after seeing which gives the preferred result.
- This remains a partial reproduction: images are 64×64 rather than the paper's original 92×112 ORL images, dimensions are fixed rather than tuned to recreate the paper's best configuration, and our random splits are not the paper's original splits.
- The 30-split estimate does not replace independent algorithm review or validation-based feature selection.

## Artifacts

- `results/2026-10-28-30-repeat-followup/unit_l2/`
- `results/2026-10-28-30-repeat-followup/within_scatter/`
- Paired analysis implementation: `scripts/analyze_paired_results.py`
