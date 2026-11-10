# Experimental protocol

## Dataset and preprocessing

There are two intentionally separate protocols:

1. **Archived partial study:** scikit-learn's 64×64 Olivetti copy, retained so the earlier work and results remain reproducible.
2. **Original-resolution reproduction follow-up:** the official Cambridge `att_faces.zip`, decoded directly from PGM to 112-row × 92-column grayscale images. The archive lives at `.cache/orl-original/att_faces.zip` and is never committed.

- Both versions contain 400 grayscale faces, 40 identities, and 10 images per identity.
- The paper uses original-resolution grayscale images; the scikit-learn copy is downsampled to 64×64.
- Apply per-image 8-bit histogram equalization before splitting. This transform is image-local and uses no identity labels or population-level test statistics. The article does not specify the equalization implementation or number of levels. The current paper-proximate default is a 64-level approximation of MATLAB's documented default; the earlier 256-level transform remains in historical artifacts.
- Cache data under `.cache/`; do not add archives or face images to Git.

## Partitions

- Training/gallery counts: 2, 3, 4, and 5 images per identity. All remaining images for each identity are probes.
- Initial exploratory run: five repeated stratified random partitions per gallery size. Follow-up: 30 repeated partitions per gallery size without replacement. The dated artifacts remain separate; do not combine the five- and 30-split results.
- Original-resolution follow-up: use the same gallery counts and seed rule, but save results in a new dated folder. The paper does not publish exact split seeds or a repeat count, so this is a new reproducible comparison, not the authors' original realization.
- Targeted gallery-size diagnostic: a separate 30-repeat fixed-6×6 check uses nested galleries (the two-image gallery is a subset of the three-image gallery) to reduce split-to-split variation. These diagnostic splits are not mixed with the main independent-seed results.
- Seed rule: `base_seed + 1000 * train_count + repeat_index`, with `base_seed=3314` and zero-based `repeat_index`.
- Reuse each exact partition across all methods. Record the seed and split-level accuracy.
- Fit every projection using training images only; transform probe images after fitting.

## Methods

1. **Eigenfaces PCA:** flatten each image, learn leading variance directions from training vectors, and project.
2. **Fisherfaces LDA:** PCA pre-reduction followed by a regularized generalized-eigenvalue LDA projection.
3. **MDA/2-1:** one learned image-mode projection; the other mode remains identity, corresponding to the paper's 2-D LDA special case.
4. **MDA/2-2:** alternate row and column projection updates using mode-wise within-/between-class scatter; the fixed study uses 6×6, while the original-resolution dimension sweep records a predeclared row/column grid.

For the closest paper-style MDA run, initialize projections with identity matrices and update the row mode before the column mode. Fig. 3 says to stop after iteration 2 when each projection's Frobenius change is below `output_dimension × input_dimension × epsilon`. The article does not report `epsilon` or a maximum iteration count; our follow-up records `epsilon=1e-4` and caps at 10 iterations. We also expose stricter projection-change and objective-based stopping rules as sensitivity options. These choices are recorded in each run's metadata.

## Classifier and metric

- Euclidean 1-nearest-neighbour in each learned feature space for every method.
- Primary metric: probe accuracy per split.
- Summary: mean and sample standard deviation over all splits in a single dated run/configuration. The original-resolution fixed-setting follow-up reports 30 splits per method/gallery size.
- Paired analysis: compute MDA/2-2 minus each comparison method on the exact shared split; bootstrap the mean paired difference by resampling splits (10,000 draws, seed 881) and report the 2.5th and 97.5th percentiles. These ranges are conditional on this dataset and split-generation protocol, not population confidence intervals or significance tests.
- Preserve fit and classification time separately in new sweeps, and retain the final Fisher objective for the paper-style projection-change run. Historical full-grid timing fields used the name `fit_and_predict_seconds` but timed fitting only.

## Sensitivity and safeguards

Rerun the same split seed rule with `--projection-normalization within_scatter`; compare with `unit_l2`. Euclidean 1-NN is scale-sensitive, so do not choose normalization after viewing which gives a preferred ranking. For paper-style dimension sweeps, keep every per-dimension score. The paper reports best results across feature dimensions but does not give its complete ORL search grid or selection procedure; group best-probe-selected values are an optimistic diagnostic, not an unbiased test estimate.

- The archived partial run uses 64×64 faces; only the original-resolution follow-up uses 112×92 source images.
- In the fixed-setting comparison, feature dimensions are fixed (PCA 36, Fisherfaces up to 36, MDA 6 per projected mode); the paper reports best results across choices, so this table is not like-for-like with its best-tuned values.
- The original-resolution follow-up now supports a full feasible integer dimension grid. It cannot recover the authors' undocumented choices. Do not present the highest probe-selected grid result as unbiased accuracy.
- Dimension-sweep runs reuse class grouping and the first row/per-row column generalized eigensystem factorizations. They form each requested output dimension using the same shape-specific matrix multiplication as the direct fit. The canonical full grids were produced by the direct path; targeted comparisons validate the optimized cache without repeating all fits.
- The `best_by_gallery.csv` dimensions are selected with the same probe scores used for reporting. This is test-set leakage. Use the per-dimension results as descriptive reproduction evidence, and use training-only validation plus an untouched outer test set for an unbiased model-selection estimate.
- The five-split run is exploratory. The 30-split follow-up improves the description of split variability but still uses one fixed small dataset and does not establish population-wide uncertainty.
- Hyperparameter selection must use training-only or validation data, never final probe scores.
- Keep source-paper Table I separate from our results in text, charts, and filenames.
