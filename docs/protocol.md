# Experimental protocol

## Dataset and preprocessing

There are two intentionally separate protocols:

1. **Archived partial study:** scikit-learn's 64×64 Olivetti copy, retained so the earlier work and results remain reproducible.
2. **Original-resolution reproduction follow-up:** the official Cambridge `att_faces.zip`, decoded directly from PGM to 112-row × 92-column grayscale images. The archive lives at `.cache/orl-original/att_faces.zip` and is never committed.

- Both versions contain 400 grayscale faces, 40 identities, and 10 images per identity.
- The paper uses original-resolution grayscale images; the scikit-learn copy is downsampled to 64×64.
- Apply per-image 8-bit histogram equalization before splitting. This transform is image-local and uses no identity labels or population-level test statistics.
- Cache data under `.cache/`; do not add archives or face images to Git.

## Partitions

- Training/gallery counts: 2, 3, 4, and 5 images per identity. All remaining images for each identity are probes.
- Initial exploratory run: five repeated stratified random partitions per gallery size. Follow-up: 30 repeated partitions per gallery size without replacement. The dated artifacts remain separate; do not combine the five- and 30-split results.
- Original-resolution follow-up: use the same gallery counts and seed rule, but save results in a new dated folder. The paper does not publish exact split seeds or a repeat count, so this is a new reproducible comparison, not the authors' original realization.
- Seed rule: `base_seed + 1000 * train_count + repeat_index`, with `base_seed=3314` and zero-based `repeat_index`.
- Reuse each exact partition across all methods. Record the seed and split-level accuracy.
- Fit every projection using training images only; transform probe images after fitting.

## Methods

1. **Eigenfaces PCA:** flatten each image, learn leading variance directions from training vectors, and project.
2. **Fisherfaces LDA:** PCA pre-reduction followed by a regularized generalized-eigenvalue LDA projection.
3. **MDA/2-1:** one learned image-mode projection; the other mode remains identity, corresponding to the paper's 2-D LDA special case.
4. **MDA/2-2:** alternate row and column projection updates using mode-wise within-/between-class scatter; the fixed study uses 6×6, while the original-resolution dimension sweep records a predeclared row/column grid.

For the paper-faithful MDA run, initialize projections with identity matrices, update the row mode before the column mode, and apply the stopping condition printed in Fig. 3: after iteration 2, stop when each projection's Frobenius change is smaller than its requested output dimension. The paper does not state a maximum iteration count; our run caps it at 10. We also expose a stricter projection tolerance and objective-based stopping mode as sensitivity options. These details are selectable and are recorded in each run's metadata.

## Classifier and metric

- Euclidean 1-nearest-neighbour in each learned feature space for every method.
- Primary metric: probe accuracy per split.
- Summary: mean and sample standard deviation over all splits in a single dated run/configuration. The original-resolution fixed-setting follow-up reports 30 splits per method/gallery size.
- Paired analysis: compute MDA/2-2 minus each comparison method on the exact shared split; bootstrap the mean paired difference by resampling splits (10,000 draws, seed 881) and report the 2.5th and 97.5th percentiles. These ranges are conditional on this dataset and split-generation protocol, not population confidence intervals or significance tests.
- Also preserve fit-and-predict runtime and MDA training-objective values by iteration.

## Sensitivity and safeguards

Rerun the same split seed rule with `--projection-normalization within_scatter`; compare with `unit_l2`. Euclidean 1-NN is scale-sensitive, so do not choose normalization after viewing which gives a preferred ranking. For paper-style dimension sweeps, keep every per-dimension score. The paper reports best results across feature dimensions but does not give its complete ORL search grid or selection procedure; group best-probe-selected values are an optimistic diagnostic, not an unbiased test estimate.

- The archived partial run uses 64×64 faces; only the original-resolution follow-up uses 112×92 source images.
- In the fixed-setting comparison, feature dimensions are fixed (PCA 36, Fisherfaces up to 36, MDA 6 per projected mode); the paper reports best results across choices, so this table is not like-for-like with its best-tuned values.
- The original-resolution sweep searches a declared, coarse grid and retains the complete curves; it cannot recover the authors' undocumented choices. Do not present the highest probe-selected grid result as unbiased accuracy.
- The five-split run is exploratory. The 30-split follow-up improves the description of split variability but still uses one fixed small dataset and does not establish population-wide uncertainty.
- Hyperparameter selection must use training-only or validation data, never final probe scores.
- Keep source-paper Table I separate from our results in text, charts, and filenames.
