# Experimental protocol

## Dataset and preprocessing

- Use `sklearn.datasets.fetch_olivetti_faces`, an AT&T ORL copy: 400 grayscale faces, 40 identities, 10 images each, distributed by scikit-learn at 64×64 pixels.
- The source paper's ORL experiment used original-resolution images (112 rows × 92 columns); scikit-learn documents its copy as 64×64. This is a known fidelity difference.
- Apply per-image 8-bit histogram equalization before splitting. This transform is image-local and uses no identity labels or population-level test statistics.
- Cache data under `.cache/olivetti/`; do not add the cache or face images to Git.

## Partitions

- Training/gallery counts: 2, 3, 4, and 5 images per identity. All remaining images for each identity are probes.
- Generate five repeated stratified random partitions per gallery size without replacement.
- Seed rule: `base_seed + 1000 * train_count + repeat_index`, with `base_seed=3314` and zero-based `repeat_index`.
- Reuse each exact partition across all methods. Record the seed and split-level accuracy.
- Fit every projection using training images only; transform probe images after fitting.

## Methods

1. **Eigenfaces PCA:** flatten each image, learn leading variance directions from training vectors, and project.
2. **Fisherfaces LDA:** PCA pre-reduction followed by a regularized generalized-eigenvalue LDA projection.
3. **MDA/2-1:** one learned image-mode projection; the other mode remains identity, corresponding to the paper's 2-D LDA special case.
4. **MDA/2-2:** alternate row and column projection updates using mode-wise within-/between-class scatter; current fixed output is 6×6.

## Classifier and metric

- Euclidean 1-nearest-neighbour in each learned feature space for every method.
- Primary metric: probe accuracy per split.
- Summary: mean and sample standard deviation over five splits per method/gallery size.
- Also preserve fit-and-predict runtime and MDA training-objective values by iteration.

## Sensitivity and safeguards

Rerun the same split seed rule with `--projection-normalization within_scatter`; compare with default `unit_l2`. Euclidean 1-NN is scale-sensitive, so do not choose normalization after viewing which gives a preferred ranking.

- Current run uses 64×64 faces, not original 112×92 images.
- Feature dimensions are fixed (PCA 36, MDA 6 per projected mode); the paper reports best results across choices, so our table is not like-for-like with its best-tuned values.
- Five splits are exploratory. A stronger follow-up should use at least 30 matched splits and paired uncertainty estimates.
- Hyperparameter selection must use training-only or validation data, never final probe scores.
- Keep source-paper Table I separate from our results in text, charts, and filenames.
