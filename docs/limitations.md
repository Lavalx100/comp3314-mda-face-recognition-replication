# Limitations and interpretation rules

1. **Separate dataset copies:** the historical partial study uses scikit-learn's 64×64 Olivetti copy. The new follow-up uses the original 112×92 PGM images, matching the paper's stated resolution.
2. **Undocumented author settings:** the paper does not publish its split seeds, number of repeated trials, complete ORL dimension grid, or all implementation details. Exact Table I values cannot be expected from the information provided.
3. **Fixed-dimension mismatch:** the 30-split original-resolution primary run uses PCA 36, Fisherfaces up to 36, and fixed 6×6 MDA/2-2. Table I reports best-over-dimension results, including Fisherface/O. These are not like-for-like numbers.
4. **Dimension-selection bias:** the broad grid keeps every result and shows the full response surface. Its highest probe-selected cell is optimistic because the probe set also participates in dimension selection; do not report that selected value as an unbiased test estimate.
5. **Uncertainty scope:** the 30-split study describes variation over generated partitions of the same 400 images. Paired bootstrap ranges are conditional on this dataset and split generator, not population-level confidence intervals or significance tests.
6. **Scale sensitivity:** generalized eigenvectors have scaling conventions. Euclidean 1-NN is not invariant to feature-axis rescaling; our matched sensitivity run changes accuracy substantially.
7. **Implementation:** this is a group implementation, not official author code. Ridge regularization, handling of singular scatter, and generalized-eigenvector scaling are implementation choices and must be disclosed.
8. **Dataset scope:** ORL is small, controlled, grayscale, and largely frontal. It does not establish modern unconstrained face-recognition performance.
9. **Training objective:** the MDA Fisher-ratio trace can rise while held-out nearest-neighbour accuracy remains poor. It is an optimization diagnostic, not evidence of generalization.

## Safe language

Say: “On the original 112×92 ORL images, our fixed 6×6 MDA/2-2 implementation fell well below the paper's best-over-dimension scores with two and three gallery images. MDA/2-1 was stronger in those fixed settings, and the two-mode result improved as the gallery grew. We are showing a broad dimension sweep because the article does not disclose its full parameter search. These findings describe our implementation and protocol.”

Do not say: “We disproved the paper,” “MDA is worse,” or “we reproduced the paper's exact results.”
