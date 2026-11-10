# Limitations and interpretation rules

1. **Separate dataset copies:** the historical partial study uses scikit-learn's 64×64 Olivetti copy. The new follow-up uses the original 112×92 PGM images, matching the paper's stated resolution.
2. **Undocumented author settings:** the paper does not publish its split seeds, number of repeated trials, complete ORL dimension grid, the value of epsilon in Fig. 3, histogram-equalization levels/software, axis normalization, or singular-scatter handling. Exact Table I values cannot be expected from the information provided.
3. **Fixed-dimension mismatch:** the 30-split original-resolution primary run uses PCA 36, Fisherfaces up to 36, and fixed 6×6 MDA/2-2. Table I reports best-over-dimension results, including Fisherface/O. These are not like-for-like numbers.
4. **Dimension-selection leakage:** the full 10,130-pair grid keeps every result and shows the response surface, but `best_by_gallery.csv` selects dimensions using mean accuracy over the same five probe splits later reported. Its selected values are optimistic and are not unbiased test estimates. A leave-one-repeat-out diagnostic still reuses the same 400 images across splits.
5. **Uncertainty scope:** the 30-split study describes variation over generated partitions of the same 400 images. Paired bootstrap ranges are conditional on this dataset and split generator, not population-level confidence intervals or significance tests.
6. **Scale sensitivity:** generalized eigenvectors have scaling conventions. Euclidean 1-NN is not invariant to feature-axis rescaling; the full grid changes substantially between within-scatter and unit-L2. The paper does not fully specify the convention. A relative ridge is added to make singular within-class scatter solvable.
7. **Earlier stopping-rule correction:** the October `paper_figure_*` runs omitted `m_k × epsilon` from the Fig. 3 threshold and stopped too early. They are retained as historical implementation results; use the November corrected run for paper-proximate comparison.
8. **Implementation:** this is a group implementation, not official author code. Ridge regularization and generalized-eigenvector scaling are implementation choices and must be disclosed. The optimized cache now reuses eigensystem factorizations while forming requested-size projections with the direct operation shape; its targeted equivalence checks do not replace a full independent implementation review.
9. **Dataset scope:** ORL is small, controlled, grayscale, and largely frontal. It does not establish modern unconstrained face-recognition performance.
10. **Training objective:** the MDA Fisher-ratio trace can rise while held-out nearest-neighbour accuracy remains poor. It is an optimization diagnostic, not evidence of generalization.

## Safe language

Say: “On the original 112×92 ORL images, our fixed 6×6 MDA/2-2 results fell below the paper's best-over-dimension scores. The full grid's gallery-size-3 dip appears only with within-scatter scaling. With fixed 6×6 dimensions and nested gallery splits, adding a third gallery image improves mean accuracy under both tested scalings. The selected full-grid values use probe scores for dimension selection, so they are optimistic. These findings describe our implementation and protocol.”

Do not say: “We disproved the paper,” “MDA is worse,” or “we reproduced the paper's exact results.”
