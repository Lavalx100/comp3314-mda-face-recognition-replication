# Paper notes: Multilinear Discriminant Analysis for Face Recognition

**Citation:** Shuicheng Yan, Dong Xu, Qiang Yang, Lei Zhang, Xiaoou Tang, and Hong-Jiang Zhang, “Multilinear Discriminant Analysis for Face Recognition,” *IEEE Transactions on Image Processing*, 16(1), 212–220 (2007), DOI 10.1109/TIP.2006.884929. [Author-hosted paper](https://mmlab.ie.cuhk.edu.hk/archive/2007/IP07_face02.pdf).

## Problem and motivation

Vector-based PCA/LDA turn an image into one long pixel vector. This discards its explicit row/column organization and produces a high-dimensional feature space relative to the number of labeled examples. MDA represents an image as a matrix (or more generally a tensor) and learns discriminative projections along its modes.

## Main idea in plain language

MDA seeks to make different identities far apart while keeping images from the same identity close, measured in the projected tensor space. Because joint optimization over several projection matrices is difficult, the method alternates: hold all but one mode fixed, unfold/project data along that mode, compute class scatter, update that mode's projection, and move to the next. For a face matrix, MDA/2-2 learns both row and column projections. MDA/2-1 uses one subspace and is treated by the paper as the 2-D LDA special case.

The paper's Fig. 3 starts each projection at an identity matrix, updates modes in order, and stops after iteration 2 once the change in each projection matrix is below that mode's requested output dimension. Our code exposes this printed stopping rule as `paper_figure`; the paper does not specify the maximum iteration cap used in our runs.

## Evaluation described by the article

The paper compares Eigenface, Fisherface, MDA/2-1, and MDA/2-2. Gallery and probe images are transformed using learned spaces and a nearest-neighbour classifier makes the final prediction. On ORL, the article describes 400 images from 40 people, histogram equalization, and random gallery/probe partitions with different gallery counts. Table I is described as reporting best recognition accuracies across these partitions.

Table I values as printed in the paper (gallery/probe order below is 5/5, 4/6, 3/7, 2/8):

| Method | 5/5 | 4/6 | 3/7 | 2/8 |
|---|---:|---:|---:|---:|
| Eigenface | 97.5% | 91.3% | 87.7% | 81.6% |
| Fisherface | 92.0% | 85.4% | 87.5% | 74.7% |
| Fisherface/O | 92.0% | 85.8% | 87.5% | 79.7% |
| MDA/2-1 | 98.5% | 96.3% | 94.3% | 88.1% |
| MDA/2-2 | 99.0% | 98.3% | 95.0% | 90.1% |

“Fisherface/O” is the best result across dimensions in the LDA step. The article says Table I shows the best accuracies across gallery/probe partitions; it does not provide the random seeds or enough detail to recreate the exact split realization and all parameter choices. Our original-resolution dimension sweep is therefore a paper-style comparison, not an exact reconstruction. Its highest probe-selected scores are optimistic; retain all dimensions and do not treat those maxima as unbiased performance estimates.

These are paper values, not our repeated-split means. Scikit-learn documents its copy as 64×64 and the original ORL images as 92×112 pixels; credit AT&T Laboratories Cambridge when using the dataset.

## Course connections

- PCA: unsupervised variance-maximizing subspace learning.
- LDA: supervised between-class versus within-class scatter.
- Generalized eigenproblems: solve a discriminative direction objective.
- k-nearest neighbours: classification after projection, with distances affected by axis scaling.
- Small-sample issues: tensor-mode projections change geometry when labeled examples are scarce.

## Critical questions for the report

- Has the alternating MDA procedure converged, and does its training objective correspond to test accuracy?
- How do output dimensions and projection scaling affect a nearest-neighbour comparison?
- How should “best” dimensions be selected without using final test partitions?
- How much of performance is due to tensor structure versus preprocessing, feature selection, or dataset protocol?
- Does a 64×64 copy support numerical comparison to original-resolution results?

Present the paper's claims as the authors' claims. The original-resolution follow-up matches the stated ORL image size and preprocessing, but it is not an exact reconstruction of the authors' random runs or undocumented best-dimension search.
