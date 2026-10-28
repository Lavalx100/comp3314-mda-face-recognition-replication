# Paper notes: Multilinear Discriminant Analysis for Face Recognition

**Citation:** Shuicheng Yan, Dong Xu, Qiang Yang, Lei Zhang, Xiaoou Tang, and Hong-Jiang Zhang, “Multilinear Discriminant Analysis for Face Recognition,” *IEEE Transactions on Image Processing*, 16(1), 212–220 (2007), DOI 10.1109/TIP.2006.884929. [Author-hosted paper](https://mmlab.ie.cuhk.edu.hk/archive/2007/IP07_face02.pdf).

## Problem and motivation

Vector-based PCA/LDA turn an image into one long pixel vector. This discards its explicit row/column organization and produces a high-dimensional feature space relative to the number of labeled examples. MDA represents an image as a matrix (or more generally a tensor) and learns discriminative projections along its modes.

## Main idea in plain language

MDA seeks to make different identities far apart while keeping images from the same identity close, measured in the projected tensor space. Because joint optimization over several projection matrices is difficult, the method alternates: hold all but one mode fixed, unfold/project data along that mode, compute class scatter, update that mode's projection, and move to the next. For a face matrix, MDA/2-2 learns both row and column projections. MDA/2-1 uses one subspace and is treated by the paper as the 2-D LDA special case.

## Evaluation described by the article

The paper compares Eigenface, Fisherface, MDA/2-1, and MDA/2-2. Gallery and probe images are transformed using learned spaces and a nearest-neighbour classifier makes the final prediction. On ORL, the article describes 400 images from 40 people, histogram equalization, and random gallery/probe partitions with different gallery counts. Table I is described as reporting best recognition accuracies across these partitions.

| Gallery images/person | Eigenface, paper Table I | MDA/2-2, paper Table I |
|---:|---:|---:|
| 2 | 81.6% | 90.1% |
| 3 | 87.7% | 95.0% |
| 4 | 91.3% | 98.3% |
| 5 | 97.5% | 99.0% |

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

Present claims as the authors' claims; our run is a partial reproduction with explicit differences.
