# Paper-proximate ORL reproduction and optimization audit

**Run date:** 2026-11-09 (Hong Kong time)

**Status:** both 10,130-pair full grids are complete; optimized implementation benchmarked and numerically checked on representative real ORL fits. The exhaustive grid was not rerun after optimization.

**Paper:** Yan et al. (2007), [Multilinear Discriminant Analysis for Face Recognition](https://mmlab.ie.cuhk.edu.hk/archive/2007/IP07_face02.pdf)

## Scope and retained results

The paper-proximate follow-up uses the original Cambridge ORL archive: 400 grayscale images, 40 identities, 10 images per identity, 112×92 pixels. The archive SHA-256 is `509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec`. No raw face images are committed.

The full MDA/2-2 grid uses all integer output dimensions allowed by Theorem 2:

`row_dim <= 39 * col_dim` and `col_dim <= 39 * row_dim`.

There are 10,130 feasible pairs. Five gallery/probe splits were evaluated at each gallery size 2–5, for 202,600 MDA/2-2 fits per normalization. The original-resolution within-scatter and unit-L2 outputs remain separate under [`results/2026-11-09-paper-proximate-full-grid/`](../../results/2026-11-09-paper-proximate-full-grid/) and [`results/2026-11-09-paper-proximate-full-grid-unit-l2/`](../../results/2026-11-09-paper-proximate-full-grid-unit-l2/). The optimized code has not overwritten or regenerated either result set.

## Paper method compared with our implementation

| Part | What the paper specifies | What this project implements or assumes |
|---|---|---|
| Images | ORL face-recognition images; histogram equalization; gallery images and remaining probes | Original 112×92 Cambridge PGM files. Equalization is per image and label-free. The paper does not report software or output levels; this run uses 64 levels as a MATLAB-default-aligned approximation, not a confirmed author setting. |
| MDA updates | Identity initialization; sequential k-mode updates; Fig. 3 stops after iteration 2 when projection changes meet a dimension-scaled epsilon test | Row then column updates, matching the paper's mode order. Fig. 3's threshold is implemented as `||U_k^t-U_k^(t-1)||_F < m'_k*m_k*epsilon`. The paper omits epsilon and maximum iterations; this run uses `epsilon=1e-4`, maximum 10. |
| Scatter matrices | Class-weighted between-class scatter and pooled within-class scatter over matching coordinates of the unfolded tensors | The same sums are evaluated by `mode_scatter`; cached class membership changes the accumulation method, not the equations. A small independent loop reference checks both modes. |
| Eigenproblem | Leading generalized eigenvectors, with output-dimension limits in Theorem 2 | Symmetric whitening solves the regularized problem. A relative ridge of `1e-6` times mean within-scatter trace is our choice because the scatter can be singular. The paper does not specify singular-scatter handling or axis scaling. |
| Projection scaling | Not reported in enough detail to determine the exact numerical convention | Both within-scatter-metric and unit-L2 axis normalization are retained. Euclidean 1-NN is sensitive to this choice. |
| Classification | Nearest-neighbour classification in projected space | Euclidean 1-nearest-neighbour; flattening the projected matrix preserves the Frobenius distance. |
| Split and dimension choice | Random gallery selection and reported best-over-dimension results; exact seeds and selection protocol are not given | Independent deterministic seeds by gallery size, five splits, and all per-split probes retained. The paper's exact split and tuning procedure cannot be reconstructed. |

The paper's recoverable mathematics and algorithm order have been audited against its equations, Theorem 2 and Fig. 3. The exact numbers in Table I cannot be recovered from the paper because preprocessing details, epsilon, eigenvector scaling, ridge or pseudoinverse handling, random splits, repeat count, and feature-dimension selection are not fully reported.

### Equation and data-flow check

For each mode, the implementation treats every matching unfolded coordinate as a separate set of feature vectors. For coordinate `q`, let `μ_cq` be the mean vector for class `c`, `μ_q` the overall mean, and `n_c` the class size. The accumulated scatters are

`S_b = Σ_q Σ_c n_c (μ_cq − μ_q)(μ_cq − μ_q)^T`

`S_w = Σ_q Σ_c Σ_{i∈c} (z_iq − μ_cq)(z_iq − μ_cq)^T`.

These are the class-weighted between-class and pooled within-class sums used in the paper's mode-wise generalized eigenproblem. The loop-reference unit test computes those same formulas directly and compares them with the vectorized scatter code for both image modes.

The two-mode sequence is `U⁰=I`, `V⁰=I`; update `U` from the vectors in `X_i V`, then update `V` from the vectors in `Uᵀ X_i`, using the newly updated `U`, and repeat. This is the sequential mode order in Fig. 3. The code solves for leading eigenvectors of `S_b u = λ S_w u` after adding its documented ridge. Projection features are `Uᵀ X_i V`. Flattening these features does not alter their Frobenius geometry, so Euclidean 1-NN compares `||Uᵀ(X_probe−X_gallery)V||_F`, as required by the matrix representation. Probe images never enter the scatter or eigensolver calls.

## Results and the gallery-size-3 dip

The full-grid values below are the **highest mean probe accuracy for one fixed dimension pair across five splits**. The dimensions were chosen using those same probe labels, so all four maxima are optimistic and are not unbiased test estimates.

| Gallery images/person | Paper MDA/2-2 | Within-scatter best mean ± SD | Selected dimensions | Unit-L2 best mean ± SD | Selected dimensions |
|---:|---:|---:|---:|---:|---:|
| 2 | 90.1% | 62.44 ± 4.14% | 18×1 | 61.81 ± 1.22% | 112×86 |
| 3 | 95.0% | 57.29 ± 2.91% | 28×8 | 82.29 ± 3.82% | 64×46 |
| 4 | 98.3% | 77.92 ± 2.38% | 25×6 | 90.75 ± 1.85% | 40×26 |
| 5 | 99.0% | 88.60 ± 2.38% | 14×5 | 94.90 ± 1.56% | 30×22 |

Gallery size 3 is below size 2 only in the within-scatter grid. On the same seeds, unit-L2 rises from 61.81% to 82.29%. A leave-one-repeat-out dimension-selection diagnostic also rises for unit-L2 (60.38% to 80.86%), while within-scatter remains lower (59.81% to 54.00%). This diagnostic selects dimensions on four repeats and evaluates on the fifth, but images overlap across different random splits, so it is not an independent test estimate.

We also ran a fixed 6×6 diagnostic on 30 nested splits: for every identity, the two-image gallery is a subset of the three-image gallery. Dimension is fixed before reading probe accuracy. Adding the third training image raises mean accuracy by 18.47 percentage points for unit-L2 (34.57% to 53.05%) and 21.23 points for within-scatter (5.47% to 26.70%). Thus the grid dip does not show that adding a gallery image generally hurts recognition. It appears only in the probe-selected within-scatter grid. The evidence points to an interaction between scale-sensitive Euclidean distances, dimension selection, and the different random partitions; the current runs do not isolate one single cause.

Within-scatter eigenvectors have unit norm in the within-scatter metric, not necessarily equal Euclidean length. Nearest-neighbour distance therefore gives different weights to the projected axes than with unit-L2 normalization. This is a plausible contributor, supported by the large same-seed results change between the two conventions, but the paper does not specify which axis scaling its authors used.

## Leakage audit

- Histogram equalization is per image and uses neither identity labels nor population statistics. It does not leak information across train and probe images.
- MDA projections and training embeddings use gallery images only. Probe images are transformed after fitting.
- **Dimension selection does use probe accuracy.** The `best_by_gallery.csv` row selects one of 10,130 pairs by mean accuracy over the same five probe splits later reported. Those selected maxima are exploratory paper-style diagnostics and are optimistic.
- The leave-one-repeat-out diagnostic reduces direct reuse of a split's score, but its random splits reuse the same 400 source images. Treat it as sensitivity evidence only.
- A less biased estimate would require selecting dimensions on training-only validation data and evaluating once on an untouched outer test set. That is a different protocol from the reported full-grid table and was not substituted into these reproduction results.

## Performance profile and optimization

### Execution environment and completed run

The original full sweep ran locally on the MacBook Pro Mac15,10 with Apple M3 Max, 14 cores (10 performance, 4 efficiency), and 36 GB unified memory. It was not running remotely. The exhaustive unit-L2 run completed in 2 h 19 m 40 s: 202,600 MDA/2-2 fits, approximately 24.17 fits/s end-to-end, with about 1.3 GB peak process footprint. The historical run's `fit_and_predict_seconds` field timed the MDA fit only; it did not separately time classification. Its saved results are preserved.

NumPy 2.5.4 and SciPy 1.18.1 use Apple's Accelerate on this machine. `threadpoolctl` reports no controllable Accelerate pool, so its `limits=1` context alone is not proof that BLAS was single-threaded. Future sweep launches now set `VECLIB_MAXIMUM_THREADS=1` before importing NumPy; the benchmark sets that variable in isolated child processes and also tests values 2 and 4. Apple documents single-thread mode for Accelerate separately from the threadpoolctl API.

### Stage timings

One direct, gallery-size-5 split was profiled over 12 representative dimension pairs (all reached 10 MDA iterations):

| Stage | Time across 12 fits | Share of MDA fit time |
|---|---:|---:|
| Scatter construction | 0.657 s | 44.0% |
| Eigen-decomposition | 0.269 s | 18.0% |
| Other projection and iteration work | 0.568 s | 38.0% |
| 1-NN classification | 0.011 s | 0.75% of fit plus classification |

Data loading took 0.032 s and 64-level equalization 0.026 s for the entire dataset. The expensive part is repeated MDA fitting, especially scatter construction and iteration, not classification or preprocessing.

### Validated implementation changes

1. Cache class labels, counts and membership once per train/probe split. The class-wise scatter equations are unchanged.
2. Reuse the first row update across all dimension pairs and the first column update across all column dimensions for a given row dimension. At identity initialization, those updates do not depend on the requested output dimensions beyond taking the leading basis columns. This removes thousands of repeated first-iteration scatter/eigen calculations.
3. Keep worker execution as threads sharing the same read-only image arrays; no per-fit process copy is introduced.
4. Measure fit and 1-NN separately and record only the final MDA objective when the paper's projection-change rule is used. Objective logging does not affect that stopping rule.

On the actual 112×92 ORL images, direct and cached fits were compared on 12 representative pairs for both normalizations. After aligning arbitrary eigenvector signs, the largest relative Frobenius projection differences were 7.0e-9 (row) and 1.21e-8 (column) for unit-L2, and 2.45e-10 and 2.08e-10 for within-scatter. Relative differences in all train/probe squared distances were at most 5.53e-11 and 4.59e-12 respectively. Across these checks, predictions and iteration counts matched exactly; objective differences were at most 6.90e-11 and 2.31e-12. These are representative numerical-equivalence checks, not a second exhaustive validation.

### Worker and BLAS benchmark

The repeatability benchmark evaluates 12 small-to-large dimension pairs, five times each, on a fixed gallery-size-5 split. Fits/s excludes the cache-build step; building the cache took 0.10–0.11 s per split, negligible across the full grid.

| Implementation | Workers | BLAS threads | Fits/s | Peak RSS |
|---|---:|---:|---:|---:|
| Direct | 8 | 1 | 32.79 | 0.88 GiB |
| Direct | 14 | 1 | 35.23 | 1.14 GiB |
| Direct | 16 | 1 | 35.25 | 1.26 GiB |
| Cached | 8 | 1 | 37.79 | 0.69 GiB |
| Cached | 10 | 1 | 39.22 | 0.74 GiB |
| Cached | 14 | 1 | 39.73 | 0.91 GiB |
| Cached | 16 | 1 | 40.79 | 0.89 GiB |
| Cached | 16 | 2 | 39.25 | 0.87 GiB |
| Cached | 16 | 4 | 39.03 | 0.93 GiB |

For this fixed representative workload, 16 workers was fastest; 14 was only about 2.7% slower. One BLAS thread per fit was best. Extrapolating 40.79 fits/s to 202,600 fits gives about 1 h 22 m 47 s, plus roughly two seconds of cache setup over the 20 splits. At 14 workers the estimate is about 1 h 25 m. These are projections from a 60-fit representative batch, not a rerun of the full experiment. They imply about 1.69× the original overall run throughput; actual total time may differ across gallery sizes.

### Other optimization options evaluated

- Class-sum aggregation alternatives (`einsum`, batched matrix multiplication) matched the current class-membership GEMM to 1e-12 but took about 2.53–2.54 ms versus 2.34 ms. The current approach stays.
- SciPy's generalized subset eigensolver was tested on real ORL fits. It was 4–7% slower across the representative batch, and one within-scatter fit changed its stopping count. It is not adopted.
- The current environment has no PyTorch, MLX, CuPy or CUDA. No GPU result is claimed. The repeatedly solved matrices are only 112×112 and 92×92; moving an iterative fit to Metal or CUDA would require a solver port and fresh numerical validation. The CPU path is currently practical.

## HKU compute options

An AMD CPU node with 32 workers could shorten another complete sweep, but no remote pilot was run and no account availability was assumed. HKU Research Computing currently limits undergraduate access and normally requires a PI-led project, PI endorsement, and a project presentation of at least five slides. See [account requirements](https://hpc.hku.hk/documentation/access/), [HPC2021 partitions and QoS](https://hpc.hku.hk/guide/slurm-guide/slurm-partitions-qos/), and [SLURM job-script guide](https://hpc.hku.hk/guide/slurm-guide/slurm-job-script/).

A ready-to-submit CPU job is in [`hpc/orl_mda_grid.sbatch`](../../hpc/orl_mda_grid.sbatch), with access and setup steps in [`hpc/README.md`](../../hpc/README.md). It has not been submitted. The CS GPU Farm teaching cluster is intended for eligible course projects, but a GPU rewrite is not justified by the small matrix sizes or the current CPU timings; the course-specific eligibility must be checked separately.

## Reproducible checks and outputs

- Unit/regression suite: `uv run --with-requirements requirements.txt --no-project python -m unittest discover -s tests -v` — **13/13 passed**.
- `python -m py_compile` passed for the experiment and benchmark scripts; `git diff --check` and `bash -n hpc/orl_mda_grid.sbatch` passed.
- Direct-vs-cache numerical results: [`numerical_equivalence_detailed.json`](../../results/2026-11-09-performance-benchmark/numerical_equivalence_detailed.json).
- Stage and worker benchmarks: [`2026-11-09-performance-benchmark/`](../../results/2026-11-09-performance-benchmark/) and [`2026-11-09-performance-repeatability/`](../../results/2026-11-09-performance-repeatability/).
- Fixed-size nested-gallery diagnostic: [`2026-11-09-nested-gallery-diagnostic/`](../../results/2026-11-09-nested-gallery-diagnostic/).
- Optimized sweep implementation: [`src/run_experiment.py`](../../src/run_experiment.py) and [`scripts/run_orl_dimension_sweep.py`](../../scripts/run_orl_dimension_sweep.py).

No further exhaustive sweep was started. The existing results, including the interrupted pre-optimization checkpoint, remain preserved. Before claiming an unbiased generalization score, run a predeclared dimension-selection procedure using training-only validation and a separate outer test set.
