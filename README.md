# COMP3314 Group 27 — MDA face-recognition replication

This is the working research repository for COMP3314 Group 27's study of Yan et al., “Multilinear Discriminant Analysis for Face Recognition” (IEEE TIP, 2007). It records the research question, implementation, protocols, run artifacts, limitations, progress, and handoff materials.

**Current status (2026-11-10):** the corrected fixed-setting runs and both full 10,130-pair original-resolution grids are complete and remain the canonical experimental datasets. The optimized implementation is now the default for dimension sweeps. It reuses eigensystem factorizations while preserving requested-dimension projection operations. All 17 configurations flagged by the interrupted run were rechecked under both axis normalizations; corrected cached and direct fits matched on predictions, accuracy, iteration counts, projections, distances, and objectives. The redundant exhaustive rerun was stopped after three completed diagnostic splits, and its checkpoints are preserved separately. See the [correctness and performance audit](docs/experiments/2026-11-09-paper-proximate-reproduction.md), [optimization validation record](docs/experiments/2026-11-10-optimized-default-and-equivalence.md), [progress](docs/PROGRESS.md), and [optional HKU HPC setup](hpc/README.md).

**GitHub:** [private working repository](https://github.com/Lavalx100/comp3314-mda-face-recognition-replication) · Review attribution and course policy as a group before making it public.

## Research question

When image structure, training/test examples, and the classifier are controlled, does matrix-based MDA improve face identification over vector-based Eigenfaces PCA and Fisherfaces LDA, particularly with few training images per identity?

## Quick start

Requires Python 3.10 or newer. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

To view the project overview locally, run `python3 -m http.server 8000` from this directory and open `http://127.0.0.1:8000/site/`.

Download the official Cambridge `att_faces.zip` archive into the ignored local cache (see [data instructions](docs/data-and-ethics.md)). The archive and face images are not committed:

```bash
mkdir -p .cache/orl-original
curl -L https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip \
  -o .cache/orl-original/att_faces.zip
```

Run the 30-split primary study and matched-seed scaling sensitivity:

```bash
python3 src/run_experiment.py --repeats 30 --output-dir results/rerun/unit_l2
python3 src/run_experiment.py --projection-normalization within_scatter \
  --repeats 30 --output-dir results/rerun/within_scatter
python3 scripts/analyze_paired_results.py results/rerun/unit_l2/raw_results.csv
```

The scikit-learn dataset is fetched and cached locally under `.cache/olivetti/`; it is not committed. Each run writes split-level CSV, summary CSV/JSON, an SVG chart, and the MDA objective trace. Keep outputs from each configuration in a separate folder.

Run the original-resolution paper-style follow-up with the paper's Fig. 3 stopping condition:

```bash
python3 src/run_experiment.py --dataset orl-original --repeats 30 \
  --histogram-levels 64 --projection-normalization within_scatter \
  --paper-epsilon 1e-4 --max-iter 10 --fisher-pca-max 10400 \
  --mda-stopping-rule paper_figure \
  --output-dir results/rerun/orl-original/within_scatter
python3 src/run_experiment.py --dataset orl-original --histogram-levels 64 \
  --projection-normalization unit_l2 --paper-epsilon 1e-4 --repeats 30 \
  --max-iter 10 --fisher-pca-max 10400 --mda-stopping-rule paper_figure \
  --output-dir results/rerun/orl-original/unit_l2
python3 scripts/analyze_paired_results.py results/rerun/orl-original/unit_l2/raw_results.csv
```

The full feasible integer-dimension grid uses the optimized implementation by default. This example writes to a separate dated directory, checkpoints after every split, and uses the M3 Max benchmark's best worker/BLAS setting:

```bash
python3 scripts/run_orl_dimension_sweep.py --full-mda-grid \
  --histogram-levels 64 --projection-normalization unit_l2 \
  --paper-epsilon 1e-4 --mda-stopping paper_figure --mda-order row_first \
  --max-iter 10 --workers 14 --repeats 5 --seed 3314 \
  --output-dir results/runs/example-optimized-full-dimension-grid
```

The optimized path reuses the first row and column eigensystem factorizations across dimensions, then forms each requested-size projection with the same shape-specific multiply as a direct fit. This avoids repeated scatter/eigensolver work without the rounding differences caused by truncating a full projection basis. On the M3 Max, 14 workers and one Accelerate thread measured 47.48 fits/s on a representative batch. If interrupted, rerun with the same arguments and `--resume`; changing the dataset hash or sweep settings is rejected. **The existing full grids do not need to be rerun just to use this optimization.** Their original direct-run CSVs remain canonical and all figures/tables in this repository use those complete outputs.

Quick smoke run (not for reported results):

```bash
python3 src/run_experiment.py --train-per-class 2,5 --repeats 1 --max-iter 2 \
  --output-dir results/smoke
```

## Repository map

```text
src/run_experiment.py      PCA, Fisherfaces, MDA/2-1, MDA/2-2, and fixed-setting runs
src/orl_data.py            Reader for the original Cambridge PGM archive
scripts/analyze_paired_results.py  Paired split differences and bootstrap ranges
scripts/run_orl_dimension_sweep.py  Original-resolution sweep; cached eigensystem reuse is default
scripts/benchmark_mda_efficiency.py  Representative stage and worker-count benchmark
scripts/validate_mda_cache_equivalence.py  Compares interrupted rows and rechecks flagged unit-L2 fits
scripts/validate_mda_targeted_equivalence.py  Rechecks selected settings under either normalization
scripts/run_nested_gallery_diagnostic.py  Fixed-6×6 nested 2-vs-3 gallery diagnostic
scripts/render_mda_dimension_heatmap.py  SVG visualization of the MDA/2-2 grid
tests/                     Deterministic unit tests for core functions
results/                   Dated baseline, follow-up, and smoke artifacts kept separate
docs/research-plan.md      Questions, hypotheses, stages, and assessment mapping
docs/protocol.md           Current data/split/model/evaluation protocol
docs/paper-notes.md        Paper summary and original ORL comparison values
docs/experiments/          Dated experiment records and handoff notes
docs/PROGRESS.md           Status, completed work, and next steps
docs/CHANGELOG.md          Human-readable project change history
report/main.tex            Current original-resolution study report source
slides/                    Nine-minute presentation outline and Q&A prep
site/index.html            Local research overview, plan, results, and team roles
hpc/                       Optional HKU HPC2021 CPU SLURM setup (not submitted)
```

The report source now reflects the original-resolution protocol, fixed-setting results, full-grid comparison, leakage limitation, and validated optimization. Both full-grid heatmaps are regenerated from their canonical summaries and correctly label their normalization. The rendered report is in `output/pdf/`.

## Corrected paper-proximate fixed-setting results

Mean accuracy ± sample SD over 30 matched splits, percent. This corrected fixed-dimension run uses the 64-level preprocessing variant and Fig. 3's dimension-scaled stopping rule. The paper's table gives best-over-dimension scores, so these fixed 6×6 numbers are a method check, not the final Table I comparison.

| Gallery images/person | PCA | Fisherfaces, unit-L2 | MDA/2-1, unit-L2 | MDA/2-2, unit-L2 | MDA/2-2, within-scatter | Paper MDA/2-2 best |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 73.9 ± 2.5 | 73.1 ± 2.8 | 81.4 ± 3.0 | 34.8 ± 5.3 | 5.4 ± 1.9 | 90.1 |
| 3 | 84.1 ± 1.8 | 84.9 ± 2.9 | 89.9 ± 2.2 | 52.6 ± 5.6 | 23.9 ± 10.3 | 95.0 |
| 4 | 88.2 ± 3.0 | 88.0 ± 2.4 | 92.0 ± 2.2 | 75.8 ± 3.6 | 64.0 ± 6.0 | 98.3 |
| 5 | 91.6 ± 2.0 | 90.4 ± 2.2 | 94.7 ± 1.7 | 87.5 ± 2.8 | 81.4 ± 3.8 | 99.0 |

The large scaling gap is still unresolved, so both conventions remain visible. The [corrected experiment record](docs/experiments/2026-11-09-paper-proximate-reproduction.md) lists assumptions, commands, and artifacts.

## Corrected full feasible dimension search

The full 112×92 grid has 10,130 dimension pairs after applying Theorem 2's rank bounds. Each normalization was evaluated on five splits for every gallery size. Best-by-grid dimensions use probe scores from the same splits, so treat the values as optimistic diagnostics.

| Gallery images/person | Within-scatter best mean | Unit-L2 best mean | Unit-L2 selected dimensions |
|---:|---:|---:|---:|
| 2 | 62.44% | 61.81% | 112×86 |
| 3 | 57.29% | 82.29% | 64×46 |
| 4 | 77.92% | 90.75% | 40×26 |
| 5 | 88.60% | 94.90% | 30×22 |

The gallery-size-3 dip appears in the within-scatter grid only. A fixed-dimension 30-repeat nested split shows accuracy increasing from two to three gallery images for both normalizations. The audit discusses why the selected-grid dip is an interaction of axis scaling, dimension selection and split variation, not a general penalty for adding an image.

Both heatmaps can be regenerated from the saved summaries without fitting any models. Pass the normalization explicitly so the SVG title, accessible description, and subtitle match the source grid:

```bash
python3 scripts/render_mda_dimension_heatmap.py \
  results/2026-11-09-paper-proximate-full-grid-unit-l2/dimension_summary.csv \
  results/2026-11-09-paper-proximate-full-grid-unit-l2/mda22_dimension_heatmap.svg \
  --normalization unit_l2
python3 scripts/render_mda_dimension_heatmap.py \
  results/2026-11-09-paper-proximate-full-grid/dimension_summary.csv \
  results/2026-11-09-paper-proximate-full-grid/mda22_dimension_heatmap.svg \
  --normalization within_scatter
```

## Historical October original-resolution results (superseded)

The following earlier fixed-setting and coarse dimension-sweep results were produced before the stopping-rule audit. The October implementation compared projection change only to `m'_k`; Fig. 3 includes `m'_k × m_k × epsilon`. Keep these results as history and don't describe them as the corrected paper-style run.

Historical mean accuracy ± sample SD across 30 matched splits, percent:

Mean accuracy ± sample SD across 30 matched splits, percent. These settings are fixed (PCA 36, Fisherfaces up to 36, MDA/2-1 with 6 row features, and MDA/2-2 with 6×6 features); they do not reproduce the paper's best-over-dimension selection.

| Train images/person | Eigenfaces PCA | Fisherfaces LDA | MDA/2-1 | MDA/2-2 (6×6) |
|---:|---:|---:|---:|---:|
| 2 | 73.8 ± 2.5 | 73.1 ± 2.9 | 81.4 ± 2.9 | 42.7 ± 5.5 |
| 3 | 84.0 ± 1.8 | 84.9 ± 2.8 | 89.9 ± 2.2 | 72.0 ± 3.9 |
| 4 | 88.2 ± 3.1 | 88.2 ± 2.4 | 92.1 ± 2.1 | 85.4 ± 2.2 |
| 5 | 91.6 ± 2.0 | 90.4 ± 2.2 | 94.6 ± 1.6 | 91.4 ± 2.3 |

![Thirty-split original-resolution accuracy chart](results/2026-10-28-original-resolution-30-repeat/paper_figure_unit_l2/accuracy_vs_training_size.svg)

The main mismatch is fixed 6×6 MDA/2-2 at two and three gallery images. The paper reports a best-over-dimension MDA/2-2 score of 90.1% with two gallery images; our fixed 6×6 mean is 42.7%. The completed dimension sweep tests how much feature size changes the result, but its best grid values are probe-selected and optimistic.

## Historical coarse dimension sweep

The five-repeat sweep retained all 156 feasible MDA/2-2 row/column pairs, along with PCA, Fisherfaces, and MDA/2-1 feature curves. The best cell for each method/gallery size was selected using the same probe splits, so these values are diagnostic and optimistic; they are not unbiased test estimates.

| Gallery images/person | PCA best | Fisherfaces best | MDA/2-1 best | MDA/2-2 best | Paper MDA/2-2 |
|---:|---:|---:|---:|---:|---:|
| 2 | 76.7% (79) | 74.4% (26) | 83.1% (9 rows) | 62.9% (112×92) | 90.1% |
| 3 | 85.9% (118) | 86.1% (38) | 90.5% (10 rows) | 83.6% (32×16) | 95.0% |
| 4 | 89.3% (159) | 88.7% (39) | 92.6% (10 rows) | 91.8% (32×24) | 98.3% |
| 5 | 92.5% (66) | 90.0% (38) | 94.5% (15 rows) | 94.6% (48×24) | 99.0% |

Parentheses show selected feature dimensions. At two gallery images, the MDA/2-2 grid's best point uses the entire 112×92 image (10,304 features) and still reaches only 62.9%. Dimension selection explains part, but not all, of the gap from Table I. This remains a paper-style follow-up, not an exact reconstruction or a verdict on the paper.

![MDA/2-2 recognition accuracy across the declared row and column dimension grid](results/2026-10-28-original-resolution-dimension-sweep/paper_figure_unit_l2/mda22_dimension_heatmap.svg)

Download the [full sweep summary](results/2026-10-28-original-resolution-dimension-sweep/paper_figure_unit_l2/dimension_summary.csv), [probe-selected maxima and dimensions](results/2026-10-28-original-resolution-dimension-sweep/paper_figure_unit_l2/best_by_gallery.csv), and [published Table I transcription](results/2026-10-28-original-resolution-dimension-sweep/paper_figure_unit_l2/published_table_i.csv). The [dated experiment record](docs/experiments/2026-10-28-original-resolution-reproduction.md) explains the methods, outputs, sensitivity results, and limitations.

## Earlier 64×64 partial study

Historical protocol: mean accuracy ± sample SD across 30 matched splits, percent:

| Train images/person | Eigenfaces PCA | Fisherfaces LDA | MDA/2-1 | MDA/2-2 |
|---:|---:|---:|---:|---:|
| 2 | 65.7 ± 2.3 | 70.6 ± 2.6 | 72.5 ± 3.5 | 42.4 ± 4.6 |
| 3 | 75.3 ± 2.6 | 84.7 ± 2.9 | 83.2 ± 2.4 | 66.7 ± 3.7 |
| 4 | 81.0 ± 3.6 | 90.0 ± 2.2 | 87.4 ± 2.5 | 82.2 ± 3.3 |
| 5 | 85.0 ± 2.4 | 93.3 ± 1.3 | 90.2 ± 2.3 | 88.1 ± 2.6 |

![Thirty-split 64×64 accuracy chart](results/2026-10-28-30-repeat-followup/unit_l2/accuracy_vs_training_size.svg)

The original five-split exploratory run remains preserved separately at `results/2026-10-24-partial-replication/`; see the [30-split 64×64 follow-up record](docs/experiments/2026-10-28-30-repeat-followup.md). Do not merge 64×64 and original-resolution results.

## Reproducibility and data handling

- Split seeds are deterministic and identical across methods for each train count/repeat.
- Learned projections use training data only; test examples are used only for evaluation.
- Dataset cache and raw face archive are excluded from version control. Cite AT&T Laboratories Cambridge and scikit-learn's loader documentation.
- Record each new experiment under `docs/experiments/` with command, configuration, seed policy, output paths, interpretation, and limitations before changing the headline result.
- The course brief asks for a public GitHub repository for code submitted for reproduction bonus consideration. Keep this working repository private until the group reviews its contents and explicitly decides whether to publish it.

## References

- Yan, Xu, Yang, Zhang, Tang, and Zhang (2007), [paper PDF](https://mmlab.ie.cuhk.edu.hk/archive/2007/IP07_face02.pdf), DOI [10.1109/TIP.2006.884929](https://doi.org/10.1109/TIP.2006.884929).
- scikit-learn, [Olivetti faces dataset documentation](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset).
