# COMP3314 Group 27 — MDA face-recognition replication

This is the working research repository for Group 27's study of Yan et al., “Multilinear Discriminant Analysis for Face Recognition” (IEEE TIP, 2007). It records the research question, implementation, protocol, run artifacts, limitations, progress, and report/presentation scaffolds.

**Current status:** a runnable partial reproduction is complete on scikit-learn's 64×64 Olivetti copy. It is not an exact reproduction of the paper's original-resolution experiment or tuned Table I results. See [progress](docs/PROGRESS.md) before describing the work as complete.

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

Run the 30-split primary study and matched-seed scaling sensitivity:

```bash
python3 src/run_experiment.py --repeats 30 --output-dir results/rerun/unit_l2
python3 src/run_experiment.py --projection-normalization within_scatter \
  --repeats 30 --output-dir results/rerun/within_scatter
python3 scripts/analyze_paired_results.py results/rerun/unit_l2/raw_results.csv
```

The dataset is fetched and cached locally under `.cache/olivetti/`; it is not committed. Each run writes split-level CSV, summary CSV/JSON, an SVG chart, and the MDA objective trace. Keep outputs from each configuration in a separate folder.

Quick smoke run (not for reported results):

```bash
python3 src/run_experiment.py --train-per-class 2,5 --repeats 1 --max-iter 2 \
  --output-dir results/smoke
```

## Repository map

```text
src/run_experiment.py      PCA, Fisherfaces, MDA/2-1, MDA/2-2, and run outputs
scripts/analyze_paired_results.py  Paired split differences and bootstrap ranges
tests/                     Deterministic unit tests for core functions
results/                   Dated baseline, follow-up, and smoke artifacts kept separate
docs/research-plan.md      Questions, hypotheses, stages, and assessment mapping
docs/protocol.md           Current data/split/model/evaluation protocol
docs/paper-notes.md        Paper summary and original ORL comparison values
docs/experiments/          Dated experiment records and handoff notes
docs/PROGRESS.md           Status, completed work, and next steps
docs/CHANGELOG.md          Human-readable project change history
report/main.tex            Editable three-page report draft
output/pdf/                Compiled, review-ready report PDF
slides/                    Nine-minute presentation outline and Q&A prep
site/index.html            Local research overview, plan, results, and team roles
```

## Results at a glance

Primary protocol: mean accuracy ± sample SD across 30 matched splits, percent:

| Train images/person | Eigenfaces PCA | Fisherfaces LDA | MDA/2-1 | MDA/2-2 |
|---:|---:|---:|---:|---:|
| 2 | 65.7 ± 2.3 | 70.6 ± 2.6 | 72.5 ± 3.5 | 42.4 ± 4.6 |
| 3 | 75.3 ± 2.6 | 84.7 ± 2.9 | 83.2 ± 2.4 | 66.7 ± 3.7 |
| 4 | 81.0 ± 3.6 | 90.0 ± 2.2 | 87.4 ± 2.5 | 82.2 ± 3.3 |
| 5 | 85.0 ± 2.4 | 93.3 ± 1.3 | 90.2 ± 2.3 | 88.1 ± 2.6 |

![Thirty-split accuracy chart](results/2026-10-28-30-repeat-followup/unit_l2/accuracy_vs_training_size.svg)

MDA/2-1 leads at two training images; Fisherfaces leads at three through five. MDA/2-2 is sensitive to projection-axis normalization: for two training images it scores 42.4% with unit-L2 axes and 20.1% with within-scatter axes. Do not merge these with the paper's Table I. This sensitivity describes the implementation and metric convention, not a general refutation of MDA. The original five-split exploratory run remains preserved separately at `results/2026-10-24-partial-replication/`; see the [30-split follow-up record](docs/experiments/2026-10-28-30-repeat-followup.md).

## Reproducibility and data handling

- Split seeds are deterministic and identical across methods for each train count/repeat.
- Learned projections use training data only; test examples are used only for evaluation.
- Dataset cache and raw face archive are excluded from version control. Cite AT&T Laboratories Cambridge and scikit-learn's loader documentation.
- Record each new experiment under `docs/experiments/` with command, configuration, seed policy, output paths, interpretation, and limitations before changing the headline result.
- The course brief asks for a public GitHub repository for code submitted for reproduction bonus consideration. Keep this working repository private until the group reviews its contents and explicitly decides whether to publish it.

## References

- Yan, Xu, Yang, Zhang, Tang, and Zhang (2007), [paper PDF](https://mmlab.ie.cuhk.edu.hk/archive/2007/IP07_face02.pdf), DOI [10.1109/TIP.2006.884929](https://doi.org/10.1109/TIP.2006.884929).
- scikit-learn, [Olivetti faces dataset documentation](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset).
