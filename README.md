# COMP3314 Group 27 — MDA face-recognition replication

This is the working research repository for Group 27's study of Yan et al., “Multilinear Discriminant Analysis for Face Recognition” (IEEE TIP, 2007). It records the research question, implementation, protocol, run artifacts, limitations, progress, and report/presentation scaffolds.

**Current status:** a runnable partial reproduction is complete on scikit-learn's 64×64 Olivetti copy. It is not an exact reproduction of the paper's original-resolution experiment or tuned Table I results. See [progress](docs/PROGRESS.md) before describing the work as complete.

## Research question

When image structure, training/test examples, and the classifier are controlled, does matrix-based MDA improve face identification over vector-based Eigenfaces PCA and Fisherfaces LDA, particularly with few training images per identity?

## Quick start

Requires Python 3.10 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the primary study and matched-seed scaling sensitivity:

```bash
python src/run_experiment.py --output-dir results/rerun/unit_l2
python src/run_experiment.py --projection-normalization within_scatter \
  --output-dir results/rerun/within_scatter
```

The dataset is fetched and cached locally under `.cache/olivetti/`; it is not committed. Each run writes split-level CSV, summary CSV/JSON, an SVG chart, and the MDA objective trace. Keep outputs from each configuration in a separate folder.

Quick smoke run (not for reported results):

```bash
python src/run_experiment.py --train-per-class 2,5 --repeats 1 --max-iter 2 \
  --output-dir results/smoke
```

## Repository map

```text
src/run_experiment.py      PCA, Fisherfaces, MDA/2-1, MDA/2-2, and run outputs
tests/                     Deterministic unit tests for core functions
results/                   Frozen artifacts from the initial partial reproduction
docs/research-plan.md      Questions, hypotheses, stages, and assessment mapping
docs/protocol.md           Current data/split/model/evaluation protocol
docs/paper-notes.md        Paper summary and original ORL comparison values
docs/experiments/          Dated experiment records and handoff notes
docs/PROGRESS.md           Status, completed work, and next steps
docs/CHANGELOG.md          Human-readable project change history
report/main.tex            Editable report draft
slides/                    Nine-minute presentation outline and Q&A prep
```

## Results at a glance

Mean accuracy ± sample SD across five matched splits, percent:

| Train images/person | Eigenfaces PCA | Fisherfaces LDA | MDA/2-1 | MDA/2-2 |
|---:|---:|---:|---:|---:|
| 2 | 67.1 ± 1.9 | 71.8 ± 2.8 | 73.5 ± 2.9 | 43.3 ± 4.8 |
| 3 | 74.5 ± 3.0 | 85.1 ± 3.2 | 83.7 ± 1.0 | 67.4 ± 3.0 |
| 4 | 81.3 ± 0.8 | 90.8 ± 2.2 | 87.1 ± 2.1 | 83.5 ± 1.4 |
| 5 | 84.0 ± 3.6 | 93.2 ± 1.2 | 89.8 ± 2.3 | 88.3 ± 2.8 |

Do not merge these with the paper's Table I. Our MDA/2-1 run leads at two training images; Fisherfaces leads at three through five. Results also change materially under an alternate projection-axis normalization. These findings describe this fixed implementation and protocol, not a general refutation of MDA.

## Reproducibility and data handling

- Split seeds are deterministic and identical across methods for each train count/repeat.
- Learned projections use training data only; test examples are used only for evaluation.
- Dataset cache and raw face archive are excluded from version control. Cite AT&T Laboratories Cambridge and scikit-learn's loader documentation.
- Record each new experiment under `docs/experiments/` with command, configuration, seed policy, output paths, interpretation, and limitations before changing the headline result.
- The course brief asks for a public GitHub repository for code submitted for reproduction bonus consideration. Keep this working repository private until the group reviews its contents and explicitly decides whether to publish it.

## References

- Yan, Xu, Yang, Zhang, Tang, and Zhang (2007), [paper PDF](https://mmlab.ie.cuhk.edu.hk/archive/2007/IP07_face02.pdf), DOI [10.1109/TIP.2006.884929](https://doi.org/10.1109/TIP.2006.884929).
- scikit-learn, [Olivetti faces dataset documentation](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset).
