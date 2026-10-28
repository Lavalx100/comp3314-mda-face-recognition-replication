# Smoke check — 2026-10-28

**Purpose:** verify the packaged runner and output writer on cached data. This is a software check only, not a result for the research question.

## Command

```bash
python src/run_experiment.py \
  --data-home /path/to/existing/olivetti-cache \
  --train-per-class 2,5 --repeats 1 --max-iter 2 \
  --output-dir results/smoke-check
```

The run completed for all four methods at both gallery sizes and wrote raw results, summary files, chart, and MDA trace. One split per size is not suitable for comparing methods or estimating uncertainty; the output is retained only as a smoke-test artifact.

## Validation

`python -m unittest discover -s tests -v` passed all seven unit tests. `python -m py_compile src/run_experiment.py tests/test_experiment.py` completed successfully.

## Software note

The test command used the existing COMP3314 virtual environment and the already-cached Olivetti archive; no dataset archive or face image was copied into this project.
