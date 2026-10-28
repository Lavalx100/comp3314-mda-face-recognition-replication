# Working agreement

This is a four-person course project. Keep the repository readable enough that another group member can rerun a result and explain its assumptions.

## For every experiment change

1. Start from a named research question or a specific reproducibility gap.
2. Write the expected comparison and evaluation protocol before looking at new test results.
3. Preserve the primary run and use a new output directory for changed settings.
4. Add or update tests for code changes; run `python -m unittest discover -s tests -v`.
5. Add a dated record under `docs/experiments/` and update `docs/PROGRESS.md` and `docs/CHANGELOG.md`.
6. Commit code, configuration, and derived tabular/plot artifacts together. Never commit `.venv/`, dataset caches, credentials, or unreviewed external data.
7. Separate paper results from group results. Do not silently replace a result because another setting scores better.

Use small commits with messages that state the research change, for example `test: cover deterministic stratified splits` or `experiment: add 30-repeat paired run`.

## Review checklist

- [ ] Same train/test partition across compared methods.
- [ ] No test data used to fit projections or select hyperparameters.
- [ ] Seeds, software versions, preprocessing, and dimensions recorded.
- [ ] Raw split rows preserved; summary statistics recomputable.
- [ ] Limitations and deviations from the source paper stated.
- [ ] Dataset and paper cited; no face images redistributed without checking terms.
