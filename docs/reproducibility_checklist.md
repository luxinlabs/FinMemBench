# Reproducibility checklist

- [ ] Record download URLs, UTC retrieval time, archive SHA-256, SEC access policy version.
- [ ] Pin Python dependencies, models, prompt templates, sampling seeds, and token budgets.
- [ ] Freeze train/validation/test manifests and company/time split logic.
- [ ] Preserve every accession reference and evidence availability timestamp.
- [ ] Run data audits for duplicate facts, unit mismatches, amended forms, and time zones.
- [ ] Report full confusion matrix and calibration/risk–coverage plots.
- [ ] Keep a human-reviewed set of ambiguous restatements and non-reliance events.
- [ ] Release runnable scripts and synthetic fixtures even if full SEC corpus is too large to redistribute.

`python -m finmemguard.demo` is a smoke test, not scientific validation.
