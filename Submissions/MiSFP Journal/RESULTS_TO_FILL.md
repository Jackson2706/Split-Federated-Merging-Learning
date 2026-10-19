# Results to fill

All measured scalars enter `results/measurements.csv`; component and feature artifacts use their dedicated files. Do not paste numbers directly into LaTeX. A run must have a stable `source_run_id`, resolved-config hash, status, unit, and one of seeds 1, 2, 3.

## Tables

- `tab:main`: final-checkpoint accuracy and macro-F1 for every method × primary partition × seed. Adapted baselines require identical data partition, architecture capacity, rounds, and evaluation split.
- `tab:resources`: cumulative serialized client–edge and edge–cloud prototype bytes; inherited model-sync bytes; total bytes; wall-clock seconds; peak allocated memory; hardware/software record.
- `tab:ablation`: H-SFP and all six registered MiSFP variants with accuracy, macro-F1, and post-compression components/class.
- `tab:sensitivity`: matched-K1 budgets (1.0/1.5/2.0), fp16/fp32, caps, minimum support, and boundary sample sweeps.
- `tab:fidelity`: held-out per-class NLL, SWD, MMD, mode coverage, minority recall, and between-source dispersion for local/shared-snapshot extraction.

## Figures

- `fig:empirical(a)`: checkpoint accuracy/F1 plus cumulative total, C–E, and E–C bytes.
- `fig:empirical(b)`: checkpoint metrics with round and wall-clock time.
- `fig:empirical(c)`: feature artifacts with one projection fit on real training features, then applied unchanged to held-out real and synthetic features.
- `fig:empirical(d)`: per-class component records, payloads, and paired quality metric.
- `fig:diagnostics`: component cap, minimum support, boundary sample count, and extraction-mode sweeps.

## Result-dependent prose

After validation, add one paragraph in Section VII reporting the principal estimates and completion counts, one paragraph interpreting fidelity/alignment outcomes, and one limitations update for failed/incomplete settings. Do not add winner language without the registered seed set and a stated uncertainty comparison.

Run `make assets validate paper` from this directory. Missing data must continue to render as `Results pending` or `\MissingResult`; never encode missing or failed records as zero.
