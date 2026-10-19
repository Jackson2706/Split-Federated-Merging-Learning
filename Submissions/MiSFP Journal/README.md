# MiSFP journal draft

The manuscript is `main.tex`. Measured values have one source of truth: `results/measurements.csv`, supplemented by the component and feature artifact indices. Current inputs are intentionally empty.

```bash
cd "/home/jackson/Desktop/Split-Federated-Merging-Learning/Submissions/MiSFP Journal"
make assets validate paper
```

`scripts/plot_assets.py` deterministically regenerates explanatory PDFs and polished missing-result templates. `scripts/generate_results.py` validates keys, units, finite values, seed membership, and aggregate configuration compatibility; it leaves a one-seed SD null rather than zero. See `RESULTS_TO_FILL.md` and `drafting_notes/source_map.md` before importing results.
