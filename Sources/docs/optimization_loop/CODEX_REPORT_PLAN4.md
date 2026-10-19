## EXECUTION REPORT

Implemented the CPU-only cached-feature analysis in [normalize_probe.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/diag/normalize_probe.py) and updated [RESULTS_SUMMARY.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/RESULTS_SUMMARY.md).

Accuracy values below are percentages.

| Boundary | Transform | NC | Mean | Diag | Cosine | Linear | Centroid cos | W/B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | raw | 10.72 | 6.53 | 6.47 | 11.65 | 29.81 | 0.9848 | 4.8961 |
| L1 | centered | 10.72 | 9.81 | 9.03 | 21.41 | 29.81 | -0.0044 | 4.8961 |
| L1 | L2 | 10.62 | 6.10 | 5.46 | 11.65 | 30.04 | 0.9850 | 4.7918 |
| L1 | centered+L2 | 10.77 | 10.25 | 9.48 | 21.41 | 29.80 | 0.0223 | 5.1684 |
| L1 | centered+PCA-white | 25.48 | **16.57** | **17.30** | **22.90** | 28.95 | -0.0047 | 13.2463 |
| L1 | centered+white+L2 | **25.58** | 16.42 | 14.41 | **22.90** | 29.00 | 0.0031 | 13.5422 |
| L2 | raw | 9.39 | 9.29 | 8.72 | 14.86 | 26.46 | 0.9339 | 4.5324 |
| L2 | centered | 9.39 | 10.38 | 8.84 | 18.70 | 26.46 | -0.0043 | 4.5324 |
| L2 | L2 | 9.69 | 9.58 | 9.21 | 14.86 | 26.68 | 0.9356 | 4.5435 |
| L2 | centered+L2 | 9.60 | 10.08 | 8.87 | 18.70 | 26.24 | 0.0216 | 4.7958 |
| L2 | centered+PCA-white | 26.11 | 20.29 | **17.21** | **22.73** | 30.97 | -0.0051 | 21.6066 |
| L2 | centered+white+L2 | **26.56** | **20.76** | 14.80 | **22.73** | **31.31** | 0.0030 | 22.0060 |

Winner: centered+whitened+L2 for nearest-centroid.

- L1: 25.58%, +14.86 points / 2.39× versus raw; 4.48 points below the 30.06% ceiling.
- L2: 26.56%, +17.17 points / 2.83× versus raw; 0.06 points below its prior 26.62% linear ceiling, or 3.50 points below 30.06%.
- Centered+L2 alone did not improve nearest-centroid. The declared ≥20% criterion passes specifically because of whitening.
- Prototype heads improve substantially but remain below the linear ceiling.

Verification passed: 12 production cells, finite metrics, train-only transform audits, eval-fit rejection, immutable head configuration, and exact fixed-seed determinism on a reduced real-cache fixture.

`full_e_hsfp` is not ready: no final `metrics.json`; only a partial epoch-1 `clients_complete` counter exists.

Unrounded results: [normalize_results.json](/home/jackson/Desktop/Split-Federated-Merging-Learning/outputs/diag/plan4_normalize_seed20260714/normalize_results.json).