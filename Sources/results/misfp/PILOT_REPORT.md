# MiSFP pilot report (exploratory; generated from completed runs only)

CIFAR-100, ResNet split (client 128-d / edge 256-d), 200 clients, frac 0.1, 5 edges, t1=5/t2=10, validation = 10% held out from train (`val_from_train=0.1`), single seed. Headline = **final-round** test metrics through the composed model (no checkpoint selection). Single-seed differences are NOT significance-tested; metric std across seeds is unavailable.


## Split: dir0.1

| Run | final acc % | Δ vs A | Δ vs B | macro-F1 % | worst-10% class acc % | best-val ckpt acc % (round) | c→e payload KB | e→c payload KB | total comm MB | mean round s | peak CUDA GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H-SFP | 9.24 | 0.00 | -0.19 | 8.14 | 0.00 | 9.24 (10) | – | – | 226.9 | 249 | 1.58 |
| MiSFP-K1 | 9.43 | 0.19 | 0.00 | 7.81 | 0.00 | 9.43 (10) | 6138 | 7138 | 227.2 | 230 | 1.58 |
| MiSFP-K2 | 12.34 | 3.10 | 2.91 | 10.70 | 0.10 | 12.34 (10) | 7776 | 14818 | 236.3 | 217 | 1.58 |

| Run | mean client K | client K1 fallback frac | edge→cloud comps/class | between/within dispersion | held-out NLL/dim mixture | held-out NLL/dim single | local fit s (cum) | edge re-est s (cum) | syn samples edge/cloud | infeasible budgets |
|---|---|---|---|---|---|---|---|---|---|---|
| H-SFP | (1) | – | (1) | – | – | – | – | – | – | – |
| MiSFP-K1 | 1.000 | 0.000 | 1.00 | 0.442 | 0.255 | -0.945 | 4.6 | 7.2 | 176850/49950 | 0 |
| MiSFP-K2 | 1.271 | 0.729 | 1.98 | 0.442 | 0.322 | -0.945 | 14.6 | 9.5 | 176850/49950 | 0 |

## Not completed (excluded from all tables)

- `pilot_iid_edge_s0`: failed
- `pilot_iid_hsfp_s0`: failed
- `pilot_iid_k1_s0`: failed

PCA figure: `results/misfp/pilot_pca.png`.


## Cost projection
Measured mean round time across completed pilot runs: 232 s (shared GPU, concurrent campaign jobs). Full matrix: 135 runs × 60 rounds ≈ 522 GPU-hours sequential (≈ 21.7 days) at that throughput.

