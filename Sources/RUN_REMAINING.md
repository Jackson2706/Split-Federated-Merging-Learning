# RUN_REMAINING — operating the E-HSFP journal completion campaign

All commands run from the repo root: `/home/jackson/Desktop/Split-Federated-Merging-Learning`.

## Environment

```bash
source results/journal_v1/queue/lane_env.sh
# Sets:
#   PATH         -> conda env AAAI-2026 (torch 2.7.0+cu128)
#   OUT_ROOT     =  /media/jackson/Data/ehsfp_campaign/outputs_journal_v1   (artifacts; data disk)
#   ROUNDS=60  MIN_FREE_GB=20  GPU_ID=0
```

Logs, markers and the manifest stay in the repo under `results/journal_v1/`. Checkpoints, predictions, history and
diagnostics go to `OUT_ROOT/<experiment_id>/`, which holds a verified copy of every earlier run too.

## What is running

Lanes are tmux windows in the session `ehsfp-lanes`. Each one drains a queue file in order:

| Lane | Queue file | Contents |
|---|---|---|
| laneA | `results/journal_v1/queue/laneA.queue` | Table II: ISIC-2018 E-HSFP seeds 1–4 |
| laneB | `results/journal_v1/queue/laneB.queue` | Table I: FedProx/FedNova on CIFAR-10/100 and HAM10000, then the shared queue |
| laneC, laneD | `results/journal_v1/queue/main.queue` | Tables VIII, III, IV, V, VI, IX, X, XII, XIII, in that priority order, seed-major |

```bash
scripts/journal_experiments/start_lanes.sh status    # which lanes are alive (pid)
tmux attach -t ehsfp-lanes                           # watch; detach with Ctrl-b d
tail -f results/journal_v1/queue/laneC.out           # lane driver output
ls results/journal_v1/.markers/*.lock                # runs in progress (lock holds pid + start time)
./journal_run.sh status                              # done / failed counts
```

Progress of one run: `grep -c "EPOCH" results/journal_v1/logs/<eid>.log`, or count the rows in
`$OUT_ROOT/<eid>/*/*/history.csv`.

## Start or resume

```bash
scripts/journal_experiments/start_lanes.sh           # idempotent: starts only lanes that are not alive
```

- **After a reboot** this runs automatically. The crontab `@reboot` entry runs it after 120 s, and only if
  the data disk is mounted. Its output goes to `results/journal_v1/queue/cron_reboot.log`.
- **Interrupted runs** leave a lock whose PID is dead. The next lane to reach that run takes the lock over
  and restarts the run from scratch: there is no mid-run optimizer/memory checkpoint, so an exact resume
  is impossible. The interrupted attempt's log is kept as `<eid>.attempt-<timestamp>.log`.
- **Finished runs** (`.done`) are never re-run. Failed runs (`.failed`) are not retried automatically.
  Inspect the log first, then:

```bash
./journal_run.sh retry_failed && scripts/journal_experiments/start_lanes.sh
```

## Supervisor, stop and restart

A cron entry runs `start_lanes.sh` every 10 minutes. It restarts any lane listed in `queue/LANES` that is
not alive. Its log is `results/journal_v1/queue/cron_supervisor.log`.

```bash
touch results/journal_v1/.markers/STOP_ALL     # every lane exits after its current run and STAYS down
rm results/journal_v1/.markers/STOP_ALL        # supervisor restarts lanes within 10 min
touch results/journal_v1/.markers/STOP_laneC   # one-shot: laneC exits after its run; the supervisor
                                               # restarts it, picking up script/env edits
```

To retire a lane permanently, delete its line from `results/journal_v1/queue/LANES`. To remove the
automation entirely, run `crontab -e` and delete the two lines tagged `# E-HSFP`.

## Add or re-prioritise work

Queue files are re-read after every run, so they can be edited while lanes are running. To print queue
lines for a stage (already-done runs are skipped):

```bash
EMIT_QUEUE=1 bash -c 'source ./journal_run.sh; c_dropout' | grep '|'
```

Completion stages: `c_fedvariants c_ablation c_dropout c_staleness c_serverless c_noniid c_memory`
`c_aggregation c_scalability c_comm200`. To add a lane:

```bash
echo laneE >> results/journal_v1/queue/LANES
cp results/journal_v1/queue/main.queue results/journal_v1/queue/laneE.queue
scripts/journal_experiments/start_lanes.sh
```

Before adding one, check `free -g` (allow ~3 GB RAM per CIFAR lane; HAM10000 E-HSFP needs more) and
`nvidia-smi` (12 GB total).

## Export (safe at any time; CPU only, about 1 minute)

```bash
OUT_ROOT=/media/jackson/Data/ehsfp_campaign/outputs_journal_v1 \
  python scripts/journal_experiments/collect_journal_v1.py
```

It writes:

- `manifests/run_status.csv` — one row per run
- `artifacts/per_run_metrics.csv`
- `artifacts/coverage_report.csv` — manuscript cell → runs → seed count
- `artifacts/tables/table_*.csv|.tex`
- `artifacts/figures/fig*.png|.pdf`

Then update `reports/simulation_completion_report.md` from those outputs.

## Tests

```bash
python -m pytest tests/ -q
```

## Blocked: needs a resource or a decision, not GPU time

See `reports/completion_audit.md` §6:

- ImageNet-1K data
- FedProto, FedGen and FedDF implementations
- Residual generator objective and L_drop placement
- Low-rank covariance rank
- Theorem 1 constants
- The conference-parity gap

---

## Moving to a new machine (state as of 2026-10-15)

Campaign status at handover:
- Protocol-v3 (`j3_*`): **55 of 144 runs done**.
- **Main journal Table I complete** (5 seeds).
- Sweep tables: single seed, partially done. Figures: seeds 0–2, partially done.

All automation on the old machine was stopped:
- lanes killed;
- the three `# E-HSFP` crontab lines removed;
- the `ehsfp-lanes` tmux session closed.

Four runs were interrupted mid-training and will restart from scratch:
- `j3_aggr_cifar100_learnable_reliability_memory_stale_s0`
- `j3_memM_cifar100_M20_s0`
- `j3_noniid_a0p7_cifar100_hsfp_s0`
- `j3_noniid_a0p7_cifar10_ehsfp_s0`

Their stale locks were cleared, so on the new machine they are simply picked up again.

1. **Clone** branch `DungTT`. Campaign state (`Sources/results/journal_v1/`: `.done` markers, queues, manifest, logs) and the manuscript are committed, so finished runs are skipped automatically.
2. **Raw run outputs** (needed for the figures and as evidence; checkpoints are excluded).
   - Copy `ehsfp_campaign_transfer.tar.gz` and its `.sha256` from the old data disk.
   - Verify with `sha256sum -c`, then extract it somewhere with space.
   - It contains `ehsfp_campaign/{outputs_journal_v1,dev,seg_hsfp_runs}`.
3. **Edit `Sources/results/journal_v1/queue/lane_env.sh`** for the new machine:
   - `PATH`: the conda env with torch.
   - `OUT_ROOT`: `<extract dir>/ehsfp_campaign/outputs_journal_v1`.
   - `PYTHONPATH`: the repo root, i.e. the parent of `Sources/`. The code imports `Sources.*`.
   - `GPU_ID`.
   - Datasets: `Sources/data/*` are symlinks to `/media/jackson/Data/HSF-P/...`. Re-point them at the new dataset location.
4. **Old run directories:** the collector maps the old absolute paths (`/media/jackson/Data/ehsfp_campaign/...`). If the new path differs, either symlink `/media/jackson/Data/ehsfp_campaign` to the new location, or set `OUT_ROOT` and re-run the collector.
5. **Lanes.** Set the number of lanes in `queue/LANES` to suit the new GPU; one lane per ~3 GB of free VRAM is a safe start. All lanes drain `queue/j3_cifar.queue`. Then start them:

   ```bash
   cd Sources && scripts/journal_experiments/start_lanes.sh
   ```

6. **Optional automation:** re-add the supervisor and hourly refresh to the crontab, using the new paths:

   ```
   */10 * * * * /bin/bash <repo>/Sources/scripts/journal_experiments/start_lanes.sh >> <repo>/Sources/results/journal_v1/queue/cron_supervisor.log 2>&1
   17 * * * * /bin/bash <repo>/Sources/scripts/journal_experiments/refresh_paper.sh >> <repo>/Sources/results/journal_v1/queue/refresh_paper.log 2>&1
   ```

7. **Tables, figures and PDF:**

   ```bash
   Sources/scripts/journal_experiments/refresh_paper.sh
   ```

   This writes `Submissions/E-HSFP Journal/main_autofilled.pdf`.

Decisions in force:
- Main Table I: 5 seeds. Sweep tables: seed 0. Figures: seeds 0–2.
- HAM10000 runs add `client_bn_eval=true` and `ssl_epochs_client=5`.
- CIFAR-100 communication is FedAvg-level (accepted trade-off).
