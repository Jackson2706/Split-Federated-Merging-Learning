# Claude Code prompt: finish the existing E-HSFP simulation campaign

Open Claude Code in your experiment repository and paste the prompt below, or ask it to read this file and execute it. Make the manuscript available in that repository. The earlier full simulation prompt is useful as a coverage checklist if available.

---

The E-HSFP implementation already exists in this repository. Some experiment results are still pending, delayed, failed, or not appearing in the output tables. Your task is to inspect the actual state, resolve the causes, and finish all outstanding experiments and result exports required by the manuscript.

Treat this as completion and recovery of an existing research campaign. Reuse the existing implementation, configurations, launchers, datasets, checkpoints, and valid results. Do not start a new framework or rewrite functioning algorithms. Make targeted fixes, run them, and continue until every unblocked required result is complete.

## 1. Inspect before launching anything

Read CLAUDE.md, AGENTS.md, README files, the manuscript, frozen configurations, experiment manifests, run scripts, result exporters, and relevant logs. The manuscript may retain the filename `_ECCV_26__H_SFP__Ultra_Lightweight_Federated_Learning_via_Hierarchical_Split_Prototyping.pdf` although its title is E-HSFP.

Inspect the current machine and any already configured, authorized execution environment: GPUs/VRAM, CPU/RAM, disk, Python environment, datasets, scheduler queues, and existing experiment processes. Do not assume local files represent the remote runner's current state. Do not print credentials or inspect unrelated users' work.

Identify active jobs before starting replacements. Associate each relevant PID, scheduler ID, or session with its configuration, seed, output directory, progress, and last checkpoint. Never duplicate an already running job. Do not kill healthy jobs or unrelated processes to free resources.

Build a persistent run ledger, preferably extending the existing manifest. Each row should contain:

`experiment_id, table_or_figure, dataset, method, variant, partition, seed, protocol, budget, config_hash, code_revision, status, progress, checkpoint, log_path, runner_id, failure_reason, next_action`.

Use statuses such as completed-valid, running-healthy, queued, pending, failed-resumable, failed-restart-required, stalled-confirmed, blocked-data, blocked-protocol, and export-only. File existence alone is insufficient evidence of completed training.

## 2. Determine the exact missing work

Map Tables I–XIII and Figures 2–11 to required run IDs and seeds. Reconcile the manuscript against the current frozen campaign configuration and the earlier full simulation prompt if present. Distinguish mandatory manuscript experiments from optional added diagnostics; complete mandatory coverage first. Do not expand a one-axis sweep into an unnecessary full Cartesian product.

Check these families:

- Main classification: CIFAR-10, CIFAR-100, HAM10000, ImageNet-1K; required FL, split/hierarchical, prototype/generative, H-SFP, and E-HSFP baselines.
- ISIC-2018 segmentation and its required baselines.
- IID, mild non-IID, and severe non-IID comparisons.
- Packet loss p=[0,0.1,0.2,0.3,0.5].
- Staleness tau=[0,1,3,5,10].
- Timeout, cold start, partial edge execution, missing edge update, and combined event stress.
- H-SFP component ablations and E-HSFP module ablations.
- Memory capacity M=[0,1,5,10,20].
- Aggregation and synthesis variants.
- Client populations K=[20,50,100,200,500].
- Aggregation intervals, communication, per-tier memory, convergence, drift, and stability diagnostics.

The supplied draft freezes new classification results at 60 rounds and seeds [0,1,2,3,4]. Non-IID alpha values are 0.7 and 0.3. Recover the native segmentation budget; the conference segmentation reference uses 200 rounds. Do not silently alter an explicitly revised frozen repository protocol: document the discrepancy and isolate incompatible comparisons.

Keep imported conference values, conference reproductions, 10-round proxies, legacy date-seed diagnostics, and five-seed journal results separate. An existing result is reusable only if its effective configuration, partition, seed, training budget, metric definitions, and compatible implementation provenance match. A changed commit alone does not invalidate a result; establish whether the changed code affected it.

Report the initial inventory concisely: valid completed runs, healthy running jobs, missing seeds, failures, export-only gaps, and real blockers. Then act on it without waiting for routine confirmation.

## 3. Diagnose why results are delayed or absent

For each gap, determine whether the cause is slow but healthy training, a scheduler dependency, GPU contention, CUDA OOM, excessive data-loader workers, blocked I/O, missing files, download/extraction failure, a corrupt sample, deadlock, distributed synchronization, numerical failure, checkpoint/config mismatch, permissions, full disk, or an exporter/glob/schema error.

Check process state together with logs, checkpoint timestamps, GPU/CPU activity, and recent iteration durations. Observe progress across multiple checks over a workload-appropriate interval. Do not declare a long ImageNet epoch stalled merely because its log is quiet. Set timeouts using measured progress expectations rather than arbitrary short limits.

If valid raw metrics already exist but a paper table is blank, repair aggregation/exporting and regenerate the artifact. Do not rerun training to fix a filename or parser problem. If a diagnostic was never collected, determine whether a saved checkpoint supports the exact computation. Endpoint checkpoints cannot reconstruct missing historical trajectories; run the required instrumented experiment when necessary.

Apply the smallest justified fix. Preserve original logs, failed outputs, and provenance. Never delete evidence or overwrite a completed run with a retry. Use separate attempt directories with one canonical successful result per configuration/seed.

For confirmed hung project-owned jobs, record evidence and checkpoint availability before a targeted graceful stop/restart. Never use blanket kill commands. Bound retries; repeated identical failure should trigger diagnosis rather than endless restarts.

## 4. Resume safely and preserve comparability

Resume from the latest valid compatible checkpoint, restoring model, optimizer, scheduler, RNG states, local/edge/cloud counters, prototype memories, packet queue, timestamps, and required data-sampler state. Validate resume semantics with a short check before spending substantial compute.

If only model weights were saved, label a continuation as a warm start, not an exact resume. Where this breaks the frozen protocol, restart only the affected required runs from the proper seed and initialization.

Reuse completed valid seeds and rerun only missing or scientifically invalid seeds. If a bug fix changes algorithm behavior, identify every affected configuration and rerun that affected comparison set. Do not patch one method and silently compare it against results from a different protocol.

Do not reduce rounds, seeds, client count, image resolution, dataset size, or model size to finish faster. Do not change mixed precision, optimizer settings, training losses, or batch semantics without recording a protocol change. First address memory problems through safe concurrency limits and resource cleanup. Gradient accumulation is not automatically equivalent when batch normalization or contrastive negatives depend on batch size.

Never fabricate, interpolate, copy, or extrapolate missing benchmark results. Retain negative and inconclusive outcomes. Tune nothing on test results and do not force E-HSFP to outperform a baseline.

## 5. Execute all remaining unblocked jobs

Use or repair the existing launcher so it supports listing, dry runs, execution, status, retries, resumption, and aggregation. Add a thin completion wrapper only if necessary.

Construct a deduplicated queue of missing valid configuration/seed pairs. Schedule within measured GPU memory, RAM, CPU, and disk capacity. Existing healthy jobs take precedence. Do not launch every job simultaneously or assume more processes mean faster completion.

Prioritize export-only fixes, near-complete resumes, missing seeds that complete paired comparisons, then the remaining required datasets and sweeps. Let long jobs progress alongside lightweight exports only when resources permit. Use a durable scheduler/session mechanism that actually exists in this environment, with run locks and atomic checkpoints.

Estimate remaining GPU-hours and disk from measured representative jobs. Continue using already authorized available compute. Do not purchase or provision external compute. If the full queue cannot run concurrently, keep it serialized or queued; do not silently omit expensive experiments.

Verify launched jobs by checking that their logs or checkpoints advance. Monitor them while this session is active. If jobs outlast the session, report their exact actual states, durable runner IDs, and commands to inspect/resume them. Do not describe running or submitted jobs as completed, or promise monitoring that has not been established.

If a dataset is unavailable or requires access credentials, record the precise missing resource and continue other jobs. If an algorithm variant is not implemented faithfully, isolate the missing component and implement a targeted repair supported by the manuscript/reference code. Do not silently substitute a different baseline. Escalate only substantive unresolved protocol choices that cannot be recovered; continue independent work.

## 6. Finish aggregation and paper artifacts

For each production cell, verify the required seeds reached the prescribed budget, metrics are finite, checkpoints/provenance are consistent, and evaluation splits match. Aggregate mean and sample standard deviation with ddof=1. Do not treat individual rounds as independent runs. Incomplete cells must show their actual seed count and status.

Regenerate CSV/LaTeX tables and PDF/PNG figures from recorded metrics. Check labels, units, error bars, legends, missing rows, duplicate seeds, and accidental mixing of protocols. Preserve the historical conference tables separately from new matched comparisons.

Keep environmental packet loss separate from training dropout; actual delayed statistics separate from merely changed age labels; simulated time separate from host wall time; and diagnostic oracle statistics separate from training inputs. Do not claim a theoretical bound was numerically verified when its constants were never specified.

Create or update:

- `reports/completion_audit.md`: initial inventory, causes, fixes, and remaining blockers.
- `manifests/run_status.csv`: current per-run state and provenance.
- `artifacts/coverage_report.csv`: each required table cell/curve linked to runs and seed counts.
- `reports/simulation_completion_report.md`: measured results and final coverage.
- `RUN_REMAINING.md`: exact environment, launch, status, resume, and export commands.

Follow existing project paths where sensible. Do not rewrite scientific claims in the paper just to hide missing experiments.

## 7. Definition of done

The campaign is complete only when every mandatory unblocked run is completed and validated, required seed aggregates and figures are generated, and each manuscript result is traceable to source runs. Distinguish any remaining genuinely blocked cells from completed coverage. A job left running is still outstanding.

Your final report must say:

1. What was already complete and reused.
2. Why the remaining results were delayed or missing.
3. What you fixed and how it was checked.
4. Which experiments actually finished during this session.
5. Which jobs remain running/queued/blocked, if any, with precise reasons and commands.
6. Where the tables, figures, and coverage report are saved.

Start now with a read-only inventory of the existing implementation, active jobs, logs, checkpoints, and missing results. Then repair and execute the remaining work. Do not stop after giving me instructions to run it myself.
