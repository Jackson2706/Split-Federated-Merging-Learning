# Prompt: apply round-2 fixes (paste into the MacBook assistant)

Continue the E-HSFP revision. Two parts: finish the code setup, then apply the paper fixes. Stop after each part and report.

## Part 1 — Code
1. I copied `run_queue.py`, `make_manifest.py`, and `runs.csv` into `Sources for coding`. Run the full test suite again; the previously skipped queue test must now pass. Commit these files on `revision/ehsfp-diagnostics`.
2. Update `CMD_TEMPLATE` and `KEY_MAP` in `run_queue.py` to the real CLI flags and config files (`configs/revision/cifar100_reference.yaml`, `configs/revision/ham10000_reference.yaml`). Every override key in `runs.csv` must map to a real flag; show me `python run_queue.py --dry-run | head -20`.
3. Check both reference YAML files against the paper and report any mismatch: 200 rounds, 200 clients, **10 edge servers**, 10% participation, batch 16, Adam lr = wd = 1e-4, 5-round warm-up, T_temp = 0.5, α = 0.3, p_train = 0.2, A_M = 20, λ_prc = 0.1, ResNet-50 backbone. Do not change them without telling me.
4. Then give me the Phase C commands for the A6000.

## Part 2 — Paper (LaTeX in `Submissions for submission`)
Back up the folder first. Apply every block in `E-HSFP_fixes_round2.tex` (I put it in the paper folder) exactly as written:
- F1 backbone sentence, F2 M_M definition, F3 scope paragraph → §VII-A.
- F4 Table IV(c) third-row label and F5 Table VI → use the values I give you; if a value is still `[[FILL]]`, leave it and list it.
- F6 insert `E-HSFP_appendix_proofs.tex` as the first subsection of Appendix B; replace the three "supplementary material [[TODO …]]" passages with `Appendix~\ref{app:mech_proofs}`; make the equation references inside the proofs match the actual `\eqref` labels.
- F7 → §VII-F. Before inserting, recompute the six Welch tests with `scipy.stats.ttest_ind_from_stats(m1, s1, 5, m2, s2, 5, equal_var=False)` and confirm the p-values.
- F8 → §II-B; F10 BibTeX → the .bib file (keep `[[VERIFY]]` fields marked). Use the paper's citation commands and reference style.
- F9 → §VIII after the Deployment paragraph.

Rules: keep labels/refs/environments intact; do not change theorems, propositions, lemmas, assumptions, or equations; compile the PDF before and after. Report changed files, a change log by section, and every remaining `[[TODO]]`, `[[FILL]]`, and `[[VERIFY]]` with its location.
