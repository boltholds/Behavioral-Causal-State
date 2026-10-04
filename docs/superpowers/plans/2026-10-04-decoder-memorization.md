# Decoder memorization implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, with one final independent review. Direct main publication is already authorized.

**Goal:** Determine whether unchanged readers/decoder can memorize one and eight actual SONAR histories before scaling L1.
**Architecture:** Separate experiment runner imports the current Grounder and verified caches. Full-batch training on a fixed train-only subset records teacher-forced and greedy metrics. Same-length input swaps and zero-input controls are evaluated after training.
**Tech Stack:** Existing Python/PyTorch runtime; cached pinned SONAR vectors; no new dependency.
**Spec:** docs/specs/03-language.md; docs/implementation/language-runtime.md; experiments/protocols/decoder-memorization-v0.1.json. This is a capacity/debugging diagnostic, not the registered L1 experiment.

## Global constraints

- Keep src/bcs architecture, masks, float32 embeddings and normalization unchanged unless a reproduced bug requires a fix.
- Reuse verified input lock; train labels only. No held-out test or new encoding.
- Branches concat/attention/gru; seed11; AdamW lr0.0003, weight_decay0.01, dropout0.1, clip1.0; CPU4 threads.
- Stage single: first train history of length4. Stage eight: first two distinct target histories of each length4/8/12/16, in this order. Reset model/optimizer for each branch/stage.
- Maximum1000 optimizer steps, full batch; evaluate step0 and every25 steps. Stop only after3 consecutive evaluations with exact greedy history recovery for all selected records and zero invalid decodings, or budget exhaustion.
- Report training-set memorization only. No result establishes semantic generalization, identifies encoder information loss, or satisfies L1 gates.

## Review focus

1. True prefixes must enter only teacher-forced loss/diagnostics, never greedy inputs or masks.
2. Input controls must preserve recipient lengths/masks; same-length swaps must be derangements with distinct target histories.
3. EOS errors and invalid histories must prevent success even when event fields match.
4. Dataset/protocol/source/weights changes must fail verification, not silently relock.
5. A single lucky checkpoint or train-set success must not become stable memorization or L1 acceptance.

## Task1 — Diagnostic measurements and fixed selection

**Files:** experiments/diagnostics/decoder_memorization.py; tests/test_decoder_memorization.py; new diagnostic protocol.
**Interfaces:** select_indices(lengths,targets); same_length_permutation(lengths,targets); measure(model,z,lengths,targets); teacher_step_parity(model,z,lengths,targets).
- [x] RED: fixed length-balanced selection; missing/distinctness failures; controls preserve length and move every item; termination error counted; teacher/step parity for each real reader.
- [x] GREEN: implement train-only selection, direct teacher metrics and free-running structural report with first-divergence histogram.
- [x] Verify: focused tests pass. Read old checkpoints and reproduce their selected training-set error.

## Task2 — Bounded memorization runner and artifact verification

**Files:** same runner/tests; docs/implementation/decoder-memorization.md.
**Interfaces:** FitConfig; SuccessStreak.observe(exact); fit(kind,z,lengths,targets,output,config,provenance); main(argv).
- [x] RED: reset success streak after failure; invalid config rejected; exhausted budget cannot claim L1; saved weights restore predictions; absent lock rejected by CLI.
- [x] GREEN: fixed full-batch loop, final zero/swap controls, artifact hashes, immutable protocol/input verification, CLI and JSON reports.
- [x] Verify: full test suite passes; one independent review and RED→GREEN correction of important findings if any.

## Task3 — Real experiment and publication

- [x] Run all6 fits under the frozen diagnostic protocol. Save step curves, teacher/free predictions, input controls, exact selection and provenance. Report every run, including exhausted budgets.
- [x] Summarize what succeeded and what remains ambiguous; retain negative results and unchanged L1 status.
- [x] Publish source/tests/protocol/compact reports directly to main; verify remote tree hashes and main ref. Generated model weights remain local.

## Recorded result

179 tests passed; all6 runs memorized their train targets. Eight-history success confirmed at275steps(concat),400(attention),275(GRU). Same-length input swaps follow donor targets8/8, while recipient targets match0/8. No generalization/L1 claim. See docs/implementation/decoder-memorization.md and experiments/results/decoder-memorization-v0.1/.
