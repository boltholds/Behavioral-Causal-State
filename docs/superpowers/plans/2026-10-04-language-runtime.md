# Language runtime implementation plan

> **For agentic workers:** Use superpowers:executing-plans; implementation and one final independent review. Direct main is already authorized.

**Goal:** Implement the specified SONAR readers and a reproducible training/validation path, verify actual SONAR embeddings and train all three readers in a bounded diagnostic run.
**Architecture:** Verified local SONAR weights produce float32 per-utterance caches. A public-only loader supplies embeddings, order, mask and declared source metadata. A shared autoregressive structured decoder produces typed events for the frozen Boolean core. Labels enter only the supervised loss/evaluator.
**Tech Stack:** Existing core; optional PyTorch training dependency; isolated official SONAR/fairseq2 environment when its NumPy/PyTorch pins differ.
**Spec:** docs/specs/03-language.md, 04-generator.md, 05-evaluation.md; experiments/protocols/language-v0.1.json.

## Global constraints

- Use the approved concat 256/256, attention 2 blocks/8 heads/FFN1024, GRU256; common decoder GRU256; dropout0.1.
- Frozen sentence encoder, rus_Cyrl, float32, no extra normalization or raw-text bypass.
- Decode time, mode, entity, predicate, negation, value autoregressively, with type masks and EOS. Greedy evaluation never receives true prefix.
- Preserve existing dataset and protocol hashes. Template review by the assistant is recorded separately from human review.
- Exact two initial C observations establish device-0 then device-1 by first introduction. This convention is checked and documented; no hidden name map enters the reader.
- Full registered L1 remains not_run until human catalog review, complete run lock, tuning and five final seeds. Diagnostic training cannot emit passed L1 gates.
- Source control through the existing API-backed main checkout; do not alter historical result manifests.

## Review focus

1. Raw text, test labels, names or event counts must not reach the neural forward API.
2. Padding must not affect any reader; attention must remain ordered and decoder conditioning causal.
3. Wrong/missing EOS, impossible clocks and inconsistent predicted evidence must count as errors.
4. Asset/cache/checkpoint corruption or mismatched dataset/protocol must fail closed.
5. Diagnostic checkpoints/reports must never be accepted as registered L1 completion; catalog human status cannot be forged by automatic review.

## Task 1 — Structured vocabulary and catalog review

Files: language_tokens.py; tests/test_language_tokens.py; docs/implementation/catalog-review.md.
Interfaces: encode_events(events)->tuple[int,...]; decode_events(tokens)->DecodedEvents|InvalidDecoding; allowed_tokens(prefix)->tuple[int,...].
- [x] RED: all six modes round-trip, unknown != false, valid masks depend only on predicted prefix, malformed/EOS/clock rejection.
- [x] GREEN: compact field vocabulary, immutable result variants, fixed 16-event cap. Inspect all 24 templates in explicit/implicit forms and entity introduction convention.
- [x] Verify: targeted tests pass; record review scope and outstanding human review.

## Task 2 — Locked SONAR assets and public-only embedding cache

Files: sonar_encoder.py, embedding_cache.py; tests/test_embedding_cache.py; experiments/encoders/sonar-v0.1.json.
Interfaces: SentenceEncoder Protocol; lock/check local checkpoint/tokenizer SHA256; build_cache(public_path,encoder,output); load_cache(output,public_path).
- [x] RED: same text encoded once, ordering/dtype preserved, no sidecar inputs, checksum and text-ID mismatch rejected.
- [x] GREEN: official local AssetCard loader, frozen eval model, exact text cache; role/speaker metadata restricted to current declared catalog.
- [x] Verify: real 1024-dimensional Russian SONAR embeddings with locked bytes and runtime manifest; tests pass.

## Task 3 — Readers and autoregressive event decoder

Files: grounder.py; tests/test_grounder.py.
Interfaces: ReaderKind enum; Grounder(kind,input_dim).teacher_logits(z,lengths,target_tokens), .greedy(z,lengths); forward accepts tensors only.
- [x] RED: all readers support lengths4/8/12/16, padding invariance, gradients, order sensitivity, no future target dependence; greedy has no targets.
- [x] GREEN: specified readers, common GRU token decoder, type masks from predicted prefix, EOS bound.
- [x] Verify: CPU tests pass; parameter counts recorded; synthetic learning control only marked test fixture.

## Task 4 — Training, evaluation, real diagnostic and publication

Files: language_training.py, language_evaluation.py, language_cli.py, language_lock.py; pyproject.toml; tests/test_language_training.py, tests/test_language_cli.py, tests/test_language_lock.py.
Interfaces: train/validation loaders, train_one, frozen checkpoint, evaluate decoded events through shared core; CLI subset/cache/lock/train; train includes checkpoint restoration and validation evaluation.
- [x] RED: deterministic training/reload; train IDs match labels; validation labels stay outside forward; altered cache/checkpoint rejected; diagnostic report gates remain not_run.
- [x] GREEN: AdamW, early stopping by free-running event error then NLL, clipped gradients, per-field/full-event/history errors and causal TV with invalid penalty1. Record seeds, hashes, parameters, timing and actual settings.
- [x] Verify: real SONAR diagnostic for concat/attention/GRU; all tests, one final independent review, one regression fix pass if needed.
- [x] Publish code and compact reproducible reports to main; explicitly report full L1 and TTT as unfinished.

## Recorded result

156 tests passed. Verified official SONAR encoded 128/32 histories; all three readers trained and restored under one immutable input lock. No reader recovered a complete validation history. Full L1 remains not_run_diagnostic. See docs/implementation/language-runtime.md and experiments/results/language-diagnostic-v0.1/summary.json.
