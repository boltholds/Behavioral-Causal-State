# Calibration and dataset implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Обучить детерминированное ядро по 12 видимым probes и подготовить воспроизводимый L1 dataset: 20k train, 4k validation, 4k IID, 7×2000 challenge pairs.
**Architecture:** Protocol Dynamics отделяет точный inference от simulator/learned tables. Calibration consumes only visible History. Pair transforms rebuild factual observations after semantic edits. Dataset exports public text separately from evaluator records and verifies group/semantic split separation.
**Tech Stack:** Python 3.12, existing NumPy/SciPy, pytest, JSONL; no new runtime dependency.
**Spec:** docs/specs/03-language.md, 04-generator.md, 05-evaluation.md; experiments/protocols/language-v0.1.json.

## Global Constraints

- Known graph; learn R/M/Y tables only, C-copy/reset/noise-zero are declared assumptions.
- 12 probes, no executed joint intervention during training; all preparatory steps counted.
- Same frozen core artifact for future branches. Missing/conflicting cells never filled from simulator formula.
- All pair derivatives remain in one split. Exact journal key is unrelated to evaluator duplicate key.
- Challenge: seven kinds, 1000 new-template and 1000 new-name groups each. Length ≤16, preserve at least one executed command.
- No neural model/test-driven tuning. Human language review remains explicit before L1 training lock.
- User authorized main; continue natively, final independent review before publishing.

## Review Focus

1. A clamped child cannot teach its natural mechanism; reject such probes.
2. Replacing execution by denial changes the clock; regenerate downstream observations and references.
3. A time permutation must preserve physical chronology and hide no changed semantics in metadata.
4. Both sides of a pair participate in duplicate exclusion; values, modes, time/order and bindings remain in the semantic key.
5. Invalid datasets never receive passed G1; audit exported artifacts, including private/public alignment and counts.

## Task 1 — Calibration and Dynamics protocol

Files: src/bcs/dynamics.py, calibration.py; simulator.py, inference.py; tests/test_calibration.py.
Interfaces: Dynamics.reset/advance/noise; CalibrationProbe(target,history,output_time,object_id); learn_core(probes) -> BooleanCore | IncompleteCalibration | ConflictingCalibration.
- [x] RED: learn changed Y mechanism, all 12 cells, missing/conflicting rows, clamped-child rejection, 22 physical actions.
- [x] GREEN: fixed panel from visible probes, immutable BooleanCore with canonical artifact hash; exact inference uses Dynamics.
- [x] Verify: `python -m pytest tests/test_calibration.py tests/test_inference.py -q`.

## Task 2 — Pair construction and annotation

Files: src/bcs/pairs.py, language_catalog.py; tests/test_pairs.py.
Interfaces: PairKind, HoldoutStratum, make_pair(seed,group_id,kind,stratum); pair events/messages plus evaluator witness.
- [x] RED: seven transformations, regenerated truthful observations, time/paraphrase invariance, order without explicit timestamps, exact distinguishing witness or structural-only tag.
- [x] GREEN: deterministic constrained transformations; catalog families/names independent of outcomes; typed pair records.
- [x] Verify: `python -m pytest tests/test_pairs.py -q`.

## Task 3 — Split builder, artifact auditor and CLI

Files: src/bcs/dataset.py, dataset_cli.py; cli.py; tests/test_dataset.py.
Interfaces: build_dataset(output,counts,split_seed); audit_dataset(output) checks content rather than trusting manifest.
- [x] RED: seeded reproducibility, derivative leakage, altered labels/text/witness, count and catalog holdout violations; learn-core integration.
- [x] GREEN: stable semantic ownership + pair exclusion, explicit rejection counts, JSONL separation, source/catalog/core hashes, CLI full/smoke modes.
- [x] Verify: whole pytest suite, smoke CLI, full registered-size dataset and audit.

## Task 4 — Evaluation, review and main

- [x] Compare learned core vs simulator for all local transitions (including joint clamps), μ_lang/μ_joint on generated validation data; no training feedback from test.
- [x] Publish compact report + frozen core + calibration evidence + reproduction commands. Large regenerable JSONL stays a generated artifact, no learner receives evaluator sidecars.
- [x] Fresh independent review; fix substantive findings RED→GREEN; whole suite (122 passed).
- [x] Update statuses precisely, commit directly to main and verify remote hashes.
