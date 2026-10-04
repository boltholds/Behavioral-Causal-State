# Executable causal core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Первый воспроизводимый Python runtime: exact-key journal, физика двух устройств, точный conditional do/CF вывод, базовые текстовые трассы, G2 и P1 с отчётами.
**Architecture:** Immutable dataclasses/enums на границах; чистые механизмы; динамическое программирование по совместным factual/counterfactual состояниям; отдельные search и certification; CLI сохраняет manifest и результаты.
**Tech Stack:** Python 3.12+, NumPy, SciPy, pytest; setuptools; без нейронных весов на этом этапе.
**Spec:** docs/specs/02-contracts.md, 04-generator.md, 06-population.md, 09-misspecification.md; существующие JSON protocols/fixtures неизменны.

## Global Constraints

- Работа непосредственно в main разрешена пользователем. Существующие remote-файлы сохраняются.
- Видимые события не содержат скрытый M в partial; evaluator trace не подаётся inference.
- CF использует factual evidence и shared U; будущий шум интегрируется.
- Найденный диапазон не становится certified bounds. Прерванная проверка не доказывает несовместимость.
- Никаких заявлений о прохождении L1, полном G1 или LearnLib TTT без соответствующего запуска.
- Этот этап не обучает SONAR/агрегаторы; challenge split manifest и обучение — следующий этап. Базовый генератор не выдаётся за полный зарегистрированный L1 dataset.

## Review Focus

Zero-support evidence; рассогласованные временные индексы; скрытые поля в JSONL; clamp persistence; поздние factual observations при CF; float/duplicate JSON keys; ошибочное повышение статуса при timeout; finite-data и population scopes; конфигурация полного и smoke запуска.

## Task 1 — Contracts and journal

Files: pyproject.toml, src/bcs/contracts.py, src/bcs/journal.py, tests/test_journal.py, tests/test_contracts.py.
Interface: frozen AnswerRecord; exact Request payload; SQLite append-only replay.
- [x] RED: golden 12 keys, invalid wire inputs, replay/collision/conflicting-write, false identification.
- [x] GREEN: implement types, validation, codec and journal.
- [x] Verify: `python -m pytest tests/test_journal.py tests/test_contracts.py -q` → all pass.

## Task 2 — Simulator and exact inference

Files: src/bcs/simulator.py, src/bcs/inference.py, tests/test_simulator.py, tests/test_inference.py.
Interface: WorldSpec/NoiseLaw, State, Step, Evidence; distribution or typed undefined/budget result.
- [x] RED: fixture laws, clamps, indexed noise, conditional past CF distinct from fresh do, impossible evidence, budget.
- [x] GREEN: exact Fraction transition kernel and twin-state forward filtering.
- [x] Verify: `python -m pytest tests/test_simulator.py tests/test_inference.py -q` → all pass.

## Task 3 — Visible traces and renderer

Files: src/bcs/generator.py, tests/test_generator.py.
Interface: immutable event variants; learner text separate from evaluator trace; deterministic seeds.
- [x] RED: fixed lengths, forced command, partial observation, hypotheses/negations no clock advance, replay of generated evidence.
- [x] GREEN: SPEC-04 base sampler and two Russian template families; JSONL export only selected public fields.
- [x] Verify: `python -m pytest tests/test_generator.py -q` → all pass.

## Task 4 — Calibration experiments and CLI

Files: src/bcs/misspecification.py, population.py, cli.py, metrics.py; tests/test_misspecification.py, test_population.py, test_cli.py.
- [x] RED: exact compatibility matrix, verified witnesses, CP edge counts, interrupted certification, M-/M+, four arm budgets, CLI smoke artifacts.
- [x] GREEN: analytical certification, confidence intervals, fixed-budget P1, protocol-controlled CLI.
- [x] Verify: full pytest suite; full G2 and P1 runs with stored per-seed results, versions, source/protocol hashes.

## Task 5 — Documentation and integration

- [x] Record exact run commands, results and limits in docs/implementation/first-runtime.md; update README status.
- [x] Fresh review of runtime against specs; tests for substantive findings.
- [x] Verify package install, CLI, full tests, JSON reports; commit to main; confirm remote file hashes.

Execution evidence: [runtime report](../../implementation/first-runtime.md). Integration uses the authorized main branch through GitHub API; remote hashes are checked after publishing.
