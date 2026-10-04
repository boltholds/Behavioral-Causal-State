# T0: real LearnLib TTT deterministic baseline

The registered `ttt-v0.1` run passes the R0 gate: **20 frozen fixtures × 3 fresh learner processes = 60 completed trials**, with zero remaining distinguishing counterexamples after exact equivalence checks. This is the finite deterministic predictive baseline, independent of language training. It completes the TTT component of G0; it does not claim the entire G0 or any L1 gate.

## What actually ran

`experiments/ttt/LearnLibTTT.java` instantiates the upstream
`de.learnlib.algorithm.ttt.mealy.TTTLearnerMealy<Integer,Integer>` with
`AcexAnalyzers.BINARY_SEARCH_BWD`, calls `startLearning()`, and refines hypotheses using exact counterexamples. There is no substitute learning algorithm, learned-model lookup, or approximate EQ. Constructor and query APIs were checked against the official 0.18.0 source artifact and compiled against the pinned upstream bytes.

- Official algorithm description: <https://learnlib.de/learnlib/maven-site/0.18.0/learnlib-build-parent/learnlib-algorithms-parent/learnlib-algorithms-active-parent/learnlib-ttt/index.html>
- Official API source: <https://repo.maven.apache.org/maven2/de/learnlib/learnlib-ttt/0.18.0/learnlib-ttt-0.18.0-sources.jar>
- Pinned runtime artifact: `de.learnlib.distribution:learnlib-distribution:jar:dependencies-bundle:0.18.0`. The lock records URL, SHA-256, and byte size. The official bundle includes LearnLib and its runtime dependencies, including AutomataLib. No jar or compiled class is part of the deliverable.

The actual runtime was OpenJDK 17.0.20. Its compiler module was invoked directly (`java -m jdk.compiler/com.sun.tools.javac.Main --release 17`) because this environment exposes the module without a `javac` executable. Maven is unnecessary for reproduction. The Python downloader verifies dependency bytes before execution. `-XX:ActiveProcessorCount=2` and `-Xmx256m` bound the Java runtime; trials run sequentially.

## Frozen fixtures and semantics

`experiments/ttt/fixtures-v0.1.json` was frozen before the registered run. Every input symbol is an explicit ordered `Step`/`Clamp` description. The 20 fixtures are the Cartesian product of reset `(C0,C1) ∈ {00,01,10,11}` and these five alphabets:

| Profile | Symbols | Exact inputs |
|---|---:|---|
| `commands` | 5 | hold both; start0; stop0; start1; stop1, in that order |
| `parallel_commands` | 9 | Cartesian product of each object's actions, ordered hold/start/stop for object0 then object1 |
| `commands_c` | 9 | commands followed by hold with C=0/1 clamps for object0 then object1 |
| `commands_single_clamps` | 21 | commands followed by every hold with one R/M/C/Y=0/1 clamp, ordered object, variable, value |
| `commands_joint_clamps` | 49 | preceding 21 symbols, then hold with each object's (R,M), (R,C), (M,C) pair clamped to 00/01/10/11; finally both objects' C clamped to 00/01/10/11 |

The JSON lists every symbol rather than relying on the description to regenerate it. Single and joint clamps only use the existing simulator API. The final profile is a declared finite joint-intervention subset, not the entire possible Step/Clamp interface.

Reset calls `WorldSpec.reset(C,0,0)` for each object. Every MQ begins from this fixed reset and executes the complete prefix plus suffix using `WorldSpec.advance` and the typed frozen `Step`. All after-step values are observed: `(R0,M0,C0,Y0,R1,M1,C1,Y1)`. They are packed injectively into an integer, MSB first; the Java output word contains these observations. A prefix/suffix query returns only suffix outputs while still executing the prefix. Empty words produce empty output words. Stochastic regimes are rejected explicitly.

The Java process receives only alphabet size, reset-based membership outputs, and counterexample input/output words. It receives no physical state, reference state ID, causal labels, table of simulator transitions, or language-training data. Alphabet indices are input symbols, not simulator states.

## Exact evaluator and independent checks

Python first enumerates the complete reachable full-variable simulator automaton by BFS from each reset under its declared alphabet. This evaluator-only reference can include transient M/Y values that do not affect future transitions. A separate Python partition-refinement implementation determines its minimal number of Mealy residual states; it is independent of LearnLib's discrimination tree.

At every conjecture Java exports its full transition/output table. Python explores the reachable reference × hypothesis product in BFS order, comparing **every outgoing output**. It returns the shortest distinguishing word or accepts only after exhausting the product. The finite product search has no sampling, horizon cutoff, or approximate EQ. Exact EQ has privileged access to the complete simulator reference; this is the R0 toy MAT setting and is not a deployable black-box EQ claim. Counterexample outputs are read from that reference table; no state information is sent back.

Each final exported hypothesis is checked again in Python, and its size must equal the independently minimized reference. Saved results retain the exported hypothesis and its SHA-256, every MQ prefix/input/output triple, and every complete EQ conjecture. Validation replays the MQ transcript against actual simulator reset and steps, derives all query/reset/symbol counters, and recomputes each counterexample and full-product traversal count. Wire-table validation rejects negative, Boolean, out-of-range, or malformed state/output values before indexing. Negative checks deliberately corrupt the learned initial hold output (distinguished by `[0]`) and offer a false one-state all-zero conjecture (distinguished by `[1]`). Both are rejected; the report also records these witnesses.

## Results

Each row below describes one fixture per reset, with identical counts in all three repetitions and all four reset choices. Repetitions are fresh deterministic Java processes; they are reproducibility repeats, not independent random seeds or uncertainty estimates.

| Profile | Reachable full states | Minimal / learned states | MQ per trial | EQ per trial | Executed symbols per trial |
|---|---:|---:|---:|---:|---:|
| commands | 4 | 4 / 4 | 59 | 2 | 158 |
| parallel_commands | 4 | 4 / 4 | 107 | 2 | 250 |
| commands_c | 16 | 16 / 16 | 539 | 2 | 1,942 |
| commands_single_clamps | 80 | 16 / 16 | 1,259 | 2 | 4,534 |
| commands_joint_clamps | 80 | 16 / 16 | 2,939 | 2 | 8,818 |

Across all 60 trials: **58,836 MQ/reset operations, 120 EQ operations, 188,424 executed MQ symbols**, and zero remaining exact counterexamples. Every trial required one counterexample refinement and a final accepting EQ. Query counters count actual bridge requests, without a membership cache. Executed symbols include complete prefix replay; they exclude privileged reference construction/EQ table traversal. Product pairs/edges examined are reported separately for every EQ. The Java bundle emits the benign SLF4J no-provider warning; exact stderr is preserved in each trial.

## Provenance and reproduction

The pre-run manifest binds the exact simulator source, Python bridge/evaluator, Java source, protocol, frozen fixture manifest, and dependency lock to SHA-256 hashes. It records dependency byte hashes, hashes of all compiled Java classes (including the oracle bridge), upstream TTT class byte hash, runtime versions, command, and registration time. Validation freshly compiles the bound Java source and checks both the compiled and upstream class identities, the declared learner class, all counterexample history, the derived summary, and the deliberately false negative controls. This API-backed checkout has no local Git metadata, so the report uses a content-addressed source snapshot instead of inventing a commit ID. The report is valid for those source bytes; edits to any bound file invalidate validation. Documentation is outside that executable source binding.

From the repository root, with its Python dependencies and Java 17:

```bash
PYTHONPATH=src python -m experiments.ttt.runner run
PYTHONPATH=src python -m experiments.ttt.runner validate
python -m pytest tests/test_ttt_oracle.py tests/test_ttt_integration.py -q
```

Runtime jars and compiled classes stay in `/tmp/artifacts` (override with `--cache`); first use downloads only the pinned official jar. The registered outputs are `experiments/results/ttt-v0.1/manifest.json`, `trials.json`, and `report.json`. Report validation re-enumerates the simulator references and checks every saved hypothesis rather than trusting a saved pass label.

The TDD development checks first failed for the missing oracle and real Java bridge, then passed against the implemented simulator and actual learner. Final targeted checks: **25 passed**, including real Java learning, full observation/reset, prefix handling, stochastic rejection, minimality of transient states, complete EQ traversal, false/corrupted conjectures, report-source corruption rejection, falsified learner/binary identities and counters, erased MQ/EQ counterexample evidence, summary/negative-control corruption, and invalid machine indices/observations. Development integration smoke runs were separate from the registered 60-trial experiment.

A full project `python -m pytest -q` during concurrent implementation produced **206 passed, 5 failed**. All five failures were in `tests/test_l1_training.py`: `test_resume_reproduces_uninterrupted_weights_and_metrics`, `test_resume_refuses_changed_config_inputs_or_identity`, `test_completed_checkpoint_corruption_is_rejected`, `test_training_never_accepts_overlapping_validation_groups`, and `test_train_job_alone_never_claims_passed_l1`. At that point `experiments.l1.training.fit_run` was still absent in the concurrently implemented L1 branch; no TTT test failed. After the concurrent implementation and validator fixes, the full suite was rerun with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q`: **235 passed** in 64.01 seconds. This later green run supersedes the earlier point-in-time failures.

## Limits

This result covers only two deterministic binary devices, the 20 declared reset/alphabet fixtures, full after-step output, reproducible reset, and privileged exact EQ. It does not establish stochastic learning, arbitrary alphabets, factorized scaling gains, partial-language grounding, causal discovery, or correctness of do/CF answers. Mealy state minimality concerns future predictive behavior: historical M/Y differences may disappear without a distinct residual state. No L1 dataset, embeddings, model checkpoint, or test label is used here.

Independent review identified validator false positives without finding an error in the actual learner results. The validator was hardened with 17 regressions first observed failing, and all 60 trials were rerun under the changed source hash. The algorithm source, fixtures, protocol thresholds, and runtime dependency bytes were unchanged; the rerun adds complete oracle transcripts and stricter evidence verification.
