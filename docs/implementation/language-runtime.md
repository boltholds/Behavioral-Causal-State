# SONAR и языковые ветки

Реализованы frozen SONAR adapter, публичный embedding cache, concat/attention/GRU, авторегрессионный event decoder, single-configuration trainer и validation evaluator. Команды обучения всегда маркируют результат `not_run_diagnostic`: orchestration полного L1 (3 learning rates × 3 branches, затем 5 seeds и закрытые test gates) ещё не реализована. TTT/EQ остаётся отдельной задачей.

## Окружения и веса

Core/training: Python≥3.12, `pip install -e '.[test,language]'`. У SONAR отдельное окружение: официальный `sonar-space==0.5.0`, `fairseq2==0.5.2`, CPU build `fairseq2n==0.5.2+cpu`, PyTorch2.8.0+cpu, NumPy1.26.4. Это позволяет сохранить NumPy2.x у core. Обмен — `.npy` float32 и JSON с hashes; нормализация не добавляется.

```sh
python -m venv /tmp/bcs-sonar-env
/tmp/bcs-sonar-env/bin/pip install torch==2.8.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
/tmp/bcs-sonar-env/bin/pip install sonar-space==0.5.0 fairseq2==0.5.2 fairseq2n==0.5.2+cpu --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cpu
```

Модель — `facebook/SONAR` revision `a551c586dcf4a49c8fd847de369412d556a7f2f2`, `text_sonar_basic_encoder`, язык `rus_Cyrl`. Checkpoint и tokenizer проверяются по размеру и SHA256 **до** загрузки; используется локальная AssetCard поверх официальной конфигурации. Зафиксированные файлы и команды загрузки находятся в [encoder manifest](../../experiments/encoders/sonar-v0.1.json). Веса имеют upstream CC-BY-NC-4.0; в этот репозиторий они не включены.

## Запуск

Из корня репозитория, с уже подготовленным полным родительским dataset:

```sh
python -m bcs.language_cli subset --parent /tmp/bcs-parent --output /tmp/bcs-data --train-count 128 --validation-count 32
PYTHONPATH=src /tmp/bcs-sonar-env/bin/python -m bcs.language_cli cache --public /tmp/bcs-data/public/train.jsonl --assets /tmp/sonar-assets --output /tmp/bcs-cache/train
PYTHONPATH=src /tmp/bcs-sonar-env/bin/python -m bcs.language_cli cache --public /tmp/bcs-data/public/validation.jsonl --assets /tmp/sonar-assets --output /tmp/bcs-cache/validation
python -m bcs.language_cli lock --dataset /tmp/bcs-data --parent /tmp/bcs-parent --cache-root /tmp/bcs-cache
python -m bcs.language_cli train --dataset /tmp/bcs-data --cache-root /tmp/bcs-cache --branch concat --output /tmp/bcs-concat
python -m bcs.language_cli train --dataset /tmp/bcs-data --cache-root /tmp/bcs-cache --branch attention --output /tmp/bcs-attention
python -m bcs.language_cli train --dataset /tmp/bcs-data --cache-root /tmp/bcs-cache --branch gru --output /tmp/bcs-gru
```

Каталоги назначения должны быть пусты. `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--device`, `--threads` фиксируются в отчёте. Даже запуск на всех данных остаётся diagnostic, пока не выполнен полный протокол.

Cache читает только один public JSONL. Он дедуплицирует точные тексты, сохраняет исходный порядок, не объединяет парафразы. Индексы, file hashes и соответствие public source проверяются при чтении. Допустимые роли этого каталога — `user/speaker-0`; другие значения отвергаются до обучения. Forward принимает только embeddings и длины. Имена, Γ, test labels и raw text не передаются модели обходным путём.

`lock` заново аудирует parent и требует точного совпадения prefix subset, train labels, validation sidecar и frozen core с ним. Фиксируются hashes файлов данных, всех cache artifacts, protocol и исходников. `train` проверяет их до и после обучения. Существующий lock не перезаписывается: изменение данных/кода требует нового каталога запуска. Generic encoder API остаётся пригодным для unit fixtures; CLI требует полный pinned SONAR identity. Это проверка контракта и происхождения локального запуска, не криптографическая аттестация произвольного стороннего cache publisher.

## Декодирование и обучение

Словарь из 40 tokens: PAD/BOS/EOS плюс отдельные time/mode/entity/predicate/negation/value значения. Один event — 6 tokens, максимум16 events и EOS. Masks разрешают типы на основании уже выведенного префикса. Physical consistency проверяется после greedy decode: malformed clock, незавершённый event и неправильный EOS дают InvalidDecoding. Unknown и бинарный0 различаются.

Concat читает flatten16×1024, mask и длину через MLP256/256. GRU читает исходные vectors и отдаёт последний hidden256. Attention использует projection256, learned positions, 2 блока, 8 heads, FFN1024; decoder читает все позиции через cross-attention. Shared decoder: token embedding64, GRU256 и vocabulary projection. Для attention начальное состояние и постоянный input context — masked mean memory; на каждом шаге добавляется cross-attention к текущему decoder state. Это конкретизация SPEC-03; повторный доступ к sequence сохранён.

Teacher forcing применяется только для loss. Greedy не принимает targets. AdamW(weight_decay0.01), clip1.0; early stopping по validation full-event error, затем teacher-forced NLL. Лучший checkpoint, его SHA256, конфигурация, исходники, target/cache hashes и версии сохраняются. При восстановлении обязательна ожидаемая input-lock identity; проверяются scope, исходники, весовые байты, identity энкодера и hash checkpoint manifest из run report.

Структурные метрики: free-running per-field confusion/errors, full-event/history errors, invalid decodings. Causal diagnostics: μ_lang по четырём категориям, μ_joint, две sensitivity смеси и TV к тому же frozen core. На invalid output TV=1. Отдельная оценка branch-vs-simulator, rare strata, challenge critical errors/noninferiority и полный набор registered gates ещё не включены в этот диагностический runner.

## Каталог и границы

[Проверка всех 48 форм и соглашений](catalog-review.md) выполнена ассистентом. Human review остаётся pending. Первый/второй initial C measurement вводит device-0/device-1; это соглашение benchmark, не entity discovery. Протокол L1 и исторические dataset manifests не изменены.

Диагностический запуск служит проверкой вычислений и воспроизводимости. Он не устанавливает достаточность SONAR, превосходство агрегатора или прохождение причинных gates. Для этих выводов нужны полный preregistered бюджет и held-out evaluation.

## Фактический диагностический результат: 2026-10-04

Проведены загрузка настоящих SONAR weights, вычисление embeddings и обучение всех трёх веток. Использованы первые 128 train и 32 validation histories из фиксированного порядка аудированного родительского датасета; группы между splits не пересекаются. Seed11, learning rate0.0003, batch64, максимум30 epochs, patience5. Набор validation использован для early stopping; held-out test не открывался для оценки языковых веток.

| Ветка | Параметры | Выполнено epochs / лучший epoch | Full-event error | Full-history error | Invalid histories /32 | TV μ_lang | TV μ_joint |
|---|---:|---:|---:|---:|---:|---:|---:|
| concat | 4 721 448 | 15 /10 | 0.937500 | 1.000000 | 32 | 1.000000 | 1.000000 |
| attention | 2 565 928 | 23 /18 | 0.894578 | 1.000000 | 25 | 0.929688 | 0.802083 |
| GRU | 1 441 320 | 30 /27 | 0.914943 | 1.000000 | 32 | 1.000000 | 1.000000 |

TV сравнивает ответы через одно frozen core при predicted и oracle grounding. Invalid decoding и неопределённый запрос получают штраф1. Full-event error учитывает недостающие и лишние события; его знаменатель может различаться между ветками. Teacher-forced validation NLL выбранных checkpoints: 0.942985 /0.740821 /0.775550 соответственно. Снижение NLL не обеспечило правильное free-running восстановление: **ни у одной ветки нет полностью правильной validation history**.

Это отрицательный результат качества при данном малом бюджете. Он не доказывает недостаточность SONAR и не устанавливает порядок качества агрегаторов. Статус полного L1 — `not_run_diagnostic`; менять preregistered gates по этим числам нельзя. Следующий эксперимент должен отдельно проверить обучаемость decoder на малом наборе до перехода к полному tuning и закрытым тестам.

SONAR закодировал 621 уникальный train text и 234 validation text за72.77s, без загрузки модели в этом таймере. Использованы CPU, 4 threads, outer cache batch32, внутренний SONAR batch16, float32 без нормализации. Обучение заняло16.30s /33.24s /32.67s. Training environment: Python3.12.14, PyTorch2.14.1+cpu, NumPy2.3.5; encoder environment приведён выше. Время обучения включает validation по эпохам, но не последующий causal report.

Проверка: **156 tests passed in20.88s**. После обучения повторно проверен input lock, каждый checkpoint восстановлен с проверкой весовых bytes, metadata и исходников. Финальное независимое ревью выявило две существенные проблемы provenance; исправлены обязательным input lock и полной проверкой pinned encoder identity. Соответствующие regression tests сначала воспроизвели ошибки, затем прошли.

Машиночитаемые [сводка](../../experiments/results/language-diagnostic-v0.1/summary.json), [input lock](../../experiments/results/language-diagnostic-v0.1/input-lock.json), per-epoch/per-field reports и сырые validation predictions сохранены в [каталоге результатов](../../experiments/results/language-diagnostic-v0.1). Checkpoint weights и embedding matrices — локальные генерируемые артефакты; в git опубликованы их hashes. Команды выше воспроизводят отбор записей и процедуру; новый runtime или иной JSON serializer требует собственного lock и не обещает побайтного совпадения всех артефактов.

## Оставшиеся ограничения после ревью

- Для внешнего token sequence с правильными событиями, но без EOS, full-event error может быть нулевым; full-history error и invalid-decoding остаются ошибочными. Текущий masked greedy всегда завершает EOS, поэтому выбор checkpoint этим случаем не затронут. Отдельный termination score отложен.
- Upstream SONAR может усекать сообщения длиннее model token limit с предупреждением. Текущий зафиксированный короткий каталог не достигает лимита. До поддержки свободного длинного текста adapter должен отклонять truncation.
- Human review каталога, полная orchestration L1, challenge/rare metrics и TTT ещё не выполнены.
