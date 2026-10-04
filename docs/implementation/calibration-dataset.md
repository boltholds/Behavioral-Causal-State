# Обученное ядро и подготовка L1 dataset

Этот этап обучает таблицы механизмов и собирает кандидатный набор для языкового эксперимента. SONAR/grounder не обучаются. `training_ready=false` сохраняется до human review каталога и фиксации encoder/tokenizer в отдельном L1 manifest.

## Выполненный запуск: 2026-10-04

122 теста пройдены. Полная сборка с seed=20261004 прошла усиленный аудит; повторная сборка дала побайтно те же файлы данных. Source hashes в manifest совпадают с публикуемой реализацией.

| Проверка | Результат |
|---|---:|
| Видимые calibration probes / физические шаги | 12 / 22 |
| Train / validation / IID groups | 20 000 / 4 000 / 4 000 |
| Challenge groups / публичные истории всего | 14 000 / 56 000 |
| Semantic duplicates между splits | 0 |
| Пары с причинным witness / invariant / structural-only | 8 000 / 4 000 / 2 000 |
| Локальные переходы / reset cases, проверенные исчерпывающе | 3 888 / 2 |
| Несовпадения learned core с reference kernel | 0 |
| Средняя TV на validation: μ_lang / μ_joint | 0 / 0 |
| Односторонняя empirical Bernstein upper, α=0.05, для каждого panel | 0.0021523844443442153 |
| Oracle validation threshold | 0.02 — passed |

Все четыре μ_lang category means равны нулю. Для каждого challenge type сохранено по 1 000 groups в двух holdout strata. Все knownness pairs получили `StructuralOnly`: при известном расписании команд новое измерение R не меняет ответ в этом стенде.

Фактические артефакты: [manifest](../../experiments/results/calibration-dataset-v0.1/manifest.json), [audit](../../experiments/results/calibration-dataset-v0.1/audit.json), [core evaluation](../../experiments/results/calibration-dataset-v0.1/core-evaluation.json), [core](../../experiments/results/calibration-dataset-v0.1/core.json), [calibration](../../experiments/results/calibration-dataset-v0.1/calibration.json), [candidate catalog](../../experiments/results/calibration-dataset-v0.1/catalog.json), [verification](../../experiments/results/calibration-dataset-v0.1/verification.json). Крупные JSONL воспроизводятся командой ниже; в git сохранены компактные отчёты и их checksums.

Нулевые ошибки относятся к learned deterministic core при oracle grounding и заданных assumptions. `oracle_validation_gate=passed` не завершает L1: языковые ветки ещё не запускались.

## Запуск

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python -m bcs prepare-l1 --smoke --output /tmp/bcs-l1-smoke
python -m bcs prepare-l1 --output /tmp/bcs-l1-full
python -m bcs audit-l1 --dataset /tmp/bcs-l1-full
```

Каталог назначения должен отсутствовать или быть пустым. По умолчанию split seed=20261004. Полный набор содержит 20 000 train, 4 000 validation, 4 000 IID histories и 14 000 challenge groups по два текста: всего 56 000 публичных историй. Сборка и полная проверка требуют нескольких минут CPU; прогресс выводится в stderr.

## Calibration

`calibration_panel` выполняет 12 probes: 6 строк R(previous_R,action), 2 строки M(R), 4 строки Y(M,C). Для подготовки используются только single-target clamps. Всего 22 физических шага, из них 10 подготовительных.

`learn_core` принимает только `CalibrationProbe` с видимой `History`. У него нет параметров WorldSpec, hidden U или evaluator trace. Target, который принудительно установлен в момент измерения, не обучает свой естественный механизм. Незаполненные ячейки возвращают `IncompleteCalibration`, противоречивые — `ConflictingCalibration`; обе ситуации блокируют freeze.

`BooleanCore` хранит immutable tables, assumptions и канонический artifact ID. Тот же Dynamics Protocol поддерживают reference simulator и обученное ядро. Контроль с изменённым Y-механизмом доказывает, что learner не копирует исходную формулу. C-copy, R-reset=0, известный граф, отсутствие шума и независимость устройств заданы явно, а не обнаружены из данных.

## Минимальные пары

| Вид | Построение | Проверка |
|---|---|---|
| paraphrase | Одинаковая Γ, другое семейство текстов | Равенство допустимых query answers |
| negation | Выполненный stop ↔ отказ от новой команды stop | Пересчитанный clock и различающий запрос |
| mode | Выполненный stop ↔ гипотетическая команда stop | Пересчитанный clock и различающий запрос |
| order | start;stop ↔ stop;start | Нет явных timestamps; различается physical transition order |
| time | Поменять порядок рассказа о двух событиях с timestamps | Physical chronology и ответы сохраняются |
| binding | Новая команда устройству A ↔ B | Верифицируется изменённая привязка |
| knownness | Новое измерение R=0 ↔ запись без нового измерения | Отдельный structural test; witness только если существует |

Transform применяется к event plan, factual observations регенерируются. Для order пары не содержат зависимых от перестановки последующих measurements. Witness выбирается до любых model outputs из фиксированного Q_eval; TV≥1/4. Отсутствие witness даёт StructuralOnly. В этом deterministic known-schedule стенде R часто полностью определяется прошлыми командами, поэтому knownness не обязан менять causal answer.

На вид — 1 000 groups новых template families и 1 000 groups новых имён. Train использует семьи 0/1, template holdout — 2/3; name holdout имеет непересекающийся словарь. Имена и семейства выбираются отдельным seed stream независимо от физических значений.

## Splits и артефакты

Semantic fingerprint исключает поверхность текста и имена, сохраняет time/mode/entity/predicate/negation/value и multiplicity; явное время определяет физическую хронологию. Это evaluator-only ключ. AnswerJournal продолжает побайтно различать тексты.

Для base group длина сначала выбирается равномерно из 4/8/12/16 и удерживается при повторных попытках. Semantic hash назначает владельца train/validation/IID в отношении 5:1:1. Дубликаты внутри split разрешены; между split запрещены. Пара отклоняется, если любая её сторона совпадает с base из другого split. Все derivatives одной группы записаны вместе. Числа rejected candidates публикуются.

Выбранные детали generator v1: challenges имеют длины 8/12/16; для каждой base history явные timestamps выбираются с вероятностью 0.5, иначе используется implicit clock. Неявное время проверяется по порядку сообщений, а out-of-order narration допускается только с явными timestamps. Эти решения записаны в manifest, они не выбираются по качеству обученных моделей.

| Артефакт | Доступ |
|---|---|
| `public/*.jsonl` | Только record ID, group ID, сообщения role/speaker/text |
| `training/labels.jsonl` | Только train Γ labels |
| `evaluator/*.jsonl` | Γ, физические references, catalog metadata, pair witnesses; learner не получает |
| `calibration.json`, `core.json` | Видимые probes и один общий frozen core |
| `catalog.json`, `manifest.json` | Families/names, source/data hashes, counts, rejection statistics |
| `audit.json`, `core-evaluation.json` | Результаты проверки файлов и oracle validation preflight |

В JSONL нет полных PhysicalTrace или hidden U. Аудитор перечитывает файлы и сверяет реальные наблюдения, rendering, witnesses, counts, IDs и splits; он не доверяет заявленному статусу или одному совпадению checksum. В regression tests содержимое намеренно портится вместе с обновлением checksum — проверка всё равно должна отказать.

Проверяется весь calibration panel: зарегистрированное расписание, single-target clamps и каждое наблюдение, включая поля, которые learner не использует. Длины обязаны принадлежать точной поддержке, а histogram в manifest — совпадать с файлами. Посторонние файлы в `public/`, `evaluator/` и `training/` отклоняются. Самостоятельный вызов `evaluate_core` всегда запускает этот аудит: при отказе возвращается `oracle_validation_gate=invalid_dataset`, числовой gate не вычисляется.

## Границы результата

Integrity audit относится к созданным артефактам. Он не заменяет проверку реального training loader после появления нейронных веток. Проверка oracle core на validation — предварительное условие для L1; итоговые held-out языковые метрики не получены. TTT/MAT, entity discovery и stochastic mechanism learning остаются следующими задачами.

Human review каталога pending: шаблоны проверены на механическую согласованность, но не считаются подтверждённой моделью естественного русского языка. До первого обучения нужны review каталога, checkpoint/tokenizer hashes и training manifest. Размеры набора и числовые gates исходного L1 протокола не изменены.
