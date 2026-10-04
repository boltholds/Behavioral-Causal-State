# Полный протокол L1: исполнитель и фактический статус

Исполнитель полного фиксированного протокола реализован в `experiments/l1/study.py`. Зарегистрированные 24 обучения на полном SONAR-наборе **ещё не выполнены**. `experiments/l1/smoke.py` проверяет всю последовательность на отдельной малой fixture с синтетическими признаками; её результат всегда `not_run_fixture`.

Реальный deterministic TTT завершён отдельно: [60 прогонов и точная проверка](ttt.md). Его успех не сообщает ничего о качестве текстового grounder.

## Зафиксированный бюджет

| Этап | Запуски | Доступные метки | Выбор |
|---|---:|---|---|
| Tuning | 3 ветки × 3 LR = 9 | train + validation | full-event error, затем NLL, затем меньший LR |
| Freeze | одна запись | уже полученные validation metrics | все 9 завершены; победитель пересчитывается при чтении |
| Final | 3 ветки × 5 seeds = 15 | train + validation | новое обучение с нуля на выбранном LR |
| Test | 5 seed-панелей, по 3 ветки | IID + challenge | все 15 checkpoint заморожены; дообучение закрыто |

Без изменения `language-v0.1.json`: LR `{0.0001,0.0003,0.001}`, tuning seed `0`, final seeds `{11,23,37,51,79}`, batch `64`, максимум `30` эпох, AdamW/weight decay `0.01`, clip `1`, dropout `0.1`. Patience `5` отсчитывает эпохи без строгого улучшения **free-running full-event error**. Улучшение только NLL выбирает checkpoint среди равных event errors, но не обнуляет patience. Финальный decoder использует greedy decoding.

Training path загружает метки только из `training/labels.jsonl` и `evaluator/validation.jsonl`. Аудит целостности отдельно может читать evaluator sidecars всех splits; это не передача test labels оптимизатору. Замороженный SONAR заранее кодирует только публичные тексты всех splits без fitted normalization. SHA256 связывают данные, cache, encoder/tokenizer identity, core, протокол, исходники, версии runtime, настройки, checkpoint и отчёты. `code_commit` должен указывать фактический опубликованный снимок; checkout через API дополнительно проверяется по хэшам исходников. Одна строка SHA не заменяет эту проверку.

## Запуск

Запускать из checkout репозитория. Для neural training нужен `pip install -e '.[test,language]'`. Для получения SONAR embeddings используется отдельная совместимая среда из [SONAR-инструкции](language-runtime.md); ready float32 caches затем читаются обычной training-средой.

```bash
# Сначала получить все четыре проверенных публичных cache.
PYTHONPATH=src /path/to/sonar-env/bin/python -m experiments.l1.preparation \
  --dataset artifacts/l1-v0.1-final --assets artifacts/sonar-assets \
  --output artifacts/l1-embeddings-v0.1 --threads 2 --batch-size 16

# CODE_COMMIT: SHA опубликованного main с этим исполнителем, не SHA старого baseline.
CODE_COMMIT=$(git rev-parse HEAD)
PYTHONPATH=src python -m experiments.l1.study init \
  --dataset artifacts/l1-v0.1-final --cache artifacts/l1-embeddings-v0.1 \
  --output artifacts/l1-study-v0.1 --code-revision "$CODE_COMMIT" --threads 2
```

`init` проверяет полный размер, SONAR identity, dataset/cache hashes и G0/G1/status/replay fixtures. Реальный TTT report проверяется отдельно точным product EQ. Без человеческого review создаётся конкретный `blocked_catalog_review`; обучение не начинается. Это требование SPEC-04, а не дополнительное согласование записи в main.

[48 форм для проверки](../../experiments/reviews/l1-catalog-review-request.md) связаны с точным каталогом. После фактической проверки человек предоставляет отдельный receipt: `schema=l1-catalog-human-review-v1`, `catalog_sha256` из запроса, `decision=approved`, непустой `reviewer`, ISO `reviewed_at`, `forms_reviewed=48`. Pending request сам по себе receipt не является. Менять dataset manifest для одобрения не нужно.

```bash
PYTHONPATH=src python -m experiments.l1.study review \
  --output artifacts/l1-study-v0.1 --review /path/to/review-receipt.json --threads 2
PYTHONPATH=src python -m experiments.l1.study run \
  --output artifacts/l1-study-v0.1 --threads 2
```

`run` последовательно выполняет tuning, freeze, final и оценку. Повторный вызов продолжает незавершённый этап. Для ограничения одной сессии без изменения общего бюджета:

```bash
PYTHONPATH=src python -m experiments.l1.study tune \
  --output artifacts/l1-study-v0.1 --max-jobs 1 --max-new-epochs 1 --threads 2
PYTHONPATH=src python -m experiments.l1.study status \
  --output artifacts/l1-study-v0.1 --threads 2
```

Отдельные команды `freeze`, `final`, `evaluate` применяют те же ограничения. После открытия test обучение и повторный подбор запрещены. Параллельные CLI-писатели в один study отвергаются OS-lock; один output принадлежит одному процессу. Прямой Python API рассчитан на того же единственного владельца.

## Возобновление и измерение ресурсов

`training.py` сохраняет текущие и лучшие веса, optimizer state, CPU/CUDA RNG, NumPy shuffle RNG и patience. Завершённая эпоха сначала записывается в отдельный checkpoint, затем атомарно публикуется pointer с хэшем. Сбой посреди эпохи приводит к повтору этой незавершённой эпохи. `committed_optimizer_steps` не включает отброшенную работу. Сбой между отчётом о последней эпохе и completion stamp восстанавливается без дополнительного шага оптимизации.

Тест с dropout и промежуточным использованием глобального RNG подтверждает побайтное равенство тензоров непрерывного и возобновлённого CPU-запуска. Изменение labels, конфигурации, кода, среды, checkpoint или validation score вызывает отказ. Гарантия не переносится автоматически на GPU/другие версии библиотек; CUDA-путь здесь не выполнялся.

В каждом job report: число параметров, committed train steps, history метрик, выбранная эпоха, wall time, process RSS high-water, CUDA allocated peak если применимо, validation duration и аналитическая оценка FLOPs. RSS относится ко всему процессу, а FLOPs — оценка с явно перечисленными исключениями. Held-out inference имеет отдельные фактические latency и token-replay checks.

CPU probe на 64 реальных обучающих SONAR histories измерил около `0.290/0.426/0.311` секунд на train step для concat/attention/GRU при двух потоках. Экстраполяция при всех 30 эпохах и 24 запусках — **21.4 часа только обучения**. Это оценка по короткому probe, без validation, checkpoint IO, test inference и causal evaluation. Early stopping может сократить бюджет. Наличие оплаченного GPU или фонового исполнителя не предполагается.

## Оценка и ограничения приёмки

[Held-out evaluator](l1-evaluation.md) публикует per-group/per-stratum ошибки структуры, три causal comparison, совместную TV, Clopper–Pearson и empirical Bernstein bounds, парные excess к concat и paraphrase consistency. Все пять seeds сохраняются. `numeric_oracle_gate` является числовой частью Oracle gate; status fixtures проверяются отдельно в preflight. Успех числовых условий требует ≥4/5 seeds для каждой ветки.

`test-open.json` связывает все финальные checkpoint до первого доступа оценщика к held-out labels. Сохранённые predictions и reports имеют отдельные хэши. Их повторное чтение проверяет frozen selection, weights, source/runtime, G0 evidence и input files. Это защита от случайных изменений, а не криптографическая подпись против владельца файлов.

В замороженной v0.1 не определена проекция «редкого контекста». `rare_contexts=not_established`: нулевая train-частота не выдаётся за вероятность генератора. Даже при прохождении числовых gates итог остаётся `incomplete_required_report`, пока этот обязательный раздел не определён в следующем явно версионированном протоколе. Если числовые gates провалены, итог `failed`; небольшой интеграционный стенд всегда `not_run_fixture`. `passed` не создаётся из факта завершения обучения.

GEO-1, active learning, обнаружение новых сущностей и идентификация произвольного SCM остаются вне этого L1. Зарегистрированное обучение и held-out качество SONAR пока не измерены. Проверенные 60 TTT trials и integration fixture публикуются отдельно.

Проверка реализации после независимого review: **237 tests passed**. Реальная integration fixture завершила 9 tuning и 15 final jobs; все 15 проверки повторного декодирования совпали. [Компактный отчёт](../../experiments/results/l1-pipeline-v0.1/smoke.json), [проверка повреждённых артефактов](../../experiments/results/l1-pipeline-v0.1/integrity-checks.json), [общая проверка](../../experiments/results/l1-pipeline-v0.1/verification.json).

## Воспроизводимая интеграционная проверка

```bash
PYTHONPATH=src python -m experiments.l1.smoke \
  --output /tmp/l1-smoke-new \
  --report /tmp/l1-smoke-report.json
python -m pytest -q
PYTHONPATH=src python -m experiments.ttt.runner validate
```

Smoke действительно обучает все 24 модели, но на 4 train / 4 validation histories, с четырёхмерными SHA256-признаками и одной эпохой. Он проверяет бюджет/переходы/стыковку вывода, не смысл SONAR embeddings и не обобщение. Даже идеальная fixture не может пройти L1.
