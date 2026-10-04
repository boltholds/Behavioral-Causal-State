# Первый исполняемый runtime v0.1

Дата: 2026-10-04. Финальная проверка: 83 автоматических теста, успешная установка editable package и CLI smoke; 600 сохранённых G2 certificates независимо перепроверены из JSON. Это reference runtime для проверки причинной семантики и экспериментальных протоколов. Обучение языковых моделей и TTT пока не реализовано. Исходные protocol JSON остаются preregistration-артефактами; фактические статусы находятся в run reports.

## Запуск

Из корня репозитория, Python 3.12+:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m bcs demo
python -m bcs generate --count 20 --regime stochastic_partial --output /tmp/bcs-histories.jsonl
python -m bcs g2 --output /tmp/bcs-g2.json
python -m bcs p1 --output /tmp/bcs-p1.json
```

Для короткой проверки `g2`/`p1` допускают `--smoke`. Такой запуск всегда получает `not_run_full_protocol`, даже если все наблюдавшиеся результаты положительны. Из другого каталога перед subcommand передать `--repo-root /path/to/repository`.

`generate` экспортирует только group ID и текстовые сообщения. `--train-labels` явно добавляет видимые Γ для supervised train. Полная физическая трасса и U доступны лишь Python-объекту evaluator; в JSONL не экспортируются. Это базовый sampler, а не полный L1 split/challenge dataset. Denied/Unknown относятся к новой команде/новому измерению в конкретной записи и не отменяют прошлые события того же среза.

## Реализованные границы

| Модуль | Поведение |
|---|---|
| `contracts` | Immutable варианты ответов, scopes, statuses, свидетельств; проверка несовместимых комбинаций |
| `queries` | Строгий wire AST → typed predictive / next-slice hard-do / past-action CF |
| `journal` | AnswerJournal/v1, все 12 golden keys; SQLite exact replay; запрет перезаписи ответа в том же snapshot; проверка Payload при hash hit |
| `simulator` | Два устройства, общий clock, пять режимов; indexed joint noise, hard replacement до вычисления потомков |
| `inference` | Fraction-based twin-state filtering с factual evidence; отдельные zero-support и budget результаты |
| `generator` | Базовый sampler длин 4/8/12/16, шесть event variants, две контролируемые русские template families |
| `misspecification` | Exact interval feasibility + независимая проверка сертификата подстановкой/неравенствами; finite-data G2 |
| `population` | Четыре P1 search arms, полный архив экстремумов, отдельные аналитические bounds |
| `metrics`, `cli` | Joint TV, empirical Bernstein bound, manifest, CLI и отчёты |

Доменный `AnswerRecord` проверяет согласованность статусов, но сам по себе не доказывает истинность приложенного сертификата. Сертификат присваивает проверивший его solver. Inference сейчас возвращает распределение, условное на одной явно переданной SCM; он не делает discovery и не доказывает идентификацию в неизвестном классе.

## Семантика точного вывода

`History` содержит полную последовательность выполненных переходов и разреженные видимые наблюдения. `evaluate` не получает simulator U. Для CF фильтруется joint factual/alternative state при одинаковом шуме каждого среза. Поздние factual observations отбрасывают несовместимый шум, но не устанавливаются в альтернативном мире. Следующие factual действия/clamps сохраняются. Будущие шумы суммируются по закону SCM.

Начальное R=0, C~Bernoulli(1/2), начальные M/Y следуют механизму. Истории SPEC-04 сообщают начальный C обоих устройств, поэтому likelihood заданной C-зависимой logging policy после conditioning сокращается. Для произвольных confounded action logs этот evaluator непригоден без явной модели policy. API трактует переданный action schedule как фиксированный экспериментальный контекст.

Пример `demo`: при C₀=0, выполненном start и наблюдаемом Y₁=1 замена прошлого start на stop даёт P(Y₁^cf=1)=0. Новое do(R=0) на следующем срезе даёт 9/50. Assumptions о XOR-форме, независимости, p=1/10 и времени указаны в ответе.

Indexed noise использует детерминированный SHA-256 driver для вектора (Nᴹ,Nʸ) с индексом seed/episode/object/time. Закон вектора задаёт независимость или shared noise. Это предотвращает изменение U из-за другого порядка вызовов или ветвления replay. Это генератор симуляции, не вероятностное утверждение о криптографической функции.

## Wire query-v1 в этом runtime

Общие поля: `schema_version`, `kind`, `object_id` (`device-0`/`device-1`), непустой массив различных `outcome`, integer `horizon`, непустой `evidence_ref`.

- Predictive: `future_actions`, длина равна horizon; второй объект выполняет hold.
- Interventional: horizon=1, `interventions=[{object_id,variable,value},...]`; обычные действия hold, hard replacement следующего среза.
- Counterfactual: horizon=0 означает результат в конце factual history; `replaced_time` ≥1, `replacement` start/stop, `continuation_policy="preserve_factual_actions"`. Существование заменяемой команды проверяет evaluator.

Журнал отвергает неизвестные поля и неподдерживаемые AST. `evidence_ref` связывает запрос с историей на вызывающей стороне; snapshot обязан идентифицировать все её данные и assumptions. Ключ не гарантирует семантическую эквивалентность перефразировок. Новая версия данных/модели/бюджета требует нового snapshot.

## Результаты

Полные [G2](../../experiments/results/runtime-v0.1/g2.json) и [P1](../../experiments/results/runtime-v0.1/p1.json) содержат все seed, параметры, версии Python/NumPy/SciPy, SHA-256 source/protocol и wall time.

G2: 100 seeds, 10 000 независимых эпизодов на каждую из 12 клеток, всего 12 000 000. Наблюдаем только Y при известных do(R,C), естественное M скрыто.

| Истинный режим | Отклонений H_ind | Отклонений H_joint |
|---|---:|---:|
| independent_xor | 0/100 | 0/100 |
| shared_xor | 100/100 | 0/100 |
| wrong_gate | 100/100 | 100/100 |

Exact matrix, budget fault и неразличимые M₋/M₊ проверены отдельно. Это результат в объявленном XOR-семействе; он не доказывает обнаружение произвольной ошибки модели.

P1: каждый из fit_only, novelty, directed, uniform достиг q_min≤0.01 и q_max≥0.99 в 100/100 seeds. На run — 6 464 кандидата, включая повторы. Для всех arms сохраняется all-time archive, а не только последнее поколение. Результат поиска типизирован WitnessRange; [0,1] сертифицируется отдельно через response-type constraints. Преимущество novelty этим запуском не установлено.

SciPy предлагает начальные Clopper–Pearson endpoints. Runtime расширяет их и проверяет определяющие binomial-tail inequalities через 60-значную Decimal interval arithmetic с направленным округлением; exact α для G2 = 1/240 на клетку. Без успешной проверки возвращается NotEstablished/NumericalFailure. Feasibility после получения interval endpoints вычисляется точно над rational numbers. Run reports сохраняют joint-noise witnesses и obstruction certificates. Statistical rejection не означает абсолютное доказательство неверности мира.

P1 acceptance также требует прохождения дополнительных controls. Finite-data control использует фиксированный seed 20261004, по 1 000 binomial observations для двух marginals и одновременные 95% CP constraints; sampling intervals и coupling bounds записаны раздельно.

## Что ещё не завершено

- G0 в полном смысле SPEC-05: numeric simulator tests выполнены, реальный TTT/MAT трек не запущен.
- G1: отсутствуют полный challenge generator, locked templates, split manifest и зарегистрированный 20k/4k/4k dataset.
- L1: не обучены truth tables общего frozen core, grounder и четыре reader branches; SONAR checkpoint не загружен.
- C1 как полный stochastic/partial learning track: reference inference и G2 существуют, обучение класса SCM не реализовано.
- Генератор пока поддерживает контролируемый русский язык; свободный текст, entity discovery, active causal design и transfer остаются в roadmap.

Тесты ядра и положительные G2/P1 не заменяют эти gates. Следующий этап: calibration learner + полный minimal-pair/split generator, затем lock manifest и L1 обучение.


## Решения реализации и ревью

План выполнен как первый runtime milestone. Все четыре существенных замечания независимого ревью исправлены; сохранение проверяемых G2-сертификатов также включено в этот коммит. Решения:

- Main выбран по прямому разрешению пользователя; внешние файлы репозитория сохраняются.
- Для произвольной скрытой logging policy требуется отдельная модель. Текущий API условен на фиксированном schedule; выводы нельзя переносить на confounded logs автоматически.
- Denied/Unknown имеют область конкретной записи; изменены тексты, распределение sampler сохранено. Полный human-reviewed template lock ещё впереди.
- Первоначальный one-ULP подход заменён проверяемыми Decimal intervals; недоказанная численная точность приводит к отказу от сертификата.
- Python module CLI поддержан независимо от shell PATH. Установка консольного entrypoint тоже проверена.
- Обучение нейронных моделей, полный G1 и LearnLib TTT остаются следующим milestone; текущий коммит не объявляет весь MVP завершённым.
