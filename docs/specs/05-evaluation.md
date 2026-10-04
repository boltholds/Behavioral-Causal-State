# SPEC-05 — Протокол экспериментов и gates

Машиночитаемые настройки: [language-v0.1.json](../../experiments/protocols/language-v0.1.json). Все пороги — инженерные критерии первого запуска, а не опубликованные результаты или общие теоремы.

## Предварительные проверки

G0: simulator, query evaluator и аналитические fixtures совпадают с точностью 10⁻¹²; δ-вырожденные распределения нормированы; значения в [0,1]. Deterministic TTT проверяется на полном конечном product automaton через точный EQ. TTT test не добавляет скрытые данные в L1.

G1: train/validation/test групповые пересечения и semantic duplicates отсутствуют; ни один test label/U не передан learner; каждая pair annotation подтверждена simulator. Невыполнение G0/G1 блокирует интерпретацию дальнейших метрик.

G2-MISSPEC: отдельная проверка реакции движка на нарушение assumptions, описанная в [SPEC-09](09-misspecification.md) и [misspecification-v0.1.json](../../experiments/protocols/misspecification-v0.1.json). Она не включена в G0/G1. G2 обязателен перед заявлением поддержки misspecification и приёмкой C1; первый ограниченный L1 можно публиковать отдельно, явно указав G2=not_run. Отсутствие выполненного G2 запрещает общий claim «движок обнаруживает неверную модель».

Первый runtime содержит simulator/inference, базовый generator и G2/P1 evaluator. Полные G2/P1 запущены; G0 с TTT и G1/L1 ещё не завершены. [Результаты и границы реализации](../implementation/first-runtime.md).

## Общие условия L1

Один deterministic режим, известный граф, выученные Boolean truth tables. Core обучается один раз на oracle train/calibration data и замораживается. Один и тот же artifact core используется всеми языковыми ветками во всех seed. Ошибки stochastic/latent core затем исследуются отдельным C1-треком; отсутствие корректного core не скрывается сравнением только с его ошибочным oracle output.

SONAR frozen. Три learning rates {0.0001,0.0003,0.001}, AdamW, weight_decay=0.01, batch=64, до 30 эпох; ранняя остановка после 5 эпох без улучшения validation free-running full-event error. Ties: validation NLL, затем меньший learning rate. Один tuning seed=0, максимум три конфигурации на ветку. Финальные seeds={11,23,37,51,79}; test открывается только после фиксации выбранной конфигурации каждой ветки. Dropout=0.1, clip global grad norm=1.0. Architecture sizes заданы в SPEC-03.

Публикуются параметры, train steps, объём данных, FLOPs/оценка, peak memory, latency и dtype. Равный search budget не называется равной мощностью моделей. Все baseline получают одинаковые данные и semantics.

## Распределения запросов

### μ_lang

Для каждой истории вычисляются четыре категории с равными весами; усреднение по объектам внутри них задано ниже:

1. Predictive: один будущий hold.
2. Interventional: один будущий hold + do(C_next=0).
3. Interventional: один будущий hold + do(C_next=1).
4. Counterfactual: заменить выполненную в прошлом start/stop на противоположную; оценить выход в конце factual horizon. Действие выбирается равномерно среди допустимых выполненных команд; целевой объект этой CF-категории — объект заменённой команды.

Точнее, категорию 4 усредняют по всем выполненным start/stop в истории; категории 1–3 — по двум объектам. Это исключает различие Monte Carlo query sampling между ветками. Обязательное действие генератора гарантирует непустоту категории 4. Ошибки усредняются сначала внутри истории, затем равномерно по историям; длинные истории не получают больший вес.

Полный режим: ответ — совместное распределение (M,Y) целевого устройства в заданном срезе. Partial: основная метрика по Y. Скрытое M не используется для основного partial score.

### Обоснование весов и sensitivity report

Вес 1/4 — первоначальный benchmark design: равный вес четырёх заранее выбранных категорий и отсутствие подгонки под результат. Это не оценка частот будущих пользовательских запросов и не утверждение равной практической ценности операций. В таком распределении predictive имеет суммарный вес 0.25, interventional — 0.50, counterfactual — 0.25.

Отчёт MUST показывать вектор ошибок по четырём категориям до агрегирования. Дополнительно на тех же ответах, без переобучения и изменения primary gates, рассчитываются два заранее заданных sensitivity scores:

| Scheme | hold | do(C=0) | do(C=1) | past CF |
|---|---:|---:|---:|---:|
| primary | 0.25 | 0.25 | 0.25 | 0.25 |
| intervention_priority | 0.10 | 0.35 | 0.35 | 0.20 |
| counterfactual_priority | 0.10 | 0.20 | 0.20 | 0.50 |

Sensitivity scores описательные и не выбираются после просмотра test для объявления победителя. Если ranking меняется, это часть результата. Для целевого продукта веса потребуют отдельного workload specification. Низкая общая ошибка не заменяет указания ошибок каждой причинной операции.

### μ_joint

Отдельный core-generalization panel: пары целей (R,M), (R,C), (M,C), для каждой значения 00,01,10,11 — всего 12 hard joint interventions следующего среза при hold. Равномерный вес истории, объекта и каждой комбинации. Train/calibration содержит только single-target interventions; joint targets отсутствуют как выполненные training interventions.

Эти вмешательства часто стирают зависимость от истории. μ_joint не заменяет μ_lang и causal-sensitive пары. Редкость определяется вероятностью контекста по training generator <0.01, а не нулевой train-частотой withheld joint action. Частоты/вероятности и отдельные rare scores публикуются; relative 30% improvement не используется как gate L1.

## Метрики

TV(p,p*)=0.5∑_y|p(y)−p*(y)|. Нельзя заменять совместную TV суммой ошибок отдельных маргиналов без нового протокола.

E_b=E_μlang TV(P_b,P_oracle), E_core=E_μlang TV(P_oracle,P_simulator). Отдельно измеряется E_true для каждой ветки. При invalid output/отказе на корректном point-query TV contribution=1; coverage публикуется. Set-valued ответы оцениваются через coverage/ширину/статус отдельно и не притворяются point-distribution.

Critical pair error=1, если хотя бы в одной стороне пары неверно восстановлено проверяемое поле или mode/negation/time/entity контекст, необходимый для его интерпретации. Общий scalar accuracy не заменяет отчёт по каждому фактору и challenge type. Для paraphrase/time-invariance сравнивают эквивалентные нормализованные события с учётом event timestamps, а не их порядок в рассказе.

Cross-paraphrase: доля несовпадающих identification/result-kind статусов; средняя TV на корректных point-ответах с invalid penalty=1. Journal replay отключён.

## Статистическая процедура

Единица независимого наблюдения — base history/group. Все queries, pair sides, paraphrases и derived traces внутри группы сначала агрегируются. Seeds обучения не считаются дополнительными независимыми test examples.

Для критических ошибок: односторонний Clopper–Pearson upper bound отдельно для каждой из двух holdout strata каждой категории; α=0.05/14 на 7×2 сравнений внутри одного seed. Обе страты обязаны пройти порог. Это избегает предположения одинаковой вероятности ошибки в фиксированной смеси 1 000 новых templates + 1 000 новых names/compositions. Per-field описательные интервалы помечаются отдельно.

Для средних bounded TV и paired excess используется односторонняя empirical Bernstein upper bound. Для n≥2 независимых group-значений X∈[a,b], выборочного variance s²:

U=mean(X)+sqrt(2s² log(2/α)/n)+7(b−a)log(2/α)/(3(n−1)), α=0.05.

Для TV: [a,b]=[0,1]; для excess E_b−E_concat: [-1,1]. Это per-comparison bound; совместный 95% family-wise claim для всех архитектур не заявляется. При невозможности iid base groups gate помечается invalid, а не заменяется bootstrap по отдельным сообщениям. Источник формулы — [Maurer–Pontil](08-literature.md).

Для фиксированных challenge strata mean bounds вычисляются отдельно по стратам, а не при предположении iid для их объединения. μ_lang IID panel использует независимо выбранные длины по declared distribution; искусственные точные квоты по длинам не подставляются в iid формулу. Для paraphrase status disagreement применяется тот же binomial upper bound внутри каждой страты.

### Gates первого запуска

| Gate | Требование |
|---|---|
| Oracle | Upper(E_core)≤0.02 на μ_lang и отдельно μ_joint; точные status fixtures пройдены |
| Каждая языковая ветка | Upper(E_b)≤0.02 и Upper critical error≤0.01 в каждой challenge category |
| Attention/GRU noninferiority | В дополнение к абсолютным gates: Upper(E_b−E_concat)≤0.005 |
| Paraphrase consistency | Upper disagreement≤0.01; Upper mean TV≤0.01 |
| Reproducibility | Идентичный AnswerRecord при exact replay неизменного snapshot |

Для каждой ветки gates должны пройти минимум в 4 из 5 финальных seeds; все пять публикуются. Это правило устойчивости к seed, не отдельная 95%-ная гарантия по распределению обучения. Провал не исправляется дополнительными seed после открытия test.

### Интерпретация

Concat pass показывает практическую извлекаемость на этом стенде. Concat pass / GRU fail означает провал выбранной конфигурации GRU, не доказательство необратимой информационной потери. Attention pass / concat fail допустим: inductive bias/оптимизация различаются. Все fail при oracle pass означает, что достаточность embeddings не показана; token-level control — следующая версия. Oracle fail блокирует causal interpretation, но не публикацию структурных ошибок.

## C1: шум, CF и несовместимость

Нормативный протокол вынесен в [G2-MISSPEC](09-misspecification.md): три режима, два проверяемых класса, четыре do-клетки, точный feasibility criterion, negative controls и поведение при исчерпании бюджета. Сохраняются 10 000 episodes на клетку, 100 seeds, не менее 95/100 отклонений для каждого заданного несовместимого сочетания и не более 10/100 ложных отклонений для каждого совместимого сочетания.

M₊/M₋ проверяет отдельно неразличимое по доступным данным нарушение coupling-предположения. Отсутствие goodness-of-fit отклонения не доказывает корректность CF coupling. Условность ответа указывается с начала расчёта.

Калиброванные интервалы для G2 не означают «p-value идентификационных bounds». Статистические p-value тесты конкретных causal null hypotheses и uncertainty самих оценённых границ отложены согласно [deferred.md](deferred.md).

## Завершение и отчёт

После трёх tuning trials и пяти финальных seeds L1 закрывается как passed/failed/invalid по каждому gate. Без автоматического продления. Следующая версия требует нового test и записи причин изменения. Дополнительный RQ-F с TTT, active design и OOD mechanisms в этот итог не включается.

Report MUST содержать manifest, параметры и бюджеты, assumptions, доступные переменные, все seeds, group counts, ошибки по полям, joint TV, statuses, rare/holdout strata, wall time/memory, нарушения протокола и пределы выводов.
