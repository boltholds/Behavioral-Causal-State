# SPEC-02 — Архитектура и контракт ответа

## Границы модулей

Будущая Python-реализация разделяется по публичным Protocol-контрактам. Доменные состояния — immutable typed variants; конечные discriminators — enum. Generic attribute bags и nullable поля не кодируют состояния «неизвестно», «неприменимо» и «не завершено». Непроверенный JSON допустим только на внешней границе и преобразуется в доменные типы.

| Модуль | Публичный контракт | Выход / ответственность |
|---|---|---|
| simulator | TraceSource | Физическая трасса; evaluator-only U изолирован от learner |
| renderer | ObservationRenderer | Текст и приватные метки Γ; запрет утечки скрытой истины |
| representation | UtteranceEncoder, HistoryReader | Z и declared способ чтения истории |
| grounding | Grounder | Совместная интерпретация событий и её неопределённость |
| predictive | PredictiveLearner | TTT/PSR adapter и predictive ответы |
| causal | ModelClass, CausalQueryEvaluator | Допустимые SCM и семантика do/CF |
| search | CandidateProposer, FeasibilityVerifier | Кандидаты и отдельно проверка допустимости |
| bounds | BoundCertifier | Внешние границы, доказательства, остаточный gap |
| journal | AnswerJournal | Воспроизведение конкретного ранее выданного ответа |
| evaluation | ExperimentEvaluator | Метрики, gates, manifest; не используется learner |

Search не может выставлять ProvenIdentified или InconsistentClass только на основании поведения популяции. Bounds не может называть найденный диапазон сертифицированным внешним интервалом.

## Состояние движка

B_t хранит joint uncertainty о (Γ₁:ₜ, M, X_t, относящемся к запросам U), ссылки на архив свидетельств и snapshot. Представление может быть symbolic constraints, belief, posterior с явным prior либо проверенным набором кандидатов плюс сертификаты. Эти представления не считаются эквивалентными автоматически.

Кэш ответов — дополнительный журнал K→AnswerRecord, а не часть доказательства идентифицируемости. Исходные свидетельства для CF не удаляются только потому, что predictive state не изменилось.

## Запрос

Query — variant Predictive / Interventional / Counterfactual. Каждый содержит объект, величину ответа, горизонт, factual evidence reference; последние два — typed interventions с временным срезом и replacement semantics. CF содержит reference на заменяемое прошлое действие и политику сохранения остальных factual interventions.

v0.1 использует hard interventions. Soft/mechanism interventions предусмотрены в модели Query, но не реализуются как неявная интерпретация hard clamp.

Запросы о скалярных вероятностях имеют bounds [0,1]. Для полного распределения сохраняется совместное допустимое множество/модели-свидетели: независимые интервалы по ячейкам не являются произвольным совместно допустимым распределением.

## AnswerRecord

| Поле | Разрешённые варианты |
|---|---|
| identification | ProvenIdentified / ProvenNonIdentified / NotEstablished / NotApplicable |
| identification_scope | PopulationLaw / FiniteDataConstraints / DeclaredFiniteClass |
| computation | Complete / BudgetExhausted / NumericalFailure / NotStarted |
| feasibility | FeasibleWitnesses / CertifiedInfeasible / NotEstablished |
| result | PointFunctional / StatisticalEstimate / CertifiedBounds / WitnessRange / NoInformativeAnswer / Undefined |
| bounds_precision | Sharp / CertifiedWithinTolerance / OuterOnly / NotApplicable |
| grounding | Resolved / Ambiguous / Invalid |
| evidence | Проверяемое доказательство, свидетели или явный NoCertificate |

Обязательны snapshot_id, history_id, query_id, model_class_id, assumption_ids, numerical tolerance, verification scope. Неиспользуемые поля выражаются variants, а не смысловой перегрузкой null. Доменный enum не обязан совпадать с отображаемой пользователю строкой.

Статистический интервал, идентификационный диапазон, вычислительные внешние границы и диапазон найденных кандидатов имеют разные типы. Объединение допустимо только с явным происхождением каждого ограничения.

| Ситуация | Пользовательский смысл |
|---|---|
| ProvenIdentified + StatisticalEstimate | «Эффект идентифицируется при A; оценка и статистическая неопределённость …» |
| ProvenNonIdentified + Complete + Sharp | «Допустимые ответы от … до …; расчёт в том же классе и при тех же ограничениях их не сузит» |
| ProvenNonIdentified + BudgetExhausted + OuterOnly | «Единственного ответа нет; текущие внешние границы ещё могут сузиться» |
| NotEstablished + BudgetExhausted | «Проверка не завершена; однозначность ответа не установлена» |
| WitnessRange | «Найдены модели с ответами …; более крайние значения не исключены» |
| CertifiedInfeasible | «Данные и заданные ограничения несовместимы; ответ в этом классе не определён» |
| Invalid grounding / zero-support evidence | «Запрос не удалось определить в заданной интерпретации», с причиной |

Если нет доказанных нетривиальных bounds, допустим тривиальный интервал [0,1] с соответствующей пометкой. Кандидатный point estimate не повышается до ProvenIdentified. Plug-in SCM может вернуть точку условно на выбранной модели, с identification_scope=DeclaredFiniteClass; это не идентификация в более широком классе.

## Журнал воспроизводимости

K=hash(snapshot_id, exact_serialization(raw_messages, roles, order), structural_query_serialization). Используется однозначное length-prefixed/эквивалентное кодирование полей. Байты Unicode сохраняются; семантическая нормализация и threshold nearest-neighbor запрещены в этом журнале.

Snapshot включает данные/constraints, assumptions, interventions, классы моделей, encoder/tokenizer, grounder, core, алгоритм и бюджет вывода, dtype и версии схем. Replay возвращает именно сохранённый ответ. Продолжение незавершённого расчёта создаёт новую ревизию ответа с ссылкой на предыдущую, не переписывая историю.

Перефразировки не имеют общего ключа. Cross-paraphrase consistency измеряется с отключённым replay. Истинная Γ доступна только oracle/evaluator и не используется для ключа рабочего движка.

## Инварианты приёмки

- Два допустимых свидетеля с разными ответами исключают ProvenIdentified в том же scope.
- Пустая популяция не означает CertifiedInfeasible.
- Поиск с единственным кандидатом не означает Complete.
- Новые данные и предположения меняют snapshot и инвалидируют перенос статуса между snapshot.
- Воспроизведение дословного запроса в неизменном snapshot возвращает идентичный AnswerRecord.
- Ошибка negation или режима не маскируется causal identification status.
- Отказ/Undefined на заранее корректном benchmark-запросе считается ошибкой и учитывается в coverage; его нельзя исключить из TV-метрики молча.
