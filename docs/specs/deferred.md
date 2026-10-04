# Граница первого запуска и отложенные задачи

Статус документа: scope v0.1. Основные числовые пороги и seeds L1/P1 сохранены. Журнал, reference inference и calibration learner реализованы, полные G2/P1 запущены; [runtime](../implementation/first-runtime.md), [calibration/dataset](../implementation/calibration-dataset.md). Held-out артефакты проверяет evaluator, языковые test metrics ещё не вычислялись и не использовались для выбора модели.

## Требования треков и текущее выполнение

| Компонент | Контракт | Что требуется до заявления о прохождении |
|---|---|---|
| Журнал воспроизводимости | Побайтный AnswerJournal/v1 и hash fixtures в [SPEC-02](02-contracts.md) | Выполнены: runtime round-trip, exact replay, разделение snapshots |
| Генератор | Физика, event plan, renderer и minimal pairs в [SPEC-04](04-generator.md) | Реализованы generator, candidate catalog и проверка реальных dataset artifacts; human review каталога pending |
| G0/G1 | [SPEC-05](05-evaluation.md) | Artifact audit реализован; остаются TTT/EQ и проверка доступа реального training loader |
| L1 | Четыре reader branches, общее frozen core, primary metrics | Locked manifest, реальные checkpoint hashes, train/test и все seed |
| G2-MISSPEC | [SPEC-09](09-misspecification.md) | Выполнены в reference runtime: exact/finite-data, certificates, false rejection, budget fault |
| P1 | [SPEC-06](06-population.md) | Выполнены четыре search arms ×100 seeds; аналитический solver остаётся отдельным |

Эти пункты не называются deferred только потому, что пока не написан код. Они входят в соответствующие declared tracks. L1 completion не означает completion C1/G2/P1.

## Deferred

| ID | Задача | Статус первого L1 | Условие включения |
|---|---|---|---|
| GEO-1 | Конечная геометрическая кривая δ̂(L) | deferred, не обязательный отчёт и не gate | Зафиксировать C_eval, Q_eval, pair IDs, нормализацию, alignment разных длин и общий admissible domain |
| GEO-GLOBAL | Глобальная Lipschitz-гарантия | Не заявляется | Отдельные математические assumptions и доказательство; finite pairs недостаточно |
| ACTIVE-1 | Активный выбор причинных экспериментов/discovery | Вне fixed-data L1/P1 | Равный action budget, доступ к oracle и контролируемые intervention semantics; отдельный manifest |
| FACTOR-1 | Sample/memory преимущество факторизации относительно TTT | Не проверяется L1 | Общий predictive/intervention интерфейс, scaling fixtures и budgets |
| STATS-BOUNDS | Sampling uncertainty оценённых identification bounds и p-value для конкретного causal null | Вне L1; конечные confidence constraints P1/C1 сами по себе эту задачу не решают | Назвать null, estimand, sampling design, dependence и correction за выбор query/модели |
| GROUND-MARGINAL | Полная маргинализация неоднозначного Γ | Вне greedy L1 | Определить joint interpretation space и учёт отброшенной probability mass |
| LANGUAGE-2 | LaBSE, token-level controls, свободный язык, NLP parsing q | Вне первого сравнения | Отдельный контролируемый reader/data protocol |
| DISCOVERY-1 | Неизвестные переменные/граф, перенос и длинные истории | Вне первого L1 с известным графом | Соответствующие gates [roadmap](07-roadmap.md) |

TTT по природе является active learner: ACTIVE-1 не запрещает MQ/EQ в отдельном T0 sanity check. Он исключает смешивание исследования causal experimental design с первым языковым benchmark.

P-value не является вероятностью корректности SCM, полноты популяции или истинности границ. G2 уже использует одновременные confidence constraints с declared α; отдельного «p-value для bounds» для его выполнения не требуется.

## До первого запуска

Реализация MUST заполнить run manifest, сгенерировать данные, подтвердить G0/G1 и записать статусы всех треков как passed/failed/invalid/not_run. Deferred не кодируется нулевой ошибкой. Изменения assumptions, первичных весов или порогов после открытия test требуют новой версии протокола и нового закрытого test.
