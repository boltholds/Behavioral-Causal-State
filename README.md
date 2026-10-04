# Behavioral Causal State

Исследовательский проект причинного рассуждающего движка: состояния относительно семейств predictive, interventional и counterfactual запросов; текстовый grounding; факторизованные SCM; проверяемые границы причинных ответов.

**Текущий статус: runtime и подготовка языкового эксперимента, 2026-10-04.** Реализованы точный симулятор и do/CF inference, журнал, обучение Boolean core по видимым calibration probes, семь видов текстовых пар, сборка и аудит splits. Полные G2/P1 пройдены на 100 seeds. Языковое обучение L1 и реальный TTT остаются следующим этапом.

[Запуск, результаты и ограничения runtime](docs/implementation/first-runtime.md).
[Обученное ядро, полный кандидатный датасет и oracle validation](docs/implementation/calibration-dataset.md).

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python -m bcs demo
python -m bcs prepare-l1 --smoke --output /tmp/bcs-l1-smoke
```

## Начать здесь

1. [Область проекта и исследовательские вопросы](docs/specs/00-scope.md).
2. [Формализация причинного состояния](docs/specs/01-causal-state.md).
3. [Архитектура, контракты и статусы ответов](docs/specs/02-contracts.md).
4. [Языковой pipeline и четыре ветки сравнения](docs/specs/03-language.md).
5. [Генератор трасс и текстовых пар](docs/specs/04-generator.md).
6. [Протокол MVP, метрики и критерии остановки](docs/specs/05-evaluation.md).
7. [Поиск популяции SCM и сертификация границ](docs/specs/06-population.md).
8. [Roadmap снятия предположений](docs/specs/07-roadmap.md).
9. [Литература и сравнение методов](docs/specs/08-literature.md).
10. [G2-MISSPEC: обнаружение нарушения предположений](docs/specs/09-misspecification.md).

[Что входит в первый запуск и что отложено](docs/specs/deferred.md). Формат ключа журнала полностью задан в [SPEC-02](docs/specs/02-contracts.md); геометрическая диагностика GEO-1 отложена. Аудитор проверяет реальные публичные/evaluator файлы, calibration, splits и pair witnesses. Полный G0 (с TTT), G1 с реальным training loader и L1 ещё не завершены. Каталог имеет статус human review pending; `training_ready=false`. G2/P1 имеют отдельные воспроизводимые run reports.

Машиночитаемые параметры: [language-v0.1.json](experiments/protocols/language-v0.1.json), [population-v0.1.json](experiments/protocols/population-v0.1.json), [misspecification-v0.1.json](experiments/protocols/misspecification-v0.1.json). Аналитические примеры: [causal-oracles-v0.1.json](experiments/fixtures/causal-oracles-v0.1.json). Побайтные примеры ключей: [journal-key-v1.json](experiments/fixtures/journal-key-v1.json).

## Первый результат

Четыре ветки — oracle grounding, SONAR concat, SONAR attention, SONAR GRU — используют одно и то же обученное и затем замороженное причинное ядро. Проверяются восстановление структуры событий, ответы на причинные запросы и устойчивость к перефразированию. Concat — baseline плотного чтения фиксированного окна; он не задаёт верхнюю или нижнюю границу качества остальных архитектур.

Отдельный небольшой эксперимент проверяет поиск альтернативных SCM на контрфактуальном запросе с точными границами `[0, 1]`. Экстремумы найденной популяции всегда маркируются как найденный диапазон; полнота требует отдельного доказательства.

TTT служит predictive baseline в детерминированном треке. Сравнение затрат активного обучения и преимуществ факторизации описано в roadmap и не входит в первый языковой запуск.

## Правила чтения спецификаций

- MUST — обязательный контракт; SHOULD — рекомендуемый способ реализации.
- Принятое определение, выбранный инженерный параметр и исследовательская гипотеза обозначаются раздельно.
- Изменение порогов, класса моделей или test-распределения после открытия test создаёт новую версию протокола.
- Реализованное поведение и оставшаяся работа перечислены в [отчёте runtime](docs/implementation/first-runtime.md) и [отчёте calibration/dataset](docs/implementation/calibration-dataset.md); спецификация описывает также будущие модули.

Лицензия репозитория: [LICENSE](LICENSE).
