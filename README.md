# Behavioral Causal State

Исследовательский проект причинного рассуждающего движка: состояния относительно семейств predictive, interventional и counterfactual запросов; текстовый grounding; факторизованные SCM; проверяемые границы причинных ответов.

**Текущий статус: causal runtime и языковой training pipeline, 2026-10-04.** Реализованы точный do/CF inference, журнал, обучение Boolean core, семь видов текстовых пар и аудит splits. Добавлены verified SONAR adapter, concat/attention/GRU, autoregressive grounder и diagnostic training с input locks. Полные G2/P1 пройдены на 100 seeds. Реальный LearnLib TTT завершён: 60/60 прогонов с точным EQ. Для полного L1 реализован исполнитель 9 tuning + 15 final запусков; полный SONAR cache готов (56 000 histories), каталог одобрен владельцем для запуска; подготовлен локальный CUDA launcher с проверкой воспроизводимости. Зарегистрированные GPU-обучения ещё не выполнены.

[Запуск, результаты и ограничения runtime](docs/implementation/first-runtime.md).
[Обученное ядро, полный кандидатный датасет и oracle validation](docs/implementation/calibration-dataset.md).
[SONAR, языковые ветки и команды обучения](docs/implementation/language-runtime.md).
[Полный L1: запуск, возобновление, freeze и блокеры](docs/implementation/full-l1.md).
[TTT: результаты 60 реальных LearnLib прогонов](docs/implementation/ttt.md).
[Запуск полного L1 на своей GPU: данные, установка, resume](docs/implementation/local-gpu.md).

Первый языковой diagnostic выполнен на128 train /32 validation histories с реальными SONAR embeddings. Все три ветки обучены, но полностью правильных validation histories пока0/32. Это проверка интеграции с отрицательным результатом качества; зарегистрированный L1 остаётся `not_run`. Проверки реализации:156 tests passed.

Последующая [диагностика запоминания](docs/implementation/decoder-memorization.md) завершена: concat/attention/GRU восстановили8/8 обучающих histories; контроль перестановки входов подтверждает использование embeddings. Это train-only результат, без вывода об обобщении. На этом этапе проверка реализации:179 tests passed. Последующие проверки полного pipeline приведены в [новом отчёте](docs/implementation/full-l1.md).

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

[Что входит в первый запуск и что отложено](docs/specs/deferred.md). Формат ключа журнала полностью задан в [SPEC-02](docs/specs/02-contracts.md); геометрическая диагностика GEO-1 отложена. Аудитор проверяет публичные/evaluator файлы, calibration, splits и pair witnesses. Training loader проверяет split IDs и input locks; точный TTT теперь выполнен; зарегистрированное обучение L1 и его test metrics ещё не получены. Исторический dataset manifest сохраняет pending; отдельный receipt фиксирует одобрение владельца для нового GPU study. Ограниченная диагностика не получает его gates. G2/P1 имеют отдельные воспроизводимые run reports.

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
