# SPEC-08 — Методы и первичные источники

Этот раздел фиксирует основания дизайна. Он не заявляет новизну соединения известных методов и не является полным обзором литературы.

## Сравнение представлений

| Метод | Что представляет | Какие ответы поддерживает | Чего сам по себе не обеспечивает |
|---|---|---|---|
| TTT | Минимальный deterministic residual I/O state | Будущее поведение по declared action alphabet; действия могут иметь do-семантику | Internal-variable SCM, cross-world coupling; точный EQ из конечного пассивного набора |
| PSR | Предсказания тестов как состояние | Stochastic controlled predictions при подходящих rank/observability assumptions | Причинную интерпретацию латентных координат и произвольные CF |
| POMDP | Belief над скрытым состоянием | Filtering, planning при transition/observation model | Идентификацию SCM из данных, сопряжение альтернативных миров |
| SCM | Structural functions, exogenous law, modular interventions | do и CF при заданной модели; анализ идентификации в классе | Автоматическое восстановление истинной структуры без assumptions |
| Causal world model | Learned transition/observation mechanisms с causal constraints | Планирование и симуляцию declared interventions | Само название не гарантирует causal identification или transfer |
| Causal representation learning | Факторы/абстракции из высокоразмерных наблюдений | Causal variables при явном supervision/identifiability setting | Уникальные causal factors из произвольных observational embeddings |
| Object-centric causal model | Объекты, типы и локальные отношения | Повторное использование динамики на объектах/композициях | Grounding идентичности, гарантированную инвариантность в новом домене |
| Causal discovery | Graph/CPDAG/PAG или scored hypotheses | Структурные ограничения при Markov/faithfulness/functional assumptions | Полные механизмы и CF coupling из одного графа |
| Do-calculus / ID | Правила преобразования causal queries | Идентификационные формулы в declared graph/data setting | Самостоятельное обнаружение графа или численную оценку по данным |
| Evolutionary population | Найденные проверенные модели | WitnessRange, proposals для экстремумов | Posterior, полноту класса, внешние bounds или доказательство пустоты |

## Чтение по задачам

1. **TTT.** Isberner, Howar, Steffen (2014), [The TTT Algorithm: A Redundancy-Free Approach to Active Automata Learning](https://link.springer.com/chapter/10.1007/978-3-319-11164-3_26). MAT, discrimination tree и counterexample refinement. [LearnLib active algorithms](https://learnlib.de/learnlib/maven-site/19.0.0/learnlib-build-parent/learnlib-algorithms-parent/learnlib-algorithms-active-parent/index.html) — первичная документация реализации; версию зависимости нужно закрепить при реализации.
2. **PSR.** Littman, Sutton, Singh, [Predictive Representations of State](https://papers.nips.cc/paper/1983-predictive-representations-of-state). Основание predictive state без обязательного явного latent-state ontology.
3. **POMDP.** Kaelbling, Littman, Cassandra, [Planning and Acting in Partially Observable Stochastic Domains](https://cs.brown.edu/research/pubs/techreports/reports/CS-96-08.html). Ссылка на авторский technical report 1996, предшествующий журнальной публикации 1998.
4. **SCM.** Pearl (2009), [Causal inference in statistics: An overview](https://escholarship.org/content/qt36w8n7pg/qt36w8n7pg.pdf). Structural interventions, confounding и counterfactual semantics.
5. **Identification.** Shpitser, Pearl (2006), [Identification of Conditional Interventional Distributions](https://arxiv.org/abs/1206.6876). Год исходной работы отличается от года размещения на arXiv.
6. **Counterfactual identification.** Shpitser, Pearl (2007), [What Counterfactuals Can Be Tested](https://arxiv.org/abs/1206.5294). Доступность interventional distributions не делает все CF идентифицированными.
7. **Конкретная шумовая модель.** Oberst, Sontag (2019), [Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models](https://proceedings.mlr.press/v97/oberst19a.html). Пример явного выбора structural coupling; не универсальное доказательство истинности такого coupling.
8. **Certified bounds.** Duarte, Finkelstein, Knox, Mummolo, Shpitser, [An Automated Approach to Causal Inference in Discrete Settings](https://arxiv.org/abs/2109.13471); [авторская журнальная версия](https://dcknox.github.io/files/DuarteEtAl_Autobounds.pdf). Polynomial programming, feasible witnesses, внешние bounds и вычислительный gap.
9. **Обзор discovery.** Peters, Janzing, Schölkopf (2017), [Elements of Causal Inference](https://mitpress.mit.edu/9780262037310/elements-of-causal-inference/). Markov/faithfulness, functional assumptions, ограничения observational discovery.
10. **Идентификация из наблюдений.** Shimizu et al. (2006), [A Linear Non-Gaussian Acyclic Model for Causal Discovery](https://jmlr.org/papers/v7/shimizu06a.html). LiNGAM: linearity, independent non-Gaussian errors, no hidden confounders.
11. **Causal representation learning.** Schölkopf et al. (2021), [Toward Causal Representation Learning](https://arxiv.org/abs/2102.11107). Карта assumptions и задач; embedding objective сам по себе не даёт causal variables.
12. **Intervention-supervised representations.** Lippe et al. (2022), [CITRIS: Causal Identifiability from Temporal Intervened Sequences](https://proceedings.mlr.press/v162/lippe22a.html). Важны временная структура и информация о вмешательствах.
13. **Causal abstraction.** Rubenstein et al. (2017), [Causal Consistency of Structural Equation Models](https://arxiv.org/abs/1707.00819). Согласование вмешательств между уровнями абстракции.
14. **Transport.** Pearl, Bareinboim (2014), [External Validity: From Do-Calculus to Transportability Across Populations](https://arxiv.org/abs/1503.01603). Перенос требует описания различий доменов.
15. **Objects.** Yu et al. (2024), [Learning Causal Dynamics Models in Object-Oriented Environments](https://proceedings.mlr.press/v235/yu24j.html). Направление object-level factorization для динамики и композиции.
16. **SONAR.** Duquenne, Schwenk, Sagot (2023), [SONAR: Sentence-Level Multimodal and Language-Agnostic Representations](https://arxiv.org/abs/2308.11466); [официальная реализация](https://github.com/facebookresearch/SONAR). Sentence representations и decoder; отсутствие causal loss не доказывает отсутствия извлекаемой информации.
17. **LaBSE.** Feng et al. (2022), [Language-agnostic BERT Sentence Embedding](https://aclanthology.org/2022.acl-long.62/). Альтернативный sentence encoder; его сравнение должно сохранять history-reader controls.
18. **Attention.** Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Sequence memory и encoder-decoder attention; фиксированная ширина токена не равна одному общему вектору истории.
19. **Статистический gate.** Maurer, Pontil (2009), [Empirical Bernstein Bounds and Sample Variance Penalization](https://arxiv.org/abs/0907.3740), Theorem 4. SPEC-05 использует выборочную дисперсию с делителем n−1 и масштабирование из [0,1] в [a,b].

## Ограничения выводов проекта

L1 не доказывает discovery, перенос на свободный язык или пользу факторизации. P1 не доказывает масштабируемость эволюции. Model-set consensus не заменяет identification theorem. Для graph recovery оценивается идентифицируемый класс, а не произвольная выбранная DAG-ориентация. При смене assumptions методы и их гарантии пересматриваются явно.
