# L1 catalog: human review requested

Status: **pending**. Catalog SHA256: `4308558244476e29d2c887dc1b10a55af8243bad1c3dbea046101f42476609ce`.

No reviewer, review date, or approval has been supplied. This request does not satisfy the human-review gate.

The frozen catalog uses four families, six modes, and explicit/implicit time. Families 0–1 are training forms; 2–3 are held out. Review the rendered forms below and record a decision tied to the exact catalog hash.

Time and identity semantics: implicit time begins at slice 0. Executed or Forced creates the next physical transition and advances the clock by one; Observation, Unknown, Hypothetical, and Denied refer to the current slice and do not advance it. Explicit positive transition timestamps identify the resulting slice. Explicit observations can refer to slice 0 or a resulting slice. Reordered narration requires explicit timestamps. The rows below are isolated examples of each form, not a chronological multi-event history.

The first two messages introduce the two devices through C measurements at slice0: the first introduced name binds to device-0, the second to device-1. This introduction order is a benchmark convention. Names consistently identify devices within one record and have no permanent ID across records. Initial introductions are retained by every challenge transform. The model receives no names-to-ID sidecar, so binding requires remembering the initial messages. Names are independent of the user/speaker metadata. All messages use role user and speaker-0. R/M/C/Y and start/stop are literal controlled-language symbols. Observation asserts a measured binary value; Unknown adds no measurement; Executed performs a command; Forced sets a variable externally; Hypothetical only discusses a possible command; Denied refuses a newly proposed command, and does not cancel an earlier execution.

| Family | Mode | Time | Rendered form |
|---|---|---|---|
| 0 | Observation | explicit | На шаге 1 у устройства Альфа значение Y равно 1. |
| 0 | Observation | implicit | У устройства Альфа значение Y равно 1. |
| 0 | Unknown | explicit | В этой записи на шаге 1 новое измерение Y устройства Альфа не приведено. |
| 0 | Unknown | implicit | В этой записи новое измерение Y устройства Альфа не приведено. |
| 0 | Executed | explicit | На шаге 1 на устройстве Альфа выполнили команду start. |
| 0 | Executed | implicit | На устройстве Альфа выполнили команду start. |
| 0 | Forced | explicit | На шаге 1 принудительно установили Y устройства Альфа в 1. |
| 0 | Forced | implicit | Принудительно установили Y устройства Альфа в 1. |
| 0 | Hypothetical | explicit | На шаге 1 рассмотрели вариант выполнения start на устройстве Альфа. |
| 0 | Hypothetical | implicit | Рассмотрели вариант выполнения start на устройстве Альфа. |
| 0 | Denied | explicit | В этой записи на шаге 1 зафиксирован отказ от выполнения ещё одной команды start на устройстве Альфа. |
| 0 | Denied | implicit | В этой записи зафиксирован отказ от выполнения ещё одной команды start на устройстве Альфа. |
| 1 | Observation | explicit | Устройство Альфа, шаг 1: измеренное значение Y — 1. |
| 1 | Observation | implicit | Устройство Альфа: измеренное значение Y — 1. |
| 1 | Unknown | explicit | Устройство Альфа, шаг 1: эта запись не добавляет измерения Y. |
| 1 | Unknown | implicit | Устройство Альфа: эта запись не добавляет измерения Y. |
| 1 | Executed | explicit | Устройство Альфа, шаг 1: выполнена команда start. |
| 1 | Executed | implicit | Устройство Альфа: выполнена команда start. |
| 1 | Forced | explicit | Устройство Альфа, шаг 1: значение Y принудительно задано равным 1. |
| 1 | Forced | implicit | Устройство Альфа: значение Y принудительно задано равным 1. |
| 1 | Hypothetical | explicit | Устройство Альфа, шаг 1: обсуждалась возможность команды start. |
| 1 | Hypothetical | implicit | Устройство Альфа: обсуждалась возможность команды start. |
| 1 | Denied | explicit | Устройство Альфа, шаг 1: ещё одна предложенная команда start оставлена без выполнения. |
| 1 | Denied | implicit | Устройство Альфа: ещё одна предложенная команда start оставлена без выполнения. |
| 2 | Observation | explicit | Шаг 1. Прибор Альфа: наблюдаем Y=1. |
| 2 | Observation | implicit | Прибор Альфа: наблюдаем Y=1. |
| 2 | Unknown | explicit | Шаг 1. Для прибора Альфа в этой записи нет нового измерения Y. |
| 2 | Unknown | implicit | Для прибора Альфа в этой записи нет нового измерения Y. |
| 2 | Executed | explicit | Шаг 1. Прибору Альфа подали команду start. |
| 2 | Executed | implicit | Прибору Альфа подали команду start. |
| 2 | Forced | explicit | Шаг 1. У прибора Альфа принудительно зафиксировали Y=1. |
| 2 | Forced | implicit | У прибора Альфа принудительно зафиксировали Y=1. |
| 2 | Hypothetical | explicit | Шаг 1. Для прибора Альфа только рассмотрели возможность команды start. |
| 2 | Hypothetical | implicit | Для прибора Альфа только рассмотрели возможность команды start. |
| 2 | Denied | explicit | Шаг 1. Новую предложенную команду start прибору Альфа оставили без выполнения. |
| 2 | Denied | implicit | Новую предложенную команду start прибору Альфа оставили без выполнения. |
| 3 | Observation | explicit | В момент 1 измерение у Альфа показало Y=1. |
| 3 | Observation | implicit | Измерение у Альфа показало Y=1. |
| 3 | Unknown | explicit | В момент 1 эта запись о Альфа не содержит нового измерения Y. |
| 3 | Unknown | implicit | Эта запись о Альфа не содержит нового измерения Y. |
| 3 | Executed | explicit | В момент 1 для Альфа исполнили команду start. |
| 3 | Executed | implicit | Для Альфа исполнили команду start. |
| 3 | Forced | explicit | В момент 1 у Альфа установили Y=1 внешним воздействием. |
| 3 | Forced | implicit | У Альфа установили Y=1 внешним воздействием. |
| 3 | Hypothetical | explicit | В момент 1 для Альфа обсуждали гипотетическое выполнение start. |
| 3 | Hypothetical | implicit | Для Альфа обсуждали гипотетическое выполнение start. |
| 3 | Denied | explicit | В момент 1 от исполнения новой предложенной команды start для Альфа отказались. |
| 3 | Denied | implicit | От исполнения новой предложенной команды start для Альфа отказались. |

Reviewer response must include the catalog SHA256, reviewer identity, timestamp, approve/reject decision, and any corrections. A corrected catalog must receive a new hash-bound review.
