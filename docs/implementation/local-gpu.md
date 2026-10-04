# Полный L1 на локальной NVIDIA GPU

Целевая среда: Ubuntu 24.04 / WSL2, Python 3.12, NVIDIA GPU, OpenJDK 17. Команды выполняются в Linux-терминале. Native Windows не поддержан этим launcher: executor использует `fcntl` и `resource`.

В архиве `BCS-L1-GPU-data.zip` находятся исходные 56 000 histories, evaluator sidecars, замороженное Boolean core и четыре готовых SONAR cache. Публичные embeddings извлечены без обучения на held-out данных. Весов SONAR и старого CPU study в архиве нет. Новая GPU study создаётся локально: пути, runtime и GPU фиксируются при `init`.

Одобрение каталога из 48 форм зафиксировано отдельным [receipt](../../experiments/reviews/l1-catalog-approval-2026-10-04.json) по ответу владельца на запрос одобрения. Это принятие каталога для запуска, без заявления о независимом лингвистическом аудите. Исходный dataset manifest с историческим `pending` не изменяется.

## Установка

Сначала убедиться, что `nvidia-smi` работает внутри WSL2. Репозиторий и study лучше держать в Linux home, например `~/projects`, для скорости файловых операций. Потребуется место под CUDA environment и checkpoints; планировать не менее 10 GB свободного диска. Не менять библиотеки и checkout во время study.

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv openjdk-17-jdk git unzip
git clone https://github.com/boltholds/Behavioral-Causal-State.git
cd Behavioral-Causal-State
# Для точного снимка: git checkout --detach <commit из сообщения с архивом>
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[test]'
python -m pip check
nvidia-smi
java -version
```

`java -version` должен начинаться с `openjdk version "17.`. При нескольких Java выбрать установленный OpenJDK 17. TTT preflight использует компилятор из JDK, скачивает две закреплённые Java dependencies из официальных URLs и проверяет их хэши.

Torch 2.8.0 / CUDA 12.6 — опубликованная комбинация [официальных wheels](https://pytorch.org/get-started/previous-versions/). Для WSL2 нужен рабочий драйвер NVIDIA на Windows; [инструкция Microsoft](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl). Установка пакетов выше не устанавливает системный драйвер.

## Данные и запуск

Скачать архив из сообщения ассистента, перенести в checkout и распаковать **из корня репозитория**:

```bash
unzip BCS-L1-GPU-data.zip
sha256sum --check --strict l1-gpu-inputs.sha256
bash experiments/l1/run_gpu.sh
```

Launcher проверяет чистоту tracked кода и checksums архива, выполняет синтетическую CUDA-проверку, затем `init → 9 tuning → freeze LR → 15 final → freeze checkpoints → held-out evaluation`. Синтетическая проверка каждой ветки делает backward/AdamW с batch 64, 16 сообщениями, 1024 признаками и 97 target tokens; проверяет повтор seeded optimizer step и greedy decoding. Она не читает train/test и не измеряет качество SONAR. При ошибке или OOM обучение не начинается; прислать лог, не менять самостоятельно batch или отключать проверки.

Бюджет v0.1: concat/attention/GRU, три LR, пять final seeds, до 30 эпох, batch 64, patience 5, float32. Launcher задаёт `CUBLAS_WORKSPACE_CONFIG=:4096:8`; runtime включает deterministic algorithms и выключает TF32, AMP не используется. Эти настройки соответствуют [рекомендациям PyTorch по воспроизводимости](https://docs.pytorch.org/docs/2.8/notes/randomness.html). Побайтное равенство CPU/GPU и разных версий/устройств не обещается. Смена GPU или numeric runtime отвергается при resume.

Процесс работает в открытом терминале. Для долгой сессии можно запускать внутри `tmux`. Повторный вызов продолжает с последней завершённой эпохи:

```bash
source .venv/bin/activate
bash experiments/l1/run_gpu.sh
```

Незавершённая эпоха повторяется. Старый CPU study не переносить и его lock не редактировать. `git pull` и обновление Python packages до завершения запрещены проверкой неизменности среды; выполнять их следует после study.

## Что прислать для разбора

Логи: `artifacts/l1-gpu-logs/`. Состояние, selected LR, отчёты, predictions и checkpoints: `artifacts/l1-study-gpu-v0.1/`. Для компактной отправки отчётов без весов и predictions:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
with ZipFile('l1-gpu-reports.zip', 'w', ZIP_DEFLATED) as archive:
    for root in (Path('artifacts/l1-study-gpu-v0.1'), Path('artifacts/l1-gpu-logs')):
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.log'):
                archive.write(path, str(path))
print('l1-gpu-reports.zip')
PY
```

Первым полезен `cuda-preflight.json` и несколько строк epoch log. После завершения — весь `l1-gpu-reports.zip`.

## Граница текущей проверки

В среде подготовки доступен только CPU. Проверены 244 tests на CPU, синтаксис launcher и CRC/SHA256 архива; CUDA preflight впервые исполняется у владельца GPU. Оценки времени GPU нет. Полный L1 ещё не измерен. Реальный TTT уже прошёл 60/60 trials; локальный `init` перепроверяет его сохранённые свидетельства.

Завершение 24 обучений не равно успеху L1: evaluator применяет замороженные gates. `rare_contexts=not_established` в v0.1 остаётся ограничением отчёта; даже при числовом успехе итог будет `incomplete_required_report`, пока следующий версионированный протокол не определит эту панель. Порогов и test-распределения этот GPU handoff не меняет.

Архив: 80 970 461 bytes, SHA256 `ed319da913f962cbf7a4e841c64da0e6e8d78065d00a55c52bc47f8902b03a45`. [Файловый манифест](../../experiments/results/l1-gpu-handoff-v0.1/transfer.json), [проверка реализации](../../experiments/results/l1-gpu-handoff-v0.1/verification.json).
