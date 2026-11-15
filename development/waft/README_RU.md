# Разработка WAFT с нуля

Это исходники для разработки. Для использования готовой ноды достаточно
`plugins/Cattery/WAFT/` в корне Kyven; см. [руководство WAFT](../../docs/WAFT.md).

## Что сохранено

- `WAFT-source/` — исходники WAFT и вложенные зависимости с локальными правками.
- `RAFT/` — Cattery-шаблон: gizmo, обвязка, метаданные и иконки; RAFT.cat для сборки WAFT не нужен.
- `tools/` — загрузка checkpoint, экспорт, упаковка, диагностика и проверки.
- `WAFT/` — лёгкая оболочка собираемого пакета, без дубликатов моделей.
- `reports/` — отчёты и логи прежней сборки; они не подтверждают новую сборку.
- `provenance.json` — URL и commit исходников, состояние submodule, SHA-256 checkpoint.
- `upstream-local.patch` — локальные правки относительно исходного commit; в исходниках уже применены.
- `requirements-lock.txt` — список версий исходной среды Python 3.10.20.

История вложенного Git убрана из этой копии; исходный commit записан в provenance.
Submodule dinov3 не был инициализирован в исходной среде и не нужен для экспорта WAFT-a1.
Обучающие датасеты и пользовательский видеоклип сюда не входят.

## 1. Создать среду

Нужны Windows, установленный Python 3.10 с Python Launcher (`py`) и NukeX 16
с действующей лицензией для создания `.cat`. FP16-экспорт и проверки CUDA требуют
NVIDIA GPU. Откройте PowerShell в `development/waft`:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r tools\requirements-export.txt
```

Файл зависимостей содержит проверенные основные версии и индекс CUDA 11.8.
Для точного повторения полного набора установленных пакетов вместо второй команды:

```powershell
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements-lock.txt
```

## 2. Восстановить готовые модели для разработки

Если в `../../plugins/Cattery/WAFT/` уже есть обе `.cat` из полного пакета Kyven:

```powershell
.\.venv\Scripts\python.exe tools\prepare_workspace.py
```

Команда копирует недостающие файлы в `WAFT/Cattery/WAFT/`, включая обе модели
(около 369 МиБ). Существующие файлы разработки не перезаписывает.
Если у вас только Git checkout без моделей, пропустите этот шаг и пересоберите их
на следующем шаге. Для обычного запуска ноды копия разработки не требуется.

## 3. Пересобрать модели из исходников

```powershell
.\.venv\Scripts\python.exe tools\download_weights.py
.\.venv\Scripts\python.exe tools\export_waft.py
.\.venv\Scripts\python.exe tools\export_waft_fp16.py
.\.venv\Scripts\python.exe tools\package_waft.py

$nukeExe = 'C:\Program Files\Nuke16.0v4\Nuke16.0.exe'
$env:WAFT_BUILD_VARIANT = 'FP32'
& $nukeExe --nukex -i --safe -t tools\create_cat.py
$env:WAFT_BUILD_VARIANT = 'FP16'
& $nukeExe --nukex -i --safe -t tools\create_cat.py
Remove-Item Env:\WAFT_BUILD_VARIANT
```

Замените `$nukeExe` на ваш путь к Nuke 16. Checkpoint скачивается из Google Drive
по сохранённому ID и проверяется по SHA-256. Изменённые веса не заменяют checkpoint.
Лицензия: `WAFT-source/LICENSE`. Промежуточные `.pt` появятся в `build/`;
готовые `.cat` — в `WAFT/Cattery/WAFT/`.

## 4. Проверить

Если вы пропустили сборку, сначала задайте `$nukeExe`, как выше.

```powershell
.\.venv\Scripts\python.exe tools\make_test_images.py
& $nukeExe --nukex -i -t tools\verify_nuke.py
& $nukeExe --nukex -i -t tools\verify_resolution_v2.py
```

Первая проверка рендерит Motion Vectors и Smart Vectors с известным движением,
проверяет каналы, размер и числовую точность. Вторая проверяет масштаб векторов.
Свежие отчёты появляются в `build/`. После полного экспорта доступны дополнительные
проверки (требуют CUDA и checkpoint):

```powershell
.\.venv\Scripts\python.exe tools\verify_upstream.py
.\.venv\Scripts\python.exe tools\benchmark_precision.py FP32
.\.venv\Scripts\python.exe tools\benchmark_precision.py FP16
```

Скрипты для пользовательского видео используют внешний клип по пути в коде;
клип не включён. Для их запуска укажите собственный файл и повторите проверки.

## 5. Упаковать и установить

```powershell
.\.venv\Scripts\python.exe tools\bundle_waft.py
```

Появится `WAFT-Cattery-Nuke16.zip` с моделями. Упаковщик берёт свежие отчёты
из `build/`, а отсутствующие — из `reports/`; смотрите происхождение отчётов
в `validation.json`. Сохранённые отчёты относятся к прежней сборке.

После проверки скопируйте весь пакет `WAFT/Cattery/WAFT/` в
`../../plugins/Cattery/WAFT/` и перезапустите Nuke. Для отдельной установки
копируйте эту же папку целиком в `.nuke/Cattery/`.

## Что игнорируется

`.gitignore` исключает `.venv/`, `checkpoints/`, `build/`, `backups/`, ZIP,
кеши Python, `.pt`, `.pth` и `.cat`. Исходники, лицензии, инструкции и базовые
отчёты сохраняются в Git. Готовые `.cat` доставляются в полном релизном архиве;
Git checkout их не содержит. Для запуска плагина `development/` не нужна.

Исходный `D:/waft` сохранён. Тяжёлые файлы убраны из папки Kyven в
`D:/waft/kyven-cleanup-backup-20261115/`.
