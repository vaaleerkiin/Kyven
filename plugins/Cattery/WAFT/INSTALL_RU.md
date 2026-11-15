# WAFT в Kyven и Cattery

Эта папка целиком переносима. Обе модели уже внутри:
`WAFT.cat` (GPU FP16) и `WAFT_FP32.cat` (FP32, в том числе CPU).

В Kyven: перезапустите Nuke 16 и выберите **Kyven → WAFT SmartVector**
или нажмите Tab и введите WAFT. Kyven Server запускать не нужно.

Для отдельной установки скопируйте всю эту папку `WAFT` в
`%USERPROFILE%/.nuke/Cattery/`. Путь должен получиться
`.nuke/Cattery/WAFT/cat.json`, без лишнего вложенного уровня.
После перезапуска Nuke плагин доступен через Cattery → Optical Flow → WAFT.

Документация модели и органов управления находится в `README_RU.md`.

Исходники и восстановление среды: `D:/Kyven/development/waft/README_RU.md`.
Исходная рабочая папка `D:/waft` также сохранена.
