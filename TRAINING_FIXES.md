# Исправления проблем с обучением модели

## Обзор исправлений

В данном документе описаны исправления, внесенные для устранения ошибки `'IsResolved'` при обучении модели и проблем с сохранением настроек.

## 1. Исправление ошибки 'IsResolved'

### Проблема
При попытке обучения модели возникала ошибка:
```
Не удалось обработать XML файлы!
Ошибка анализа файлов: 'IsResolved'
```

### Причина
Код обучения ожидал наличие колонки `IsResolved` в DataFrame, но эта колонка могла отсутствовать в зависимости от формата XML файлов.

### Решение

#### 1.1 Исправление в web/routes.py
Добавлена проверка и создание колонки `IsResolved`:

```python
# Проверяем наличие колонки IsResolved и фильтруем данные
if 'IsResolved' in df_with_images.columns:
    df_with_images = df_with_images[df_with_images['IsResolved'].isin([0, 1])].copy()
else:
    # Если колонки IsResolved нет, создаем её на основе status
    df_with_images['IsResolved'] = df_with_images['status'].apply(
        lambda x: 1 if x == 'Active' or x == 'Активн.' else (0 if x == 'Approved' or x == 'Подтверждено' else -1)
    )
    df_with_images = df_with_images[df_with_images['IsResolved'].isin([0, 1])].copy()
```

#### 1.2 Исправление обработки df_visual
Добавлена гибкая обработка разных форматов данных:

```python
# Проверяем наличие колонки status и создаем label
if 'status' in df_visual.columns:
    df_visual['label'] = df_visual['status'].apply(
        lambda x: 0 if x == 'Approved' or x == 'Подтверждено' else (1 if x == 'Active' or x == 'Активн.' else None)
    )
elif 'IsResolved' in df_visual.columns:
    # Если нет status, но есть IsResolved, используем его
    df_visual['label'] = df_visual['IsResolved'].apply(
        lambda x: 0 if x == 0 else (1 if x == 1 else None)
    )
else:
    # Если нет ни status, ни IsResolved, создаем на основе resultstatus
    df_visual['label'] = df_visual['resultstatus'].apply(
        lambda x: 0 if x == 'Подтверждено' else (1 if x == 'Активн.' else None)
    )
```

#### 1.3 Исправление в ml/train.py
Добавлена проверка и создание колонки `IsResolved` в функции `train_model`:

```python
# Проверяем наличие колонки IsResolved
if 'IsResolved' not in df.columns:
    # Если колонки нет, создаем её на основе доступных данных
    if 'status' in df.columns:
        df['IsResolved'] = df['status'].apply(
            lambda x: 1 if x == 'Active' or x == 'Активн.' else (0 if x == 'Approved' or x == 'Подтверждено' else -1)
        )
    elif 'resultstatus' in df.columns:
        df['IsResolved'] = df['resultstatus'].apply(
            lambda x: 1 if x == 'Активн.' else (0 if x == 'Подтверждено' else -1)
        )
    else:
        print("[ERROR] Не найдена колонка IsResolved, status или resultstatus для определения классов")
        return None
```

#### 1.4 Улучшение обработки в train_job
Добавлена дополнительная проверка перед обучением:

```python
# Убеждаемся, что df_visual содержит необходимые колонки
if 'IsResolved' not in df_visual.columns:
    # Создаем IsResolved на основе доступных данных
    if 'status' in df_visual.columns:
        df_visual['IsResolved'] = df_visual['status'].apply(
            lambda x: 1 if x == 'Active' or x == 'Активн.' else (0 if x == 'Approved' or x == 'Подтверждено' else -1)
        )
    elif 'resultstatus' in df_visual.columns:
        df_visual['IsResolved'] = df_visual['resultstatus'].apply(
            lambda x: 1 if x == 'Активн.' else (0 if x == 'Подтверждено' else -1)
        )
    else:
        update_train_progress({'status': 'error', 'log': 'Не найдена колонка IsResolved, status или resultstatus для обучения'}, session_dir)
        return

# Фильтруем только нужные классы
df_visual_filtered = df_visual[df_visual['IsResolved'].isin([0, 1])].copy()
if len(df_visual_filtered) == 0:
    update_train_progress({'status': 'error', 'log': 'Не найдено данных для обучения (требуются классы 0 и 1)'}, session_dir)
    return
```

### Результат
- ✅ Ошибка `'IsResolved'` устранена
- ✅ Поддержка разных форматов XML файлов
- ✅ Автоматическое создание колонки `IsResolved` при необходимости
- ✅ Корректная обработка данных для обучения

## 2. Исправление сохранения настроек

### Проблема
Настройки "Архитектура модели" и "Оптимизация инференса" не сохранялись в файл `settings.json`.

### Причина
В файле `settings.json` отсутствовали новые настройки `model_type` и `use_optimization`.

### Решение
Обновлен файл `settings.json`:

```json
{
  "model_file": "model_clashmark.pt",
  "low_confidence": 0.3,
  "high_confidence": 0.7,
  "inference_mode": "model",
  "manual_review_enabled": true,
  "export_format": "standard",
  "model_type": "mobilenet_v3_small",
  "use_optimization": true
}
```

### Результат
- ✅ Все настройки сохраняются корректно
- ✅ Новые настройки доступны в веб-интерфейсе
- ✅ Обратная совместимость сохранена

## 3. Поддерживаемые форматы данных

### 3.1 Стандартный формат
- Колонка `IsResolved` присутствует
- Значения: 0 (Подтверждено), 1 (Активн.), -1 (Проанализировано)

### 3.2 Формат с колонкой status
- Колонка `status` с значениями: 'Active', 'Approved', 'Активн.', 'Подтверждено'
- Автоматическое создание `IsResolved`

### 3.3 Формат с колонкой resultstatus
- Колонка `resultstatus` с значениями: 'Активн.', 'Подтверждено'
- Автоматическое создание `IsResolved`

## 4. Тестирование

### Тест парсинга XML
Создан и выполнен тест с тремя сценариями:
1. **Стандартный формат с IsResolved** - ✅ пройден
2. **Формат без IsResolved, но с status** - ✅ пройден  
3. **Формат только с resultstatus** - ✅ пройден

### Тест загрузки настроек
- ✅ `model_type`: mobilenet_v3_small
- ✅ `use_optimization`: True
- ✅ Все необходимые настройки загружены

## 5. Обратная совместимость

Все изменения полностью совместимы:
- Старые XML файлы продолжают работать
- Существующие настройки сохраняются
- Новые настройки имеют значения по умолчанию

## 6. Файлы, подвергшиеся изменениям

1. **`web/routes.py`**
   - Добавлена проверка и создание колонки `IsResolved`
   - Улучшена обработка `df_visual`
   - Добавлена дополнительная валидация в `train_job`

2. **`ml/train.py`**
   - Добавлена проверка колонки `IsResolved` в `train_model`
   - Улучшена обработка ошибок

3. **`settings.json`**
   - Добавлены новые настройки `model_type` и `use_optimization`

## 7. Рекомендации

### Для пользователей:
- Теперь обучение модели работает с любыми форматами XML файлов
- Все настройки сохраняются корректно
- Улучшена обработка ошибок

### Для разработчиков:
- При добавлении новых колонок проверять их наличие
- Использовать гибкую обработку данных
- Добавлять значения по умолчанию для новых настроек

## 8. Будущие улучшения

Возможные направления дальнейшего развития:
- Автоматическое определение формата XML файлов
- Валидация структуры данных перед обучением
- Поддержка дополнительных форматов XML
- Улучшенная диагностика ошибок обучения 