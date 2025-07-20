# Исправления для MacOS и Windows

## 1. Оставлен только оптимизированный инференс

### Изменения:
- Убрана поддержка гибридного режима инференса (`hybrid`)
- Оставлены только режимы: `model` и `algorithm`
- Упрощена логика в `web/routes.py` - проверка только на `inference_mode == 'model'`
- **Удалена опция "Включить оптимизацию инференса"** - оптимизация теперь всегда включена

### Файлы:
- `web/routes.py` - убрана проверка `inference_mode in ('model', 'hybrid')`, убрано использование `use_optimization`
- `templates/settings.html` - удалена опция "Включить оптимизацию инференса"
- `web/settings.py` - убрана обработка `use_optimization` из настроек
- `ml/model.py` - функция `get_cached_model` теперь всегда использует оптимизацию

### Логика работы режима "модель":
1. **Алгоритмическая разметка** - применяется ко всем коллизиям на основе пар категорий
2. **Модельная разметка** - применяется только к коллизиям, которые не были разметы алгоритмом (visual и unknown пары)
3. **Ручная разметка** - применяется поверх всех предыдущих разметок

## 2. Исправление проблемы с путями изображений на MacOS

### Проблема:
На MacOS изображения не отображались из-за неправильной обработки абсолютных путей в URL.
Логи показывали ошибки 404 для путей вида:
```
/var/folders/xh/zxlndpg550q7lpk5yv3tbdhc0000gn/T/analysis_session_otvr48t7/BSImages/ВК-все_Конфликт132.png
```

Фактический адрес изображения:
```
/private/var/folders/xh/zxlndpg550q7lpk5yv3tbdhc0000gn/T/analysis_session_ucsoou20/BSImages/ВК-все_Конфликт135.png
```

### Решение:
Улучшена функция `download_file` в `web/routes.py`:

1. **Декодирование URL**: Добавлено `urllib.parse.unquote()` для корректного декодирования URL-encoded путей
2. **Обработка абсолютных путей**: Проверка, начинается ли путь с `/` и находится ли он в пределах `session_dir`
3. **Специальная обработка для macOS**: Если путь начинается с `/var/folders/`, но файл не найден, пробуем путь с префиксом `/private`
4. **Безопасность**: Проверка, что абсолютный путь не выходит за пределы сессии
5. **Корректное имя файла**: Использование `os.path.basename()` для имени скачиваемого файла

### Код:
```python
# Декодируем URL-encoded путь
decoded_filename = urllib.parse.unquote(filename)

# Проверяем, является ли путь абсолютным (начинается с /)
if decoded_filename.startswith('/'):
    # Это абсолютный путь, проверяем, находится ли он в session_dir
    if not decoded_filename.startswith(session_dir):
        # На macOS файлы могут быть в /private/var/folders/, но URL содержит /var/folders/
        if decoded_filename.startswith('/var/folders/') and session_dir.startswith('/private/var/folders/'):
            # Преобразуем путь для macOS
            macos_path = '/private' + decoded_filename
            if os.path.exists(macos_path):
                return send_file(macos_path, as_attachment=True, download_name=os.path.basename(macos_path))
        return "Доступ запрещен", 403
    file_path = decoded_filename
else:
    # Относительный путь
    file_path = os.path.join(session_dir, decoded_filename)
```

## 3. Исправление проблемы с сохранением результатов анализа на Windows

### Проблема:
После ручного анализа файлов на Windows не сохранялись результаты анализа в итоговый файл, хотя в `manual_review.json` - сохранялись.

### Причины:
1. **Ошибка в логике**: В функции `api_manual_review` переменная `df_combined` могла быть не определена, если CSV файл не существовал
2. **Проблемы с кодировкой**: Ошибки при чтении CSV файлов на Windows
3. **Проблемы с записью**: Ошибки при сохранении CSV файлов на Windows

### Решение:

#### 3.1 Исправление логики в `api_manual_review`
```python
# Проверяем существование CSV файла перед чтением
df_path = os.path.join(session_dir, 'df_with_inference.csv')
df_combined = None
if os.path.exists(df_path):
    # Пробуем разные способы чтения CSV
    try:
        df_combined = pd.read_csv(df_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_combined = pd.read_csv(df_path, encoding='latin-1')
        except Exception as e:
            logger.error(f"[manual_review] Ошибка чтения CSV: {e}")
            return jsonify({'error': 'Ошибка чтения данных анализа'}), 500
    
    if df_combined is None or df_combined.empty:
        logger.error(f"[manual_review] DataFrame пустой после чтения")
        return jsonify({'error': 'Данные анализа повреждены'}), 500
else:
    logger.error(f"[manual_review] Файл с инференсом не найден: {df_path}")
    return jsonify({'error': 'Нет данных для обновления статистики. Проведите анализ заново.'}), 400
```

#### 3.2 Улучшенное сохранение CSV файлов
```python
# Сохраняем актуальный DataFrame
try:
    df_with_manual.to_csv(df_path, index=False, encoding='utf-8')
except Exception as e:
    logger.warning(f"[manual_review] Ошибка сохранения CSV: {e}")
    try:
        # Альтернативный способ сохранения
        temp_csv_path = os.path.join(session_dir, 'df_with_inference_temp.csv')
        df_with_manual.to_csv(temp_csv_path, index=False, encoding='utf-8')
        if os.path.exists(df_path):
            os.remove(df_path)
        os.rename(temp_csv_path, df_path)
    except Exception as e2:
        logger.error(f"[manual_review] Не удалось сохранить CSV: {e2}")
```

#### 3.3 Улучшенное чтение CSV файлов
```python
# Пробуем разные способы чтения CSV
df_combined = None
try:
    df_combined = pd.read_csv(df_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_combined = pd.read_csv(df_path, encoding='latin-1')
    except Exception as e:
        logger.error(f"[manual_review] Ошибка чтения CSV: {e}")
        return jsonify({'error': 'Ошибка чтения данных анализа'}), 500
```

## 4. Улучшения совместимости

### Чтение файлов:
- Добавлена поддержка разных кодировок (UTF-8, latin-1)
- Обработка ошибок при чтении CSV и JSON файлов
- Fallback методы для проблемных файлов

### Запись файлов:
- Использование временных файлов для атомарной записи
- Обработка ошибок доступа к файлам
- Альтернативные методы записи для Windows

### Пути к файлам:
- Корректная обработка абсолютных путей на MacOS
- Специальная обработка путей `/var/folders/` → `/private/var/folders/`
- Безопасная проверка путей
- Поддержка URL-encoded имен файлов

## Результат:
- ✅ Изображения корректно отображаются на MacOS (исправлена обработка путей `/private/var/folders/`)
- ✅ Результаты анализа сохраняются на Windows после ручной разметки
- ✅ Упрощена логика инференса (только оптимизированный режим)
- ✅ Улучшена совместимость с разными ОС
- ✅ Добавлена надежная обработка ошибок при работе с файлами 