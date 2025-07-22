# ClashMark — API

## Анализ файлов
**POST** `/analyze`
- Вход: XML и ZIP
- Выход: статистика, ссылки на скачивание, список для ручной разметки

## Сохранение ручной разметки
**POST** `/api/manual_review`
- Вход: { session_id, reviews: [{clash_id, status}] }
- Выход: { success: true }

## Получение обновлённой статистики
**GET** `/api/updated_stats/<session_id>`
- Выход: { total_collisions, total_approved, ... detailed_stats }

## Скачивание файла
**GET** `/download/<session_id>/<filename>`

## Получение настроек
**GET** `/api/settings`
- Выход: текущие настройки инференса, экспорта, модели и др.

## Пример ручной разметки (POST /api/manual_review)
```json
{
  "session_id": "analysis_session_xxx",
  "reviews": [
    {"clash_id": "123", "status": "Approved"},
    {"clash_id": "124", "status": "Active"}
  ]
}
```

## Пример ответа на анализ (POST /analyze)
```json
{
  "success": true,
  "session_id": "analysis_session_xxx",
  "download_links": [
    {"name": "cv_results_Имя.xml", "url": "/download/analysis_session_xxx/cv_results_Имя.xml"}
  ],
  "stats_total": {
    "total_files": 1,
    "total_collisions": 100,
    "total_approved": 50,
    "total_active": 30,
    "total_reviewed": 20
  },
  "manual_review_collisions": [
    {"clash_id": "123", "image_file": "BSImages/Имя.png", ...}
  ],
  "detailed_stats": [
    {"file_name": "Имя.xml", "algorithm": {...}, "model": {...}, "manual": {...}}
  ]
}
```

---

## Визуализация интерфейса

Скриншоты основных страниц и функций веб-приложения размещены в папке `images/` и приведены в README.md и INSTRUCTION.md для наглядности. 

---

## Примечание о Docker

- Все API-эндпоинты доступны при запуске приложения в Docker-контейнере (порт 5001).
- Для обмена файлами модели используйте volume: папка model/ на хосте монтируется в /app/model внутри контейнера.
- Пример запуска: docker run -p 5001:5001 -v $(pwd)/model:/app/model clashmark 