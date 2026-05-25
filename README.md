# PatentSearch — Веб-приложение семантического поиска патентов

## Структура проекта

```
patent_search_app/
├── main.py                  ← FastAPI сервер + API
├── indexer.py               ← Индексирование патентов в Qdrant
├── requirements.txt
├── templates/
│   └── index.html           ← HTML интерфейс
├── static/
│   ├── style.css            ← Стили
│   └── app.js               ← JavaScript
└── models/                  ← Папка с дообученными моделями
    ├── e5-base-hierarchical/
    ├── ruscibert-hierarchical/
    └── rubert-large-hierarchical/
```

---

## Установка

```bash
pip install -r requirements.txt
```

---

## Шаг 1 — Запустить Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

---

## Шаг 2 — Проиндексировать патенты

```bash
# Базовый запуск (читает patents_merged.jsonl)
python indexer.py

# С указанием файла и модели
python indexer.py --file /path/to/patents_merged.jsonl --model e5-base

# Только первые 10 000 для теста
python indexer.py --limit 10000

# Все параметры
python indexer.py \
  --file    patents_merged.jsonl \
  --model   e5-base \
  --batch   64 \
  --host    localhost \
  --port    6333 \
  --recreate
```


---

## Шаг 3 — Запустить сервер

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Открыть в браузере: **http://localhost:8000**

---

## Переменные окружения

| Переменная   | По умолчанию | Описание          |
|--------------|--------------|-------------------|
| QDRANT_HOST  | localhost    | Хост Qdrant       |
| QDRANT_PORT  | 6333         | Порт Qdrant       |

---

## API эндпоинты

| Метод | URL               | Описание                    |
|-------|-------------------|-----------------------------|
| GET   | /                 | Веб-интерфейс               |
| POST  | /api/search       | Поиск патентов              |
| GET   | /api/models       | Список моделей              |
| POST  | /api/load-model   | Переключить модель          |
| GET   | /api/health       | Статус системы              |

### POST /api/search

```json
{
  "query":      "текст реферата патентной заявки",
  "top_k":      10,
  "ipc_filter": "G06F",
  "model_name": "e5-base"
}
```

### Ответ

```json
{
  "results": [
    {
      "rank":       1,
      "patent_id":  "RU-2506129-C1",
      "score":      0.923,
      "ipc":        "B03C3/34",
      "title":      "Электродинамический фильтр",
      "abstract":   "Изобретение относится к..."
    }
  ],
  "total":      10,
  "time_ms":    342.1,
  "model_used": "E5-base (дообученная)",
  "query":      "текст запроса"
}
```

---
