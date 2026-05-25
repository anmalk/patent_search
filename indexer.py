"""
Индексатор патентов в Qdrant
============================
Запускается один раз перед стартом приложения.
Читает patents_merged.jsonl и загружает все патенты в Qdrant.

Использование:
    python indexer.py
    python indexer.py --model ruscibert
    python indexer.py --file my_patents.jsonl --batch 128
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

# ─────────────────────────────────────────────
# Аргументы
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Индексирование патентов в Qdrant")
parser.add_argument("--file",       default="patents_merged.jsonl",
                    help="Путь к JSONL файлу с патентами")
parser.add_argument("--model",      default="e5-base",
                    choices=["e5-base", "ruscibert", "rubert-large"],
                    help="Модель для кодирования")
parser.add_argument("--batch",      type=int, default=64,
                    help="Размер батча")
parser.add_argument("--limit",      type=int, default=None,
                    help="Ограничение числа патентов (для тестирования)")
parser.add_argument("--host",       default=os.getenv("QDRANT_HOST", "localhost"))
parser.add_argument("--port",       type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
parser.add_argument("--collection", default="patents")
parser.add_argument("--recreate",   action="store_true",
                    help="Пересоздать коллекцию если существует")
args = parser.parse_args()

# ─────────────────────────────────────────────
# Конфигурация моделей
# ─────────────────────────────────────────────

MODEL_CONFIGS = {
    "e5-base": {
        "path":     "models/e5-base-hierarchical",
        "fallback": "intfloat/multilingual-e5-base",
        "dim":      768,
        "prefix":   "passage: ",   # e5 требует префикс
    },
    "ruscibert": {
        "path":     "models/ruscibert-hierarchical",
        "fallback": "ai-forever/ruSciBERT",
        "dim":      768,
        "prefix":   "",
    },
    "rubert-large": {
        "path":     "models/rubert-large-hierarchical",
        "fallback": "ai-forever/ruBert-large",
        "dim":      1024,
        "prefix":   "",
    },
}

# ─────────────────────────────────────────────
# Шаг 1 — Проверка файла
# ─────────────────────────────────────────────

patents_file = Path(args.file)
if not patents_file.exists():
    print(f"[ERROR] Файл не найден: {patents_file}")
    print("Укажите путь через --file: python indexer.py --file /path/to/patents.jsonl")
    sys.exit(1)

print(f"[INFO] Файл: {patents_file}")

# ─────────────────────────────────────────────
# Шаг 2 — Загрузка модели
# ─────────────────────────────────────────────

cfg = MODEL_CONFIGS[args.model]
model_path = cfg["path"] if os.path.exists(cfg["path"]) else cfg["fallback"]
prefix = cfg["prefix"]
dim = cfg["dim"]

print(f"[MODEL] Загружаем: {model_path}")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_path)
print(f"[MODEL] Готово. Размерность: {dim}")

# ─────────────────────────────────────────────
# Шаг 3 — Подключение к Qdrant
# ─────────────────────────────────────────────

print(f"[QDRANT] Подключаемся к {args.host}:{args.port}...")
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    HnswConfigDiff, OptimizersConfigDiff
)

client = QdrantClient(host=args.host, port=args.port)

# Проверяем соединение
try:
    client.get_collections()
    print(f"[QDRANT] Подключено успешно")
except Exception as e:
    print(f"[ERROR] Не удалось подключиться к Qdrant: {e}")
    print("Убедитесь что Qdrant запущен: docker run -p 6333:6333 qdrant/qdrant")
    sys.exit(1)

# Создаём или пересоздаём коллекцию
existing = [c.name for c in client.get_collections().collections]

if args.collection in existing:
    if args.recreate:
        print(f"[QDRANT] Удаляем коллекцию '{args.collection}'...")
        client.delete_collection(args.collection)
    else:
        info = client.get_collection(args.collection)
        existing_count = info.points_count
        print(f"[QDRANT] Коллекция уже существует ({existing_count:,} документов)")
        ans = input("Пересоздать? (y/N): ").strip().lower()
        if ans == "y":
            client.delete_collection(args.collection)
        else:
            print("[INFO] Добавляем к существующей коллекции")

if args.collection not in [c.name for c in client.get_collections().collections]:
    print(f"[QDRANT] Создаём коллекцию '{args.collection}' (dim={dim})...")
    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )
    print(f"[QDRANT] Коллекция создана")

# ─────────────────────────────────────────────
# Шаг 4 — Чтение и фильтрация патентов
# ─────────────────────────────────────────────

print(f"\n[DATA] Читаем патенты из {patents_file}...")

patents = []
skipped = 0

with open(patents_file, encoding="utf-8") as f:
    for line in tqdm(f, desc="Читаем файл"):
        try:
            p = json.loads(line.strip())
        except json.JSONDecodeError:
            skipped += 1
            continue

        abstract = (p.get("abstract_ru") or "").strip()
        if not abstract:
            skipped += 1
            continue

        patents.append({
            "patent_id": p.get("publication_number", ""),
            "abstract":  abstract[:1500],
            "title":     (p.get("title") or "").strip(),
            "ipc":       (p.get("ipc_codes") or "").strip(),
            "family_id": (p.get("family_id") or "").strip(),
        })

        if args.limit and len(patents) >= args.limit:
            break

print(f"[DATA] Загружено: {len(patents):,} патентов (пропущено: {skipped:,})")

if not patents:
    print("[ERROR] Нет патентов для индексирования")
    sys.exit(1)

# ─────────────────────────────────────────────
# Шаг 5 — Кодирование и загрузка в Qdrant
# ─────────────────────────────────────────────

print(f"\n[INDEX] Начинаем индексирование батчами по {args.batch}...")
print(f"[INDEX] Всего батчей: {len(patents) // args.batch + 1}")

t_start = time.time()
total_indexed = 0
point_id = 0

# Получаем начальный ID если коллекция не пустая
try:
    info = client.get_collection(args.collection)
    point_id = info.points_count
except Exception:
    point_id = 0

pbar = tqdm(range(0, len(patents), args.batch), desc="Индексируем")

for batch_start in pbar:
    batch = patents[batch_start: batch_start + args.batch]

    # Формируем тексты с префиксом если нужен
    texts = [prefix + p["abstract"] for p in batch]

    # Кодируем
    try:
        vectors = model.encode(
            texts,
            batch_size=args.batch,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
    except Exception as e:
        print(f"\n[WARN] Ошибка кодирования батча: {e}, пропускаем")
        continue

    # Формируем точки для Qdrant
    points = [
        PointStruct(
            id=point_id + j,
            vector=vectors[j],
            payload={
                "patent_id": batch[j]["patent_id"],
                "abstract":  batch[j]["abstract"],
                "title":     batch[j]["title"],
                "ipc":       batch[j]["ipc"],
                "family_id": batch[j]["family_id"],
            },
        )
        for j in range(len(batch))
    ]

    # Загружаем в Qdrant
    try:
        client.upsert(collection_name=args.collection, points=points)
    except Exception as e:
        print(f"\n[WARN] Ошибка загрузки батча: {e}")
        continue

    point_id += len(batch)
    total_indexed += len(batch)

    elapsed = time.time() - t_start
    speed = total_indexed / elapsed if elapsed > 0 else 0
    remaining = (len(patents) - total_indexed) / speed if speed > 0 else 0

    pbar.set_postfix({
        "проиндексировано": f"{total_indexed:,}",
        "ск/сек":           f"{speed:.0f}",
        "осталось":         f"{remaining/60:.1f}мин",
    })

# ─────────────────────────────────────────────
# Шаг 6 — Итог
# ─────────────────────────────────────────────

elapsed_total = time.time() - t_start
info = client.get_collection(args.collection)

print(f"\n{'='*55}")
print(f"ИНДЕКСИРОВАНИЕ ЗАВЕРШЕНО")
print(f"{'='*55}")
print(f"  Проиндексировано: {total_indexed:,} патентов")
print(f"  Итого в Qdrant:   {info.points_count:,} документов")
print(f"  Время:            {elapsed_total/60:.1f} минут")
print(f"  Скорость:         {total_indexed/elapsed_total:.0f} патентов/сек")
print(f"  Коллекция:        {args.collection}")
print(f"  Модель:           {args.model} ({model_path})")
print(f"{'='*55}")
print(f"\nТеперь запускайте сервер: uvicorn main:app --reload")
