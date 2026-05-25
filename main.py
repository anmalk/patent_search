"""
PatentSearch — Веб-приложение семантического поиска патентов
============================================================
Запуск:
    pip install -r requirements.txt
    python indexer.py          # один раз — индексирует патенты
    uvicorn main:app --reload  # запускает сервер
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from typing import Optional
import os
import time

app = FastAPI(title="PatentSearch")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────

COLLECTION_NAME = "patents"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

AVAILABLE_MODELS = {
    "e5-base": {
        "path":     "models/e5-base-hierarchical",
        "fallback": "intfloat/multilingual-e5-base",
        "label":    "E5-base (дообученная)",
        "dim":      768,
    },
    "ruscibert": {
        "path":     "models/ruscibert-hierarchical",
        "fallback": "ai-forever/ruSciBERT",
        "label":    "ruSciBERT (дообученная)",
        "dim":      768,
    },
    "rubert-large": {
        "path":     "models/rubert-large-hierarchical",
        "fallback": "ai-forever/ruBert-large",
        "label":    "ruBert-large (дообученная)",
        "dim":      1024,
    },
}

# ─────────────────────────────────────────────
# Глобальное состояние
# ─────────────────────────────────────────────

current_model = None
current_model_name = None
qdrant_client = None

# ─────────────────────────────────────────────
# Pydantic модели
# ─────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    ipc_filter: Optional[str] = None
    model_name: str = "e5-base"

class PatentResult(BaseModel):
    rank: int
    patent_id: str
    score: float
    ipc: str
    abstract: str
    title: str = ""

class SearchResponse(BaseModel):
    results: list[PatentResult]
    total: int
    time_ms: float
    model_used: str
    query: str

# ─────────────────────────────────────────────
# Инициализация
# ─────────────────────────────────────────────

def load_model(model_key: str):
    global current_model, current_model_name
    from sentence_transformers import SentenceTransformer

    cfg = AVAILABLE_MODELS.get(model_key)
    if not cfg:
        raise ValueError(f"Неизвестная модель: {model_key}")

    model_path = cfg["path"] if os.path.exists(cfg["path"]) else cfg["fallback"]
    print(f"[MODEL] Загружаем: {model_path}")
    current_model = SentenceTransformer(model_path)
    current_model_name = model_key
    print(f"[MODEL] Готово: {model_key}")


def get_qdrant():
    global qdrant_client
    if qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
            qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            qdrant_client.get_collections()  # проверка соединения
            print(f"[QDRANT] Подключено: {QDRANT_HOST}:{QDRANT_PORT}")
        except Exception as e:
            print(f"[QDRANT] Недоступен: {e}")
            qdrant_client = None
    return qdrant_client


@app.on_event("startup")
async def startup():
    try:
        load_model("e5-base")
    except Exception as e:
        print(f"[WARN] Модель не загружена при старте: {e}")
    get_qdrant()

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    client = get_qdrant()
    qdrant_ok = False
    doc_count = 0
    if client:
        try:
            info = client.get_collection(COLLECTION_NAME)
            qdrant_ok = True
            doc_count = info.points_count
        except Exception:
            pass
    return {
        "status":            "ok",
        "model_loaded":      current_model is not None,
        "model_name":        current_model_name,
        "qdrant_connected":  qdrant_ok,
        "documents_indexed": doc_count,
    }


@app.get("/api/models")
async def get_models():
    result = {}
    for key, cfg in AVAILABLE_MODELS.items():
        result[key] = {
            "label":     cfg["label"],
            "available": os.path.exists(cfg["path"]),
            "active":    key == current_model_name,
        }
    return result


@app.post("/api/load-model")
async def load_model_api(model_name: str):
    try:
        load_model(model_name)
        return {"status": "ok", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Запрос не может быть пустым")

    # Переключаем модель если нужно
    if req.model_name != current_model_name:
        try:
            load_model(req.model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {e}")

    if current_model is None:
        raise HTTPException(status_code=503, detail="Языковая модель не загружена")

    t0 = time.time()
    query_vector = current_model.encode(req.query).tolist()

    client = get_qdrant()
    if client is None:
        return _demo_results(req, time.time() - t0)

    # Фильтр по МПК
    search_filter = None
    if req.ipc_filter and req.ipc_filter.strip():
        from qdrant_client.models import Filter, FieldCondition, MatchText
        search_filter = Filter(
            must=[FieldCondition(
                key="ipc",
                match=MatchText(text=req.ipc_filter.strip()),
            )]
        )

    try:
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=req.top_k,
            query_filter=search_filter,
        ).points
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка Qdrant: {e}")

    elapsed_ms = (time.time() - t0) * 1000

    results = [
        PatentResult(
            rank=i + 1,
            patent_id=h.payload.get("patent_id", str(h.id)),
            score=round(float(h.score), 4),
            ipc=h.payload.get("ipc", "—"),
            abstract=h.payload.get("abstract", "")[:600],
            title=h.payload.get("title", ""),
        )
        for i, h in enumerate(hits)
    ]

    return SearchResponse(
        results=results,
        total=len(results),
        time_ms=round(elapsed_ms, 1),
        model_used=AVAILABLE_MODELS[current_model_name]["label"],
        query=req.query,
    )


def _demo_results(req: SearchRequest, elapsed: float) -> SearchResponse:
    """Демо-режим если Qdrant недоступен"""
    demo = [
        ("RU-2506129-C1", "B03C3/34",
         "Электродинамический фильтр",
         "Изобретение относится к электрической очистке газов от взвешенных частиц. Предложен электрофильтр с коронирующими электродами новой геометрии для улавливания субмикронных частиц. Эффективность очистки газа составляет 99,95%."),
        ("RU-2537812-C1", "A23L2/38",
         "Способ производства тонизирующего напитка",
         "Изобретение относится к безалкогольной промышленности. Предложен способ получения тонизирующего напитка с радиопротекторными и адаптогенными свойствами на основе лимонника китайского."),
        ("RU-2521678-C1", "G01C19/56",
         "Способ изготовления микрогироскопа",
         "Изобретение относится к гироскопии. Предложен способ изготовления микрогироскопа с обезгаживанием в вакуумной камере при остаточном давлении не более 5·10⁻⁵ мм рт.ст."),
        ("RU-2438125-C1", "G01N33/24",
         "Способ повышения водопрочности почвенных агрегатов",
         "Изобретение относится к сельскому хозяйству. Предложен способ повышения водопрочности почвенных агрегатов для улучшения структуры почвы."),
        ("RU-2070350-C1", "H01L21/20",
         "Способ создания эпитаксиального слоя",
         "Изобретение относится к полупроводниковой технике. Предложен метод создания эпитаксиального слоя кремния на подложке с субмикронными маскирующими участками."),
    ]
    results = [
        PatentResult(
            rank=i + 1,
            patent_id=pid,
            score=round(0.96 - i * 0.05, 4),
            ipc=ipc,
            title=title,
            abstract=abstract,
        )
        for i, (pid, ipc, title, abstract) in enumerate(demo[:req.top_k])
    ]
    return SearchResponse(
        results=results,
        total=len(results),
        time_ms=round(elapsed * 1000, 1),
        model_used="⚠ ДЕМО-РЕЖИМ (Qdrant недоступен)",
        query=req.query,
    )
