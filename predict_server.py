import os
import time
import logging
from typing import List, Tuple, Optional
import asyncio

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ----------------------------
# Настройка логирования
# ----------------------------
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Классы Pydantic
# ----------------------------
class PredictIn(BaseModel):
    input: str
    model_dir: Optional[str] = None
    max_length: int = 128

class SpanOut(BaseModel):
    start_index: int
    end_index: int
    entity: str

# ----------------------------
# FastAPI приложение
# ----------------------------
app = FastAPI(title="NER Hackathon (Model-based)")

MODEL_DIR_DEFAULT = os.environ.get("MODEL_DIR", "./models/deeppavlov")

_tokenizer = None
_model = None
_id_to_label = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Middleware для логирования времени
# ----------------------------
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    total_time = (time.perf_counter() - start_time) * 1000  # ms
    client_host = request.client.host if request.client else "unknown"
    client_port = request.client.port if request.client else 0
    logging.info(f"{client_host}:{client_port} "
                 f"{request.method} {request.url.path} -> {response.status_code} | TOTAL: {total_time:.2f} ms")
    return response

# ----------------------------
# Загрузка модели
# ----------------------------
def _ensure_model(model_dir: str):
    global _tokenizer, _model, _id_to_label
    if _tokenizer is not None and _model is not None and _id_to_label is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForTokenClassification.from_pretrained(model_dir)
    _model.to(_device)
    _model.eval()
    label_map = getattr(_model.config, "id2label", None)
    if isinstance(label_map, dict) and len(label_map) > 0:
        _id_to_label = {int(i): str(l) for i, l in label_map.items()}
    else:
        _id_to_label = {i: str(v) for i, v in enumerate(_model.config.id2label.values())}

# ----------------------------
# Декодирование BIO
# ----------------------------
def decode_bio_spans(tokens_offsets: List[Tuple[int, int]], labels: List[str]) -> List[Tuple[int, int, str]]:
    def normalize_label(curr_label: str, prev_base: str) -> str:
        if curr_label.startswith("I-"):
            base = curr_label.split("-", 1)[1]
            if not prev_base or prev_base != base:
                return "B-" + base
        return curr_label

    token_segments: List[Tuple[int, int, str]] = []
    prev_base_for_norm: str = ""
    for (start, end), raw_lab in zip(tokens_offsets, labels):
        if end == 0 and start == 0:
            continue
        lab = normalize_label(raw_lab, prev_base_for_norm)
        if lab == "O":
            base = "O"
            prev_base_for_norm = ""
        else:
            base = lab.split("-", 1)[1]
            prev_base_for_norm = base
        token_segments.append((int(start), int(end), base))

    if not token_segments:
        return []

    # Merge contiguous subword pieces
    word_spans: List[Tuple[int, int, str]] = []
    for s, e, base in token_segments:
        if not word_spans:
            word_spans.append((s, e, base))
            continue
        ps, pe, pbase = word_spans[-1]
        if base == pbase and pe == s:
            word_spans[-1] = (ps, e, pbase)
        else:
            word_spans.append((s, e, base))

    # Assign BIO across words
    spans: List[Tuple[int, int, str]] = []
    prev_entity_base: str = ""
    for s, e, base in word_spans:
        if base == "O":
            spans.append((s, e, "O"))
            prev_entity_base = ""
        else:
            tag = ("B-" if prev_entity_base != base else "I-") + base
            spans.append((s, e, tag))
            prev_entity_base = base

    return spans

# ----------------------------
# Синхронная функция предсказания (для run_in_executor)
# ----------------------------
def _predict_sync(text: str, max_length: int) -> List[Tuple[int, int, str]]:
    enc = _tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping").squeeze(0).tolist()
    enc = {k: v.to(_device) for k, v in enc.items()}
    with torch.no_grad():
        logits = _model(**enc).logits
    pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
    pred_labels = [_id_to_label.get(i, "O") for i in pred_ids]
    return decode_bio_spans(offsets, pred_labels)

# ----------------------------
# Эндпоинты
# ----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/predict", response_model=List[SpanOut])
async def predict(payload: PredictIn) -> List[SpanOut]:
    text = payload.input or ""
    if text == "":
        return []

    model_dir = payload.model_dir or MODEL_DIR_DEFAULT
    _ensure_model(model_dir)

    # Асинхронно вызываем синхронную функцию через пул потоков
    spans = await asyncio.get_running_loop().run_in_executor(
        None, _predict_sync, text, payload.max_length
    )

    return [SpanOut(start_index=s, end_index=e, entity=lab) for (s, e, lab) in spans]

# ----------------------------
# Запуск сервера
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "predict_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=2,   # Можно оставить 1, CPU будет использовать пул потоков
        reload=False
    )
