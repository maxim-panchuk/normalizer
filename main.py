from __future__ import annotations
import os
import re
from typing import List, Dict, Any, Optional, Tuple

import yaml
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from rapidfuzz import process, fuzz
import pymorphy3

# ──────────────────────────────────────────────────────────────
# Конфиг
LEXICON_PATH = os.getenv("LEXICON_PATH", "ingredients.yml")
FUZZY_THRESHOLD = int(os.getenv("FUZZY_THRESHOLD", "90"))  # 0..100
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "200"))             # кап на кол-во позиций в одном тексте

MORPH = pymorphy3.MorphAnalyzer()
STOP = {"свежий","свежая","свежие","резаный","резаные","молотый","варёный","вареный","жареный","отварной","без","со","из"}
VARIETY = {"черри","сливовидные","пальчиковые","крупный","мелкий"}

PAT = re.compile(r'\s*([^,0-9]+?)\s*([0-9]+(?:[.,][0-9]+)?)\s*(г|гр|грамм|мл|шт)?\s*(?:,|$)', re.IGNORECASE)

def lemma_word(w: str) -> str:
    return MORPH.parse(w)[0].normal_form

def lemma_phrase(s: str) -> str:
    s = re.sub(r'[—–-]+', ' ', s.lower()).strip()
    s = re.sub(r'\s+', ' ', s)
    toks = [t for t in s.split() if t not in STOP and t not in VARIETY]
    lemmas = [lemma_word(t) for t in toks]
    return " ".join(lemmas).strip()

# ──────────────────────────────────────────────────────────────
# Загрузка лексикона (каноны и синонимы)
class Lexicon:
    def __init__(self, path: str):
        self.path = path
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Поддерживаем два формата:
        #   canonical:
        #     Сыр:
        #       synonyms: [сыр, чеддер, ...]
        #       id: CHEESE
        #  или
        #     Сыр: [сыр, чеддер, ...]
        self.canon_meta: Dict[str, Dict[str, Any]] = {}
        syn2canon: Dict[str, str] = {}

        for canon, val in (cfg.get("canonical") or {}).items():
            if isinstance(val, dict):
                syns = list(set([canon] + (val.get("synonyms") or [])))
                cid  = val.get("id")
                meta_extra = {k: v for k, v in val.items() if k not in ("synonyms", "id")}
            else:
                syns = list(set([canon] + list(val)))
                cid = None
                meta_extra = {}

            canon_lemma = lemma_phrase(canon)
            self.canon_meta[canon] = {"id": cid, "lemma": canon_lemma, **meta_extra}

            for s in syns:
                l = lemma_phrase(s)
                if not l:
                    continue
                syn2canon[l] = canon

        # для фуззи
        self._syn_lemmas = list(syn2canon.keys())
        self._syn2canon  = syn2canon

    def reload(self):
        self._load()

    def match(self, raw_name: str, fuzzy_threshold: int) -> Tuple[Optional[str], Optional[str], int, str]:
        """
        Возвращает (canonical_name, canonical_id, score[0..100], matched_lemma)
        """
        q = lemma_phrase(raw_name)
        if not q:
            return None, None, 0, ""

        # 1) точное совпадение по лемме
        if q in self._syn2canon:
            canon = self._syn2canon[q]
            return canon, self.canon_meta[canon].get("id"), 100, q

        # 2) резерв: фуззи-поиск среди всех известных лемм-синонимов
        if not self._syn_lemmas:
            return None, None, 0, ""

        choice, score, _ = process.extractOne(
            q,
            self._syn_lemmas,
            scorer=fuzz.token_set_ratio
        )
        if choice and score >= fuzzy_threshold:
            canon = self._syn2canon[choice]
            return canon, self.canon_meta[canon].get("id"), int(score), choice

        return None, None, int(score or 0), q

LEX = Lexicon(LEXICON_PATH)

# ──────────────────────────────────────────────────────────────
# Парсер ингредиентной строки
def parse_ingredients(text: str) -> List[Dict[str, Any]]:
    text = text or ""
    out: List[Dict[str, Any]] = []
    for m in PAT.finditer(text):
        raw = m.group(1).strip()
        qty = float(m.group(2).replace(',', '.'))
        unit_raw = (m.group(3) or "").lower()
        unit = "г"
        if unit_raw in ("мл",):
            unit = "мл"
        elif unit_raw in ("шт",):
            unit = "шт"
        out.append({"raw": raw, "amount": qty, "unit": unit})
        if len(out) >= MAX_ITEMS:
            break

    if not out and text:
        # fallback: разделение по запятой без граммовок
        chunks = [c.strip() for c in text.split(",") if c.strip()]
        for ch in chunks[:MAX_ITEMS]:
            out.append({"raw": ch, "amount": None, "unit": None})

    return out

# ──────────────────────────────────────────────────────────────
# Pydantic схемы
class NormalizeRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Строка состава блюда")
    items: Optional[List[str]] = Field(default=None, description="Альтернатива: массив строк-ингредиентов")
    threshold: Optional[int] = Field(default=None, ge=0, le=100)

class NormalizedItem(BaseModel):
    raw: str
    name: Optional[str] = None
    id: Optional[str] = None
    score: int
    matched_lemma: str
    amount: Optional[float] = None
    unit: Optional[str] = None

class NormalizeResponse(BaseModel):
    ok: bool
    threshold: int
    normalized: List[NormalizedItem]
    unknown: List[str]
    canonical_string: str

class BulkRequest(BaseModel):
    texts: List[str]
    threshold: Optional[int] = None

# ──────────────────────────────────────────────────────────────
# FastAPI
app = FastAPI(title="Ingredient Normalizer (lemma + lexicon)")

@app.get("/health")
def health():
    return {
        "ok": True,
        "lexicon": {
            "path": LEXICON_PATH,
            "canon_count": len(LEX.canon_meta),
            "synonym_lemmas": len(LEX._syn_lemmas)
        },
        "threshold_default": FUZZY_THRESHOLD
    }

@app.post("/reload")
def reload_lexicon():
    LEX.reload()
    return {"ok": True, "reloaded": True, "canon_count": len(LEX.canon_meta)}

@app.post("/normalize", response_model=NormalizeResponse)
def normalize(req: NormalizeRequest = Body(...)):
    thr = req.threshold if req.threshold is not None else FUZZY_THRESHOLD

    # 1) получить список элементов
    if req.items:
        parsed = [{"raw": it, "amount": None, "unit": None} for it in req.items[:MAX_ITEMS]]
    else:
        parsed = parse_ingredients(req.text or "")

    # 2) сопоставление с базой
    result: List[NormalizedItem] = []
    unknown: List[str] = []

    for p in parsed:
        name, cid, score, ml = LEX.match(p["raw"], thr)
        if not name:
            unknown.append(p["raw"])
        result.append(NormalizedItem(
            raw=p["raw"], name=name, id=cid, score=score, matched_lemma=ml,
            amount=p.get("amount"), unit=p.get("unit")
        ))

    # 3) «человекочитаемая» строка:
    #    - если распознали → используем canonical name
    #    - если нет        → используем исходный raw
    #    порядок сохраняем, граммовки/единицы — тоже
    parts: List[str] = []
    for x in result:
        display = x.name if x.name else x.raw
        if x.amount is not None and x.unit:
            amt = int(x.amount) if abs(x.amount - int(x.amount)) < 1e-6 else x.amount
            parts.append(f"{display} {amt} {x.unit}")
        else:
            parts.append(display)
    canon_str = ", ".join(parts)

    return NormalizeResponse(
        ok=True,
        threshold=thr,
        normalized=result,
        unknown=sorted(set(unknown)),
        canonical_string=canon_str
    )

@app.post("/bulk_normalize")
def bulk_normalize(req: BulkRequest):
    out = []
    for t in req.texts[:1000]:
        r = normalize(NormalizeRequest(text=t, threshold=req.threshold))
        out.append(r.dict())
    return {"ok": True, "results": out}
