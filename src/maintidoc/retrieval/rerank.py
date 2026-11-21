"""Reranker multilingue : reclasse les candidats par pertinence question/passage."""
from __future__ import annotations

import os
from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


@lru_cache(maxsize=1)
def reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL, max_length=512)


def rerank(query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    """Reclasse les candidats qui ont un texte. Le score sert à ordonner, ce n'est pas une confiance."""
    cands = [h for h in hits if h.get("text")]
    if not cands:
        return hits[:top_n]
    scores = reranker().predict([(query, h["text"]) for h in cands])
    for h, s in zip(cands, scores):
        h["rerank_score"] = float(s)
    return sorted(cands, key=lambda h: h["rerank_score"], reverse=True)[:top_n]