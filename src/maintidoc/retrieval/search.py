"""
Recherche MaintiDoc 2.1 : texte + image, filtrées par groupes d'accès.
Ex : uv run python scripts/search.py "Que signifie MPIO ?" --groups maintenance
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from sentence_transformers import SentenceTransformer
from maintidoc.retrieval.fusion import rrf

load_dotenv()
TEXT_COL, VISUAL_COL = "maintidoc_text", "maintidoc_visual"
TEXT_MODEL = "BAAI/bge-m3"
CLIP_TXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    return QdrantClient(url=os.environ["QDRANT_URL"], check_compatibility=False)


@lru_cache(maxsize=1)
def text_model() -> SentenceTransformer:
    return SentenceTransformer(TEXT_MODEL)


@lru_cache(maxsize=1)
def clip_text_model() -> SentenceTransformer:
    return SentenceTransformer(CLIP_TXT_MODEL)


def access_filter(user_groups: list[str]) -> Filter:
    """Ne garde que les points dont au moins un groupe est celui de l'utilisateur."""
    return Filter(must=[FieldCondition(key="access_groups", match=MatchAny(any=user_groups))])


def _to_hit(p) -> dict:
    pl = p.payload
    return {"id": str(p.id), "score": getattr(p, "score", 0.0), "type": pl["type"],
            "document_id": pl["document_id"], "revision": pl["revision"], "page": pl["page"],
            "element_id": pl.get("element_id"), "chunk_index": pl.get("chunk_index"),
            "table_id": pl.get("table_id"), "row_key": pl.get("row_key"),
            "text": pl.get("text") or pl.get("description"), "image_path": pl.get("image_path")}


def search_text(query: str, user_groups: list[str], k: int = 10) -> list[dict]:
    vec = text_model().encode(query, normalize_embeddings=True).tolist()
    res = client().query_points(TEXT_COL, query=vec, query_filter=access_filter(user_groups),
                                limit=k, with_payload=True)
    return [_to_hit(p) for p in res.points]


def search_visual(query: str, user_groups: list[str], k: int = 5) -> list[dict]:
    vec = clip_text_model().encode(query, normalize_embeddings=True).tolist()
    res = client().query_points(VISUAL_COL, query=vec, query_filter=access_filter(user_groups),
                                limit=k, with_payload=True)
    return [_to_hit(p) for p in res.points]


def hybrid_search(query: str, user_groups: list[str], k_text: int = 10,
                  k_visual: int = 5, top_n: int = 5) -> list[dict]:
    """Recherche texte + image (filtrées par droits), fusionnées par RRF."""
    return rrf([search_text(query, user_groups, k_text),
                search_visual(query, user_groups, k_visual)], top_n=top_n)


def search(query: str, user_groups: list[str], use_reranker: bool = True,
           top_n: int = 5, candidates: int = 10) -> list[dict]:
    """Pipeline complet : hybride + RRF, puis reranker (optionnel)."""
    fused = hybrid_search(query, user_groups, top_n=candidates)
    if not use_reranker:
        return fused[:top_n]
    from maintidoc.retrieval.rerank import rerank
    return rerank(query, fused, top_n)