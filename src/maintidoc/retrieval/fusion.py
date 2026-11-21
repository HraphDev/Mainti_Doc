"""Fusion RRF (Reciprocal Rank Fusion) de listes de résultats."""
from __future__ import annotations

from typing import Any

RRF_K = 60


def evidence_key(hit: dict[str, Any]):
    """Même preuve = même (document, page, figure). Sinon, l'id du point."""
    if hit.get("element_id"):
        return (hit["document_id"], hit["page"], hit["element_id"])
    return hit["id"]


def rrf(rankings: list[list[dict]], k: int = RRF_K, top_n: int = 5) -> list[dict]:
    scores: dict[Any, float] = {}
    merged: dict[Any, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = evidence_key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in merged:
                merged[key] = dict(hit)
            else:  # complète les champs manquants (ex. image_path d'un côté, texte de l'autre)
                for field, value in hit.items():
                    if merged[key].get(field) is None and value is not None:
                        merged[key][field] = value
    out = []
    for key in sorted(scores, key=scores.get, reverse=True)[:top_n]:
        item = merged[key]
        item["rrf_score"] = round(scores[key], 5)
        if item.get("image_path"):
            item["type"] = "figure"
        out.append(item)
    return out