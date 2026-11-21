"""Métriques de recherche à pertinence binaire."""
from __future__ import annotations

import math


def matches(hit: dict, alt: dict) -> bool:
    if "page" in alt and hit.get("page") != alt["page"]:
        return False
    if "element_id" in alt and hit.get("element_id") != alt["element_id"]:
        return False
    if "element_prefix" in alt and not str(hit.get("element_id") or "").startswith(alt["element_prefix"]):
        return False
    if "contains_any" in alt:
        text = (hit.get("text") or "").lower()
        if not any(k.lower() in text for k in alt["contains_any"]):
            return False
    return True


def is_relevant(hit: dict, gold: list[dict]) -> bool:
    return any(matches(hit, alt) for alt in gold)


def metrics_for(hits: list[dict], gold: list[dict], n_relevant: int) -> dict:
    """n_relevant = nombre total de preuves pertinentes dans le corpus accessible."""
    rels = [1 if is_relevant(h, gold) else 0 for h in hits[:10]]
    recall5 = sum(rels[:5]) / n_relevant if n_relevant else 0.0
    mrr10 = next((1 / (i + 1) for i, r in enumerate(rels) if r), 0.0)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:5]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(n_relevant, 5)))
    return {"recall5": recall5, "mrr10": mrr10, "ndcg5": dcg / ideal if ideal else 0.0,
            "first_rank": next((i + 1 for i, r in enumerate(rels) if r), None)}