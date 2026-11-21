"""Citations : schéma Pydantic et validation contre les preuves réellement récupérées."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: str
    page: int = Field(ge=1)
    element_id: str


def citation_of(hit: dict) -> Citation:
    eid = hit.get("element_id") or f"text-p{hit['page']}-c{hit.get('chunk_index') or 0}"
    return Citation(document_id=hit["document_id"], page=hit["page"], element_id=eid)


def validate_citations(citations: list[Citation], retrieved: list[dict]):
    """Valide uniquement les citations qui pointent vers un élément réellement récupéré.
    Retourne (valides, invalides)."""
    allowed = {citation_of(h) for h in retrieved}
    valid = [c for c in citations if c in allowed]
    return valid, [c for c in citations if c not in allowed]