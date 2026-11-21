"""
Ingestion MaintiDoc v3 : PDF -> Qdrant (index texte + index visuel).
Tableaux ligne par ligne, jeu de questions exporté (non indexé), images décoratives ignorées.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pymupdf as fitz
from dotenv import load_dotenv
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct
from sentence_transformers import SentenceTransformer
from functools import lru_cache

load_dotenv()
TEXT_COL, VISUAL_COL = "maintidoc_text", "maintidoc_visual"
TEXT_MODEL = "BAAI/bge-m3"
CLIP_IMG_MODEL = "clip-ViT-B-32"
CHUNK_SIZE, MIN_IMAGE_PX = 500, 50
CAPTION_RE = re.compile(r"^(FIG-\d{2})\s*:")
NAMESPACE = uuid.UUID("6f1c2f0e-3b6a-4c1e-9a55-0d6f7b8d1a11")


def pid(*parts) -> str:
    return str(uuid.uuid5(NAMESPACE, "|".join(map(str, parts))))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# ------------------------------------------------------------ texte
def _in_bbox(rect, bbox) -> bool:
    cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def extract_paragraphs(page, exclude_bboxes=()) -> list[str]:
    """Paragraphes de la page, hors zones de tableaux (traités ligne par ligne)."""
    paras = []
    for b in page.get_text("blocks", sort=True):
        if b[6] != 0:
            continue
        if any(_in_bbox(fitz.Rect(b[:4]), bb) for bb in exclude_bboxes):
            continue
        text = _clean(b[4])
        if text:
            paras.append(text)
    return paras


def extract_tables(page) -> list[dict]:
    out = []
    try:
        found = page.find_tables().tables
    except Exception:
        return out
    for t_idx, t in enumerate(found, start=1):
        rows = []
        for row in t.rows:  # texte lu par zone de cellule : les espaces sont conservés
            rows.append([_clean(page.get_text("text", clip=fitz.Rect(c))) if c else ""
                         for c in row.cells])
        rows = [r for r in rows if any(r)]
        if len(rows) >= 2:
            key = "".join(rows[0]).lower().replace(" ", "")
            out.append({"index": t_idx, "bbox": tuple(t.bbox), "header": rows[0], "rows": rows[1:],
            "caption": _table_caption(page, t.bbox),
            "is_eval": "question" in key and "source" in key})
    return out

def _valid_caption(text: str, max_len: int = 80) -> bool:
    """Titre de tableau plausible. Rejette les fins de phrase, les légendes FIG-xx et les textes longs."""
    if not text or len(text) > max_len:
        return False
    if CAPTION_RE.match(text) or text.endswith("."):
        return False
    return text[0].isupper() or text[0].isdigit()


def _table_caption(page, bbox) -> str:
    """Bloc de texte juste au-dessus du tableau, gardé seulement s'il ressemble à un titre."""
    best = None
    for b in page.get_text("blocks"):
        if b[6] != 0:
            continue
        if b[3] <= bbox[1] + 2 and bbox[1] - b[3] < 60 and (best is None or b[3] > best[3]):
            best = b
    text = _clean(best[4]) if best else ""
    return text if _valid_caption(text) else ""

def table_row_records(tbl: dict, page_no: int, base: dict) -> list[tuple[str, dict]]:
    recs = []
    table_id = f"TAB-p{page_no}-t{tbl['index']}"
    for r_idx, row in enumerate(tbl["rows"], start=1):
        cells = {h: v for h, v in zip(tbl["header"], row) if v}
        row_text = " ; ".join(f"{h} : {v}" for h, v in cells.items())
        text = f"{tbl['caption']} | {row_text}" if tbl.get("caption") else row_text
        recs.append((text, {**base, "type": "table_row", "table_id": table_id,
                            "table_caption": tbl.get("caption", ""),
                            "element_id": f"{table_id}-r{r_idx}", "row_index": r_idx,
                            "row_key": row[0], "cells": cells, "text": text}))
    return recs


def split_long(text: str, size: int) -> list[str]:
    parts, current = [], ""
    for sent in re.split(r"(?<=[.!?;:])\s+", text):
        while len(sent) > size:
            cut = sent.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            if current:
                parts.append(current)
                current = ""
            parts.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if len(current) + len(sent) + 1 <= size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                parts.append(current)
            current = sent
    if current:
        parts.append(current)
    return parts


def chunk_paragraphs(paragraphs: list[str], size: int = CHUNK_SIZE) -> list[str]:
    chunks, current = [], ""
    for p in paragraphs:
        if len(p) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long(p, size))
        elif len(current) + len(p) + 1 <= size:
            current = f"{current} {p}".strip()
        else:
            chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


# ------------------------------------------------------------ images
def extract_images(doc, page, out_dir: Path, document_id: str, paragraphs: list[str]) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    found, seen = [], set()
    for idx, img in enumerate(page.get_images(full=True), start=1):
        if img[0] in seen:
            continue
        seen.add(img[0])
        info = doc.extract_image(img[0])
        if not info or min(info["width"], info["height"]) < MIN_IMAGE_PX:
            continue
        path = out_dir / f"{document_id}_p{page.number + 1}_img{idx}.{info['ext']}"
        path.write_bytes(info["image"])
        found.append({"path": path.as_posix(), "kind": "embedded"})
    if not found and any(CAPTION_RE.match(p) for p in paragraphs):  # schéma vectoriel
        path = out_dir / f"{document_id}_p{page.number + 1}_render.png"
        page.get_pixmap(dpi=150).save(path)
        found.append({"path": path.as_posix(), "kind": "page_render"})
    return found


def figure_description(paragraphs: list[str]) -> tuple[str | None, str]:
    for p in paragraphs:
        m = CAPTION_RE.match(p)
        if m:
            return m.group(1), p[:CHUNK_SIZE]
    return None, (paragraphs[0][:CHUNK_SIZE] if paragraphs else "")

    # ------------------------------------------------------------ OCR
@lru_cache(maxsize=1)
def ocr_reader():
    import easyocr
    return easyocr.Reader(["fr", "en"], gpu=True)


def ocr_image(path: str, min_conf: float = 0.4) -> str:
    """Texte lu dans une image. Donnée non fiable : jamais traité comme une instruction."""
    results = ocr_reader().readtext(path, detail=1, paragraph=False)
    words = [_clean(t) for _, t, conf in results if conf >= min_conf and len(_clean(t)) > 1]
    return " ; ".join(words)[:400]
# ------------------------------------------------------------ ingestion
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--equipment-id", required=True)
    ap.add_argument("--access-groups", nargs="+", required=True)
    ap.add_argument("--revision", default="1.0")
    ap.add_argument("--images-dir", default="data/processed/images")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--text-model", default=TEXT_MODEL)
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args()

    client = QdrantClient(url=os.environ["QDRANT_URL"], check_compatibility=False)
    for col in (TEXT_COL, VISUAL_COL):
        if not client.collection_exists(col):
            raise SystemExit(f"Collection {col} absente : lance scripts/init_collections.py")

    file_hash = sha256_file(args.pdf)
    doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=args.document_id))])
    same = Filter(must=[*doc_filter.must,
                        FieldCondition(key="source_sha256", match=MatchValue(value=file_hash))])
    if not args.force and client.count(TEXT_COL, count_filter=same, exact=True).count > 0:
        print("PDF inchangé : ingestion ignorée (--force pour réindexer).")
        return

    provenance = {
        "document_id": args.document_id, "revision": args.revision,
        "equipment_id": args.equipment_id, "access_groups": args.access_groups,
        "source_path": Path(args.pdf).as_posix(), "source_sha256": file_hash,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    doc = fitz.open(args.pdf)
    text_recs: list[tuple[str, dict]] = []
    visual_recs: list[tuple[str, dict]] = []
    eval_rows: list[dict] = []

    for page in doc:
        page_no = page.number + 1
        tables = extract_tables(page)
        paragraphs = extract_paragraphs(page, [t["bbox"] for t in tables])
        base = {**provenance, "page": page_no}

        for tbl in tables:
            if tbl["is_eval"]:  # jeu de questions : exporté pour l'évaluation, jamais indexé
                eval_rows.extend(dict(zip(tbl["header"], r)) for r in tbl["rows"])
            else:
                text_recs.extend(table_row_records(tbl, page_no, base))

        for i, chunk in enumerate(chunk_paragraphs(paragraphs)):
            text_recs.append((chunk, {**base, "type": "text", "chunk_index": i, "text": chunk}))

        for img in extract_images(doc, page, Path(args.images_dir), args.document_id, paragraphs):
            fig_id, desc = figure_description(paragraphs)
            if fig_id is None:
                 continue  # image décorative : non indexée
            ocr = "" if args.no_ocr else ocr_image(img["path"])
            indexed = f"{desc} Texte du schéma : {ocr}" if ocr else desc
            payload = {**base, "type": "image", "element_id": fig_id, "image_path": img["path"],
           "image_kind": img["kind"], "description": desc, "ocr_text": ocr}
            visual_recs.append((img["path"], payload))
            text_recs.append((indexed, {**payload, "type": "figure_caption", "text": indexed}))
    if eval_rows:
        Path("data/eval").mkdir(parents=True, exist_ok=True)
        Path(f"data/eval/{args.document_id}_questions.json").write_text(
            json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{len(eval_rows)} questions d'évaluation exportées dans data/eval (non indexées).")

    text_model = SentenceTransformer(args.text_model)
    tvec = text_model.encode([t for t, _ in text_recs], batch_size=16,
                             normalize_embeddings=True, show_progress_bar=True)
    text_points = [
        PointStruct(id=pid(p["document_id"], p["page"], p["type"],
                           p.get("chunk_index", p.get("image_path", p.get("element_id", "")))),
                    vector=v.tolist(), payload={**p, "text_model": args.text_model})
        for (_, p), v in zip(text_recs, tvec)
    ]

    visual_points = []
    if visual_recs:
        clip = SentenceTransformer(CLIP_IMG_MODEL)
        imgs = [Image.open(path).convert("RGB") for path, _ in visual_recs]
        ivec = clip.encode(imgs, batch_size=8, normalize_embeddings=True)
        visual_points = [
            PointStruct(id=pid(p["document_id"], p["page"], "image", p["image_path"]),
                        vector=v.tolist(), payload={**p, "image_model": CLIP_IMG_MODEL})
            for (_, p), v in zip(visual_recs, ivec)
        ]

    for col in (TEXT_COL, VISUAL_COL):
        client.delete(col, points_selector=FilterSelector(filter=doc_filter))
    client.upsert(TEXT_COL, points=text_points)
    if visual_points:
        client.upsert(VISUAL_COL, points=visual_points)

    print(f"OK : {len(text_points)} points texte, {len(visual_points)} points visuels "
          f"(document {args.document_id}, rev {args.revision}).")


if __name__ == "__main__":
    main()

