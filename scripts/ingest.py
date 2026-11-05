"""
Ingestion MaintiDoc v2 : PDF -> Qdrant serveur (index texte + index visuel).
Ex : uv run python scripts/ingest.py data/raw/Manuel.pdf --document-id SAN-MAN-001 \
       --equipment-id SAN-LAB-2000 --access-groups maintenance stockage
"""
from __future__ import annotations

import argparse
import hashlib
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

load_dotenv()
TEXT_COL, VISUAL_COL = "maintidoc_text", "maintidoc_visual"
TEXT_MODEL = "BAAI/bge-m3"                                   # 1024 dim
CLIP_IMG_MODEL = "clip-ViT-B-32"                             # 512 dim (images)
CLIP_TXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"  # 512 dim (requêtes FR)
CHUNK_SIZE, MIN_IMAGE_PX = 500, 50
FIG_RE = re.compile(r"\bFIG-\d{2}\b")
NAMESPACE = uuid.UUID("6f1c2f0e-3b6a-4c1e-9a55-0d6f7b8d1a11")


def pid(*parts) -> str:
    return str(uuid.uuid5(NAMESPACE, "|".join(map(str, parts))))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------ texte
def extract_paragraphs(page: fitz.Page) -> list[str]:
    paras = []
    for b in page.get_text("blocks", sort=True):
        if b[6] != 0:
            continue
        text = re.sub(r"\s+", " ", b[4]).strip()
        if text:
            paras.append(text)
    return paras


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
def extract_images(doc, page, out_dir: Path, document_id: str) -> list[dict]:
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
        found.append({"path": str(path), "kind": "embedded"})
    if not found and FIG_RE.search(page.get_text()):  # schéma vectoriel : rendu de la page
        path = out_dir / f"{document_id}_p{page.number + 1}_render.png"
        page.get_pixmap(dpi=150).save(path)
        found.append({"path": str(path), "kind": "page_render"})
    return found


def figure_description(paragraphs: list[str]) -> tuple[str | None, str]:
    for p in paragraphs:
        m = FIG_RE.search(p)
        if m:
            return m.group(0), p[:CHUNK_SIZE]
    return None, (paragraphs[0][:CHUNK_SIZE] if paragraphs else "")


# ------------------------------------------------------------ ingestion
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--equipment-id", required=True)
    ap.add_argument("--access-groups", nargs="+", required=True)
    ap.add_argument("--revision", default="1.0")
    ap.add_argument("--images-dir", default="data/images")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--text-model", default=TEXT_MODEL)
    args = ap.parse_args()

    client = QdrantClient(url=os.environ["QDRANT_URL"])
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
        "source_path": str(Path(args.pdf).as_posix()), "source_sha256": file_hash,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    doc = fitz.open(args.pdf)
    text_recs: list[tuple[str, dict]] = []
    visual_recs: list[tuple[str, dict]] = []  # (chemin image, payload)

    for page in doc:
        page_no = page.number + 1
        paragraphs = extract_paragraphs(page)
        base = {**provenance, "page": page_no}

        for i, chunk in enumerate(chunk_paragraphs(paragraphs)):
            text_recs.append((chunk, {**base, "type": "text", "chunk_index": i, "text": chunk}))

        for img in extract_images(doc, page, Path(args.images_dir), args.document_id):
            fig_id, desc = figure_description(paragraphs)
            payload = {**base, "type": "image", "element_id": fig_id, "image_path": img["path"],
                       "image_kind": img["kind"], "description": desc}
            visual_recs.append((img["path"], payload))
            # la description de la figure est aussi cherchable via l'index texte
            text_recs.append((desc, {**payload, "type": "figure_caption", "text": desc}))

    # --- encodage
    text_model = SentenceTransformer(args.text_model)
    tvec = text_model.encode([t for t, _ in text_recs], batch_size=16,
                             normalize_embeddings=True, show_progress_bar=True)
    text_points = [
        PointStruct(id=pid(p["document_id"], p["page"], p["type"], p.get("chunk_index", p["image_path"] if "image_path" in p else "")),
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

    # --- mise à jour propre : suppression des anciens points du document, puis insertion
    for col in (TEXT_COL, VISUAL_COL):
        client.delete(col, points_selector=FilterSelector(filter=doc_filter))
    client.upsert(TEXT_COL, points=text_points)
    if visual_points:
        client.upsert(VISUAL_COL, points=visual_points)

    print(f"OK : {len(text_points)} points texte, {len(visual_points)} points visuels "
          f"(document {args.document_id}, rev {args.revision}).")


if __name__ == "__main__":
    main()