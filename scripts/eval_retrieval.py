"""Évaluation de la recherche : texte seul, hybride, hybride + reranker."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from maintidoc.evaluation.retrieval_metrics import is_relevant, metrics_for
from maintidoc.retrieval import search as S
from maintidoc.retrieval.fusion import evidence_key

DOC = "SAN-MAN-001"
CONFIGS = {
    "texte_seul": lambda q, g: S.search_text(q, g, k=10),
    "hybride": lambda q, g: S.search(q, g, use_reranker=False, top_n=10, candidates=15),
    "hybride+reranker": lambda q, g: S.search(q, g, use_reranker=True, top_n=10, candidates=15),
}


def corpus_units(groups: list[str]) -> list[dict]:
    """Toutes les preuves accessibles (image + légende comptent pour une seule)."""
    hits = []
    for col in (S.TEXT_COL, S.VISUAL_COL):
        offset = None
        while True:
            pts, offset = S.client().scroll(col, scroll_filter=S.access_filter(groups), limit=100,
                                            offset=offset, with_payload=True)
            hits += [S._to_hit(p) for p in pts]
            if offset is None:
                break
    units: dict = {}
    for h in hits:
        units.setdefault(evidence_key(h), h)
    return list(units.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["maintenance"])
    a = ap.parse_args()

    questions = json.loads(Path(f"data/eval/{DOC}_questions.json").read_text(encoding="utf-8"))
    gold = {g["id"]: g["gold"] for g in json.loads(
        Path(f"data/eval/{DOC}_gold.json").read_text(encoding="utf-8"))}
    units = corpus_units(a.groups)
    print(f"{len(units)} preuves accessibles dans le corpus.\n")

    report = {}
    for name, fn in CONFIGS.items():
        fn("test", a.groups)  # préchauffage : le chargement des modèles n'est pas compté
        rows = []
        for row in questions:
            qid, question = list(row.values())[:2]
            g = gold[qid]
            n_rel = sum(is_relevant(u, g) for u in units)
            if n_rel == 0:
                print(f"! {qid} : aucune preuve du corpus ne correspond au fichier gold.")
            t0 = time.perf_counter()
            hits = fn(question, a.groups)
            dt = time.perf_counter() - t0
            rows.append({"id": qid, "question": question, "n_relevant": n_rel,
                         "latency_s": round(dt, 3), **metrics_for(hits, g, n_rel),
                         "top3": [{"page": h["page"], "element_id": h.get("element_id"),
                                   "type": h["type"]} for h in hits[:3]]})
        report[name] = {
            "recall@5": round(statistics.mean(r["recall5"] for r in rows), 3),
            "ndcg@5": round(statistics.mean(r["ndcg5"] for r in rows), 3),
            "mrr@10": round(statistics.mean(r["mrr10"] for r in rows), 3),
            "latence_mediane_s": round(statistics.median(r["latency_s"] for r in rows), 3),
            "questions": rows,
        }

    print(f"{'configuration':<20} {'Recall@5':>9} {'nDCG@5':>8} {'MRR@10':>8} {'latence(s)':>11}")
    for name, r in report.items():
        print(f"{name:<20} {r['recall@5']:>9} {r['ndcg@5']:>8} {r['mrr@10']:>8} {r['latence_mediane_s']:>11}")

    print("\nRang de la 1re source attendue (- = absente du top 10) :")
    print(f"{'':<6}" + "".join(f"{n:>20}" for n in report))
    for i, row in enumerate(questions):
        qid = list(row.values())[0]
        print(f"{qid:<6}" + "".join(f"{str(report[n]['questions'][i]['first_rank'] or '-'):>20}"
                                    for n in report))

    out = Path("data/eval/results_retrieval.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats détaillés : {out.as_posix()}")


if __name__ == "__main__":
    main()