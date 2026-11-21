import json
import sys
from pathlib import Path

from maintidoc.evaluation.retrieval_metrics import is_relevant
from maintidoc.retrieval import search as S

qid = sys.argv[1]
questions = json.loads(Path("data/eval/SAN-MAN-001_questions.json").read_text(encoding="utf-8"))
gold = {g["id"]: g["gold"] for g in
        json.loads(Path("data/eval/SAN-MAN-001_gold.json").read_text(encoding="utf-8"))}
row = next(r for r in questions if list(r.values())[0] == qid)
question = list(row.values())[1]
print("Question :", question)

print("\nPreuves que le gold considère pertinentes :")
pts, _ = S.client().scroll(S.TEXT_COL, limit=200, with_payload=True)
for h in map(S._to_hit, pts):
    if is_relevant(h, gold[qid]):
        print(f"  p.{h['page']} | {h['type']} | {h.get('element_id')} | {(h['text'] or '')[:100]}")

print("\nTop 10 de la recherche texte :")
for i, h in enumerate(S.search_text(question, ["maintenance"], k=10), 1):
    flag = "OK" if is_relevant(h, gold[qid]) else "  "
    print(f"{i:2} {flag} p.{h['page']} | {h['type']} | {h.get('element_id')} | {(h['text'] or '')[:80]}")