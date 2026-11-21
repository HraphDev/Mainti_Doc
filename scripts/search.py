import argparse

from maintidoc.retrieval.search import search


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()

    results = search(a.query, a.groups, use_reranker=not a.no_rerank, top_n=a.n)
    if not results:
        print("Aucun résultat (aucun document accessible pour ces groupes).")
        return

    for h in results:
        score = h.get("rerank_score", h.get("rrf_score", 0))
        print(f"{score:8.4f} | p.{h['page']} | {h['type']:<14} | "
              f"{h.get('element_id') or '-':<7} | {(h.get('text') or '')[:90]}")
        if h.get("image_path"):
            print(f"           image : {h['image_path']}")


if __name__ == "__main__":
    main()