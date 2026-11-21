from maintidoc.evaluation.retrieval_metrics import metrics_for


def test_source_en_rang_1():
    hits = [{"page": 3, "text": "MPIO : chemins"}, {"page": 1, "text": "autre"}]
    m = metrics_for(hits, [{"page": 3, "contains_any": ["mpio"]}], n_relevant=1)
    assert m["mrr10"] == 1.0 and m["ndcg5"] == 1.0 and m["recall5"] == 1.0


def test_source_absente():
    m = metrics_for([{"page": 1, "text": "x"}], [{"page": 3}], n_relevant=1)
    assert m["mrr10"] == 0.0 and m["first_rank"] is None