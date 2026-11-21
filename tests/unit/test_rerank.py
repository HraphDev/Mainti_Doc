from maintidoc.retrieval import rerank as r


class FakeReranker:
    def predict(self, pairs):
        return [0.1, 0.9]


def test_rerank_reordonne(monkeypatch):
    monkeypatch.setattr(r, "reranker", lambda: FakeReranker())
    hits = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    assert r.rerank("q", hits, top_n=2)[0]["id"] == "b"