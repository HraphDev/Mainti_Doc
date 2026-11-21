from maintidoc.answering.citations import Citation, validate_citations

HITS = [{"document_id": "D", "page": 3, "element_id": "FIG-02"}]


def test_citation_recuperee_valide():
    valid, invalid = validate_citations([Citation(document_id="D", page=3, element_id="FIG-02")], HITS)
    assert len(valid) == 1 and not invalid


def test_citation_inventee_rejetee():
    valid, invalid = validate_citations([Citation(document_id="D", page=18, element_id="FIG-99")], HITS)
    assert not valid and len(invalid) == 1