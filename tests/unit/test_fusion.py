from maintidoc.retrieval.fusion import rrf


def test_preuve_trouvee_par_texte_et_image_passe_en_tete():
    text = [
        {"id": "a", "document_id": "D", "page": 3, "element_id": None, "type": "text"},
        {"id": "b", "document_id": "D", "page": 3, "element_id": "FIG-02", "type": "figure_caption", "text": "FIG-02 ..."},
    ]
    visual = [{"id": "c", "document_id": "D", "page": 3, "element_id": "FIG-02",
               "type": "image", "image_path": "x.png"}]
    out = rrf([text, visual])
    assert out[0]["element_id"] == "FIG-02"
    assert out[0]["image_path"] == "x.png" and out[0]["text"] == "FIG-02 ..."
    assert out[0]["type"] == "figure"