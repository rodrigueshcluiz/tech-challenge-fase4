"""Teste da tela (Streamlit) com índice e modelos falsos: abre, responde e mostra as evidências."""
import os
import tempfile

import pandas as pd
import pytest

from tests.test_rag import FakeEmbedder, FakeLLM, _doc
from voc_rag.index import build_index


def recursos_falsos():
    d = pd.DataFrame([
        _doc("a1", "Não recebi o produto", 1, "nao_entregue", "telefonia", True),
        _doc("a2", "Produto chegou atrasado", 2, "atrasado", "beleza_saude"),
        _doc("a3", "Ótimo produto, ótima qualidade", 5, cats="telefonia"),
    ])
    d["days_late"] = d["days_late"].astype("Int64")
    emb = FakeEmbedder()
    col = build_index(d, emb, tempfile.mkdtemp())
    llm = FakeLLM(rewrite={"in_scope": True, "broad": False, "query": "não recebi o produto"})
    llm.model = "falso"
    return col, emb, llm, d


def test_tela_responde_com_evidencias(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("VOC_RAG_TESTE", "1")
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.text_input[0].input("Quem não recebeu o pedido?").run()
    assert not at.exception
    assert any("Resposta baseada" in s.value for s in at.success)
    assert at.dataframe[0].value["Ref."].tolist()[0] == "R1"
