"""Testes das métricas da etapa 5."""
import pandas as pd

from voc_rag.evaluate import bm25_search, precision, refusal_ok, suggest_threshold
from voc_rag.retrieve import Evidence


def ev(text):
    return Evidence(text, 0.9, 1, "no_prazo", "", "2018-01-10", ["x"])


def test_precision_conta_evidencias_que_batem_com_o_padrao():
    assert precision([ev("Não chegou"), ev("Produto ótimo")], r"cheg|entreg") == 0.5
    assert precision([], r"cheg") is None  # sem evidência não há precisão


def test_recusa_correta():
    assert refusal_ok("sem_evidencia", "fora_do_escopo")  # recusar por qualquer motivo é aceitável
    assert refusal_ok("fora_do_escopo", "sem_evidencia")
    assert not refusal_ok("ok", "sem_evidencia")
    assert refusal_ok("ok", "ok") and not refusal_ok("sem_evidencia", "ok")


def test_limite_sugerido_fica_entre_os_grupos():
    # perguntas no tema têm semelhança máxima alta; sem evidência, baixa
    t = suggest_threshold(no_tema=[0.88, 0.90, 0.86], sem_evidencia=[0.80, 0.82])
    assert 0.82 < t < 0.86


def test_bm25_respeita_filtro_e_ordena_por_relevancia():
    docs = pd.DataFrame({"review_id": ["a", "b", "c"], "text": ["não chegou o produto", "produto ótimo", "não chegou nada ainda"],
                         "score": [1, 5, 5]})
    top = bm25_search(docs[docs.score <= 2], "não chegou", k=5)
    assert list(top["review_id"]) == ["a"]


def test_run_registra_bm25_com_a_pergunta_reescrita_e_a_versao_do_prompt(tmp_path):
    from tests.test_rag import FakeEmbedder, FakeLLM, _doc
    from voc_rag.evaluate import run
    from voc_rag.generate import PROMPT_VERSION
    from voc_rag.index import build_index

    docs = pd.DataFrame([_doc("a1", "Não recebi o produto", 1, "nao_entregue"), _doc("a2", "Adorei, chegou rápido", 5)])
    emb = FakeEmbedder()
    col = build_index(docs, emb, tmp_path / "c")
    llm = FakeLLM(rewrite={"in_scope": True, "broad": False, "query": "não recebi o produto"})
    q = [{"id": "Q1", "tipo": "no_tema", "esperado": "ok", "pergunta": "Ficou faltando entrega?", "relevante": "receb"}]
    linha = run(q, col, emb, llm, docs, min_sim=0.0).iloc[0]
    assert linha["precisao_bm25"] is None or pd.isna(linha["precisao_bm25"])  # pergunta original não tem as palavras
    assert linha["precisao_bm25_reescrita"] == 1.0  # com a reescrita, o BM25 acha
    assert linha["prompt_version"] == PROMPT_VERSION


def test_precisao_usa_todas_as_evidencias_e_marca_uso_das_estatisticas(tmp_path):
    from tests.test_rag import FakeEmbedder, FakeLLM, _doc
    from voc_rag.evaluate import run
    from voc_rag.index import build_index

    textos = [f"não recebi o pedido {i}" for i in range(6)] + [f"adorei a cor {i}" for i in range(6)]
    docs = pd.DataFrame([_doc(f"a{i}", t, 1 if i < 6 else 5) for i, t in enumerate(textos)])
    emb = FakeEmbedder()
    col = build_index(docs, emb, tmp_path / "c")
    llm = FakeLLM(rewrite={"in_scope": True, "broad": True, "query": "não recebi o pedido"},
                  answer_text="Panorama: 50% sem entrega [Estatísticas]. Não chegou [R1].")
    q = [{"id": "Q1", "tipo": "no_tema", "esperado": "ok", "pergunta": "Principais problemas?", "relevante": "receb"}]
    linha = run(q, col, emb, llm, docs, min_sim=0.0, k=2).iloc[0]
    assert linha["n_evid"] == 12  # ampla: busca até k_broad, não para em k=2
    assert linha["precisao_rag"] == 0.5  # mede as 12 evidências usadas, não só as 2 primeiras
    assert linha["usou_estatisticas"] == True  # noqa: E712
    assert linha["recusa_por"] == ""


def test_recusa_pela_trava_de_semelhanca_e_registrada(tmp_path):
    from tests.test_rag import FakeEmbedder, FakeLLM, _doc
    from voc_rag.evaluate import report, run
    from voc_rag.index import build_index

    docs = pd.DataFrame([_doc("a1", "adorei a cor", 5)])
    emb = FakeEmbedder()
    col = build_index(docs, emb, tmp_path / "c")
    llm = FakeLLM(rewrite={"in_scope": True, "broad": False, "query": "produto quebrado"})
    q = [{"id": "Q1", "tipo": "sem_evidencia", "esperado": "sem_evidencia", "pergunta": "Chegou quebrado?"}]
    res = run(q, col, emb, llm, docs, min_sim=0.5)
    assert res.iloc[0]["recusa_por"] == "semelhanca"
    assert not llm.generation_calls  # a trava barata não chama o LLM
