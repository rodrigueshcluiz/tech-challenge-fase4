"""Testes do cliente Groq sem internet: a função de envio é trocada por uma falsa."""
import pytest

from voc_rag import llm as L


@pytest.fixture(autouse=True)
def chave_falsa(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # sem .env de verdade
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER", "GROQ_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "teste")


def test_groq_monta_mensagens_e_le_resposta():
    enviados = []
    g = L.GroqLLM(post=lambda p: enviados.append(p) or {"choices": [{"message": {"content": "ok [R1]"}}]})
    assert g("regras", "pergunta") == "ok [R1]"
    p = enviados[0]
    assert p["model"] == L.GROQ_MODELO_PADRAO
    assert p["messages"] == [{"role": "system", "content": "regras"}, {"role": "user", "content": "pergunta"}]
    assert "response_format" not in p


def test_groq_modo_json():
    g = L.GroqLLM(post=lambda p: {"choices": [{"message": {"content": "{}"}}]})
    assert g.payload("s", "p", json_mode=True)["response_format"] == {"type": "json_object"}


def test_escolha_do_provedor(monkeypatch):
    assert isinstance(L.make_llm(), L.GroqLLM)  # tem GROQ_API_KEY
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(L, "GeminiLLM", lambda: "gemini")
    assert L.make_llm() == "gemini"


def test_groq_espera_e_tenta_de_novo_no_limite_por_minuto():
    respostas = [RuntimeError("Groq HTTP 429: Please try again in 3.6s"), {"choices": [{"message": {"content": "ok"}}]}]

    def post(p):
        r = respostas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    esperas = []
    assert L.GroqLLM(post=post)("s", "p", dormir=esperas.append) == "ok"
    assert esperas == [4.6]


def test_groq_nao_insiste_em_outros_erros():
    def post(p):
        raise RuntimeError("Groq HTTP 404: model_not_found")

    with pytest.raises(RuntimeError, match="404"):
        L.GroqLLM(post=post)("s", "p", dormir=lambda s: None)
