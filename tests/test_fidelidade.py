"""Testes da planilha de fidelidade."""
import pandas as pd

from voc_rag.fidelidade import amostra, frases, linhas_da_resposta
from voc_rag.retrieve import Evidence


def test_frases_quebra_por_linha_e_ponto_e_ignora_titulos():
    t = "**Atraso**\n- Vários pedidos chegaram depois do prazo [R1]. Alguns nunca chegaram [R2].\n- Produto veio com defeito [R3]"
    assert frases(t) == ["Vários pedidos chegaram depois do prazo [R1].", "Alguns nunca chegaram [R2].", "Produto veio com defeito [R3]"]


def test_cada_frase_leva_o_texto_das_avaliacoes_citadas():
    ev = [Evidence(f"texto {i}", 0.9, 1, "no_prazo", "", "2018-01-10", [f"r{i}"]) for i in (1, 2)]
    l = linhas_da_resposta("Q1", "p?", "Muitos não receberam o pedido (R1, R2).", ev)
    assert l[0]["citacoes"] == "R1, R2" and l[0]["evidencias_citadas"] == "R1: texto 1 || R2: texto 2"


def test_amostra_so_de_respostas_validas_e_estavel():
    res = pd.DataFrame({"id": [f"Q{i}" for i in range(30)], "status": ["ok"] * 25 + ["sem_evidencia"] * 5,
                        "tipo": ["no_tema"] * 20 + ["filtro"] * 5 + ["sem_evidencia"] * 5})
    a = amostra(res)
    assert len(a) == 10 and set(a.status) == {"ok"} and a.equals(amostra(res))
