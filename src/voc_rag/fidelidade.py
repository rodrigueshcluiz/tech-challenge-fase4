"""Planilha para conferência manual de fidelidade (faithfulness, como no Ragas).

Para cada resposta "ok" da amostra, quebra o texto em frases, mostra as citações de cada frase
e o texto das avaliações citadas. Quem confere marca se a frase é sustentada (sim / parcial / não).
As evidências são buscadas de novo no índice com a mesma busca, os mesmos filtros e o mesmo k da
resposta, então [R1], [R2]... são as mesmas que o modelo viu. Não chama o LLM.

Uso: python -m voc_rag.fidelidade             ->  gera eval/fidelidade.csv (abre no Excel)
     python -m voc_rag.fidelidade --calcular  ->  lê a planilha preenchida e calcula a fidelidade

Fidelidade = (sim + 0,5 x parcial) / (sim + parcial + nao). Frases marcadas "na" (título, ligação,
frase sem afirmação) ficam fora da conta; frases em branco são contadas como pendentes.
"""
import json
import re

import pandas as pd

from voc_rag.generate import cited_ids, mask_names

COLUNAS = ["id", "pergunta", "n_frase", "frase", "citacoes", "evidencias_citadas", "sustentada (sim/parcial/nao)", "comentario"]


def frases(texto: str) -> list[str]:
    """Quebra por linha e por ponto final; descarta títulos sem conteúdo (ex.: "**Atraso**")."""
    partes = []
    for linha in texto.splitlines():
        linha = linha.strip(" -*•\t|")
        partes += [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ*])", linha) if p.strip()]
    return [p for p in partes if len(re.sub(r"[\W_]", "", p)) > 15]


def linhas_da_resposta(qid, pergunta, resposta, evidencias):
    out = []
    for n, f in enumerate(frases(resposta), 1):
        ids = sorted(cited_ids(f))
        textos = " || ".join(f"R{i}: {mask_names(evidencias[i - 1].text)}" for i in ids if 1 <= i <= len(evidencias))
        out.append(dict(zip(COLUNAS, [qid, pergunta, n, f, ", ".join(f"R{i}" for i in ids), textos, "", ""])))
    return out


def amostra(res: pd.DataFrame, n=10) -> pd.DataFrame:
    ok = res[(res.status == "ok") & res.tipo.isin(["no_tema", "filtro"])]
    return ok.iloc[:: max(1, len(ok) // n)].head(n)  # espalhada pelas perguntas, sempre a mesma


MARCAS = {"sim": "sim", "parcial": "parcial", "nao": "nao", "não": "nao", "na": "na", "n/a": "na"}


def calcular(planilha: pd.DataFrame) -> dict:
    """Conta as marcações e calcula a fidelidade geral e por resposta."""
    m = planilha[COLUNAS[6]].fillna("").astype(str).str.strip().str.lower().map(lambda x: MARCAS.get(x, "" if x == "" else "invalida"))
    conta = m.value_counts().to_dict()
    validas = m.isin(["sim", "parcial", "nao"])
    pontos = m.map({"sim": 1.0, "parcial": 0.5, "nao": 0.0})
    por_resposta = pontos[validas].groupby(planilha.loc[validas, "id"]).mean().round(3).to_dict()
    total = int(validas.sum())
    return {"sim": conta.get("sim", 0), "parcial": conta.get("parcial", 0), "nao": conta.get("nao", 0),
            "na": conta.get("na", 0), "pendentes": conta.get("", 0), "invalidas": conta.get("invalida", 0),
            "fidelidade": round(float(pontos[validas].mean()), 3) if total else None, "por_resposta": por_resposta}


if __name__ == "__main__":
    import sys

    if "--calcular" in sys.argv:
        r = calcular(pd.read_csv("eval/fidelidade.csv", sep=";", encoding="utf-8-sig"))
        print(f"sim {r['sim']} | parcial {r['parcial']} | nao {r['nao']} | na {r['na']} | pendentes {r['pendentes']} | inválidas {r['invalidas']}")
        print(f"Fidelidade: {r['fidelidade']}")
        for q, v in r["por_resposta"].items():
            print(f"  {q}: {v}")
        sys.exit(0)

    from voc_rag.cli import CHROMA
    from voc_rag.embeddings import E5Embedder
    from voc_rag.filters import Filters
    from voc_rag.index import open_index
    from voc_rag.retrieve import retrieve

    res = pd.read_csv("eval/resultados.csv")
    qs = {q["id"]: q for q in json.load(open("eval/perguntas.json", encoding="utf-8"))}
    col, emb = open_index(CHROMA), E5Embedder()
    linhas = []
    for _, r in amostra(res).iterrows():
        f = Filters(**qs[r.id].get("filtros", {}))
        ev = retrieve(col, emb, r.busca, f, k=int(r.n_evid), min_sim=0).evidences  # o mesmo k da resposta: com k maior a busca pode mudar a ordem
        linhas += linhas_da_resposta(r.id, qs[r.id]["pergunta"], r.resposta, ev)
    pd.DataFrame(linhas, columns=COLUNAS).to_csv("eval/fidelidade.csv", index=False, sep=";", encoding="utf-8-sig")
    print(f"{len(linhas)} frases de {res.pipe(amostra).shape[0]} respostas -> eval/fidelidade.csv")
