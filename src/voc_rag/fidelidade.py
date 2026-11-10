"""Planilha para conferência manual de fidelidade (faithfulness, como no Ragas).

Para cada resposta "ok" da amostra, quebra o texto em frases, mostra as citações de cada frase
e o texto das avaliações citadas. Quem confere marca se a frase é sustentada (sim / parcial / não).
As evidências são buscadas de novo no índice com a mesma busca e os mesmos filtros: a busca é
determinística, então [R1], [R2]... são as mesmas que o modelo viu. Não chama o LLM.

Uso: python -m voc_rag.fidelidade  ->  eval/fidelidade.csv (abre no Excel)
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


if __name__ == "__main__":
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
        ev = retrieve(col, emb, r.busca, f, k=40, min_sim=0).evidences  # as primeiras são as mesmas de k=8
        linhas += linhas_da_resposta(r.id, qs[r.id]["pergunta"], r.resposta, ev)
    pd.DataFrame(linhas, columns=COLUNAS).to_csv("eval/fidelidade.csv", index=False, sep=";", encoding="utf-8-sig")
    print(f"{len(linhas)} frases de {res.pipe(amostra).shape[0]} respostas -> eval/fidelidade.csv")
