"""Mede a busca semântica sem a reescrita da pergunta (não chama o LLM).

Para cada pergunta respondida na avaliação, busca de novo com a mesma quantidade de evidências e os
mesmos filtros, de duas formas: com a consulta reescrita gravada em eval/resultados.csv (deve
reproduzir a precisão do RAG) e com a pergunta original. Assim dá para separar o efeito da
reescrita do efeito da busca semântica.

Uso: python -m voc_rag.sem_reescrita  ->  eval/sem_reescrita.csv
"""
import json

import pandas as pd

from voc_rag.evaluate import precision


def medir(res, qs, col, emb):
    from voc_rag.filters import Filters
    from voc_rag.retrieve import retrieve

    linhas = []
    for _, r in res[res.status == "ok"].iterrows():
        q, k = qs[r.id], int(r.n_evid)
        f = Filters(**q.get("filtros", {}))
        reescrita = retrieve(col, emb, r.busca, f, k=k, min_sim=0).closest
        original = retrieve(col, emb, q["pergunta"], f, k=k, min_sim=0).closest
        linhas.append({"id": r.id, "k": k, "precisao_rag_avaliacao": r.precisao_rag,
                       "densa_reescrita_agora": precision(reescrita, q["relevante"]),
                       "densa_pergunta_original": precision(original, q["relevante"])})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    from voc_rag.cli import CHROMA, REVISAO_E5
    from voc_rag.embeddings import E5Embedder
    from voc_rag.index import open_index

    res = pd.read_csv("eval/resultados.csv")
    qs = {q["id"]: q for q in json.load(open("eval/perguntas.json", encoding="utf-8"))}
    col, emb = open_index(CHROMA), E5Embedder(revision=REVISAO_E5)
    t = medir(res, qs, col, emb)
    t.to_csv("eval/sem_reescrita.csv", index=False)
    print(t.round(3).to_string(index=False))
    print("\nMédias:", t.drop(columns=["id", "k"]).mean().round(3).to_dict())

    # Q15: o que o modelo viu (k=8) e o que a planilha antiga mostrou (k=40)
    q = qs["Q15"]; busca = res.set_index("id").loc["Q15", "busca"]
    from voc_rag.filters import Filters
    from voc_rag.retrieve import retrieve
    for k in (8, 40):
        ev = retrieve(col, emb, busca, Filters(**q.get("filtros", {})), k=k, min_sim=0).closest[:9]
        print(f"\nQ15 com k={k}:")
        for i, e in enumerate(ev, 1):
            print(f"  R{i}: {e.text[:90]}")
