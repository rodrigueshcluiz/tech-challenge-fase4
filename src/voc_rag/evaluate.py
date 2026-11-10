"""Etapa 5: avaliação com 30 perguntas fixas (eval/perguntas.json).

Mede, para cada pergunta: se o sistema respondeu ou recusou como esperado, a precisão das
evidências (fração que bate com o padrão de relevância da pergunta) e a mesma precisão para
uma busca por palavra (BM25), que serve de baseline. Também sugere o limite de semelhança.

Uso: python -m voc_rag.evaluate   (grava eval/resultados.csv e eval/resultados.md)
"""
import json
import re
import time

import pandas as pd

RECUSAS = {"fora_do_escopo", "sem_evidencia"}


def precision(evidences, pattern):
    if not evidences:
        return None
    rx = re.compile(pattern, re.I)
    return sum(bool(rx.search(e.text)) for e in evidences) / len(evidences)


def refusal_ok(status, esperado):
    return status in RECUSAS if esperado in RECUSAS else status == esperado


def suggest_threshold(no_tema, sem_evidencia):
    """Ponto médio entre a menor semelhança das perguntas no tema e a maior das sem evidência."""
    return round((min(no_tema) + max(sem_evidencia)) / 2, 4)


def _tokens(t):
    return re.findall(r"\w+", t.lower())


def bm25_search(docs, query, k=8):
    from rank_bm25 import BM25Okapi

    if docs.empty:
        return docs
    toks = [_tokens(t) for t in docs["text"]]
    q = _tokens(query)
    s = BM25Okapi(toks).get_scores(q)
    tem_termo = [bool(set(q) & set(t)) for t in toks]  # com poucos documentos o BM25 pode dar nota <= 0
    top = docs.assign(_s=s, _t=tem_termo).query("_t").sort_values("_s", ascending=False, kind="stable")
    return top.head(k).drop(columns=["_s", "_t"])


def run(questions, col, embedder, llm, docs, min_sim, k=8, parcial=None):
    from voc_rag.filters import Filters, apply_filters
    from voc_rag.generate import PROMPT_VERSION
    from voc_rag.pipeline import answer
    from voc_rag.retrieve import retrieve

    linhas = []
    for q in questions:
        f = Filters(**q.get("filtros", {}))
        t = time.time()
        r = answer(q["pergunta"], f, col, embedder, llm, docs, k=k, min_sim=min_sim)
        seg = time.time() - t
        top_sim = retrieve(col, embedder, r["query"], f, k=1, min_sim=0).closest
        linha = {
            "id": q["id"], "tipo": q["tipo"], "esperado": q["esperado"], "status": r["status"],
            "acertou": refusal_ok(r["status"], q["esperado"]), "sim_max": top_sim[0].similarity if top_sim else None,
            "n_evid": len(r["evidences"]), "segundos": round(seg, 1), "busca": r["query"],
            "resposta": r.get("raw_answer") or r["answer"],
            "prompt_version": PROMPT_VERSION, "modelo": getattr(llm, "model", ""),  # texto do modelo, para a conferência manual
        }
        if "relevante" in q:
            linha["precisao_rag"] = precision(r["evidences"][:k], q["relevante"]) if r["status"] == "ok" else None
            bm = bm25_search(apply_filters(docs, f), q["pergunta"], k=k)
            rx = re.compile(q["relevante"], re.I)
            linha["precisao_bm25"] = bm["text"].map(lambda s: bool(rx.search(s))).mean() if len(bm) else None
            bm2 = bm25_search(apply_filters(docs, f), r["query"], k=k)  # mesma entrada do RAG: comparação justa
            linha["precisao_bm25_reescrita"] = bm2["text"].map(lambda s: bool(rx.search(s))).mean() if len(bm2) else None
        linhas.append(linha)
        print(f"{q['id']} {r['status']} ({seg:.0f}s)", flush=True)
        if parcial:  # salva a cada pergunta: se parar no meio, não perde o que já foi feito
            pd.DataFrame(linhas).to_csv(parcial, index=False)
    return pd.DataFrame(linhas)


def report(res: pd.DataFrame) -> str:
    validas = res[res["tipo"].isin(["no_tema", "filtro"])]
    recusar = res[~res["tipo"].isin(["no_tema", "filtro"])]
    resumo = {
        "Perguntas válidas respondidas (status ok)": f"{(validas.status == 'ok').sum()}/{len(validas)}",
        "Recusas corretas (fora do escopo ou sem evidência)": f"{recusar.acertou.sum()}/{len(recusar)}",
        "Precisão média das evidências, RAG": f"{validas.precisao_rag.mean():.2f}",
        "Precisão média das evidências, BM25 (pergunta original)": f"{validas.precisao_bm25.mean():.2f}",
        "Precisão média das evidências, BM25 (pergunta reescrita)": f"{validas.precisao_bm25_reescrita.mean():.2f}",
        "Tempo médio por pergunta (s)": f"{res.segundos.mean():.1f}",
    }
    md = "| Métrica | Resultado |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in resumo.items())
    return md + "\n\n" + res.drop(columns=["resposta", "prompt_version", "modelo"], errors="ignore").to_markdown(index=False)


if __name__ == "__main__":
    from voc_rag.cli import CHROMA, DOCS, REVISAO_E5
    from voc_rag.embeddings import E5Embedder
    from voc_rag.index import open_index
    from voc_rag.llm import make_llm

    import sys

    min_sim = float(sys.argv[1]) if len(sys.argv) > 1 else 0.80
    qs = json.load(open("eval/perguntas.json", encoding="utf-8"))
    res = run(qs, open_index(CHROMA), E5Embedder(revision=REVISAO_E5), make_llm(), pd.read_parquet(DOCS), min_sim,
              parcial="eval/resultados_parcial.csv")
    res.to_csv("eval/resultados.csv", index=False)
    no_tema = res[res.tipo.isin(["no_tema", "filtro"])].sim_max.dropna()
    sem = res[res.tipo == "sem_evidencia"].sim_max.dropna()
    sugestao = suggest_threshold(list(no_tema), list(sem)) if len(no_tema) and len(sem) else None
    md = f"Prompt {res.prompt_version.iloc[0]}, modelo {res.modelo.iloc[0]}. Limite usado: {min_sim}. Limite sugerido pelos dados: {sugestao}\n\n" + report(res)
    open("eval/resultados.md", "w", encoding="utf-8").write(md)
    print(md)
