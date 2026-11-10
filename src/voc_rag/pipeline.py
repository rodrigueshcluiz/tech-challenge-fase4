"""Pipeline completo: pergunta -> reescrita -> busca -> (estatísticas) -> resposta -> conferência."""
import json
import re

from voc_rag.filters import Filters, apply_filters
from voc_rag.index import text_key
from voc_rag.generate import REWRITE_SYSTEM, SYSTEM_PROMPT, build_prompt, cited_ids
from voc_rag.retrieve import retrieve
from voc_rag.stats import format_stats, summarize

AMPLA = re.compile(r"\b(principais|mais (frequentes|comuns)|padr(ão|ões)|panorama|em geral|recorrentes)\b", re.I)

MSG = {
    "fora_do_escopo": "Essa pergunta está fora do escopo da base: ela só tem avaliações de clientes da Olist (2016-2018).",
    "sem_evidencia": "Não encontrei avaliações suficientemente relacionadas a essa pergunta para responder com segurança.",
    "resposta_invalida": "A resposta gerada citou evidências que não existem ou não citou nenhuma, por isso foi descartada. "
    "Veja abaixo as avaliações encontradas.",
}


def rewrite(question, llm):
    try:
        r = json.loads(llm(REWRITE_SYSTEM, question, json_mode=True))
        return bool(r["in_scope"]), bool(r["broad"]), str(r["query"]) or question
    except Exception:  # Gemini fora do ar ou JSON quebrado: segue com a pergunta original
        return True, bool(AMPLA.search(question)), question


def answer(question, filters: Filters, col, embedder, llm, docs, k=8, k_broad=40, min_sim=0.85, min_evidences=1):
    in_scope, broad, query = rewrite(question, llm)
    out = {"question": question, "query": query, "broad": broad, "k": k_broad if broad else k,
           "evidences": [], "stats": None, "answer": None}
    if not in_scope:
        return {**out, "status": "fora_do_escopo", "answer": MSG["fora_do_escopo"]}

    r = retrieve(col, embedder, query, filters, k=out["k"], min_sim=min_sim, min_evidences=min_evidences)
    if not r.sufficient:
        return {**out, "status": "sem_evidencia", "answer": MSG["sem_evidencia"], "evidences": r.closest}

    recorte = apply_filters(docs, filters)
    repeticoes = recorte["text"].map(text_key).value_counts()
    for e in r.evidences:
        e.total = int(repeticoes.get(text_key(e.text), len(e.review_ids)))

    stats_text = None
    if broad:
        out["stats"] = summarize(recorte)
        stats_text = format_stats(out["stats"])
    texto = llm(SYSTEM_PROMPT, build_prompt(question, r.evidences, stats_text))
    if texto.strip().upper().startswith("SEM_EVIDENCIA"):  # 3ª trava: o modelo leu as evidências e nenhuma responde
        return {**out, "status": "sem_evidencia", "answer": MSG["sem_evidencia"], "evidences": r.evidences, "raw_answer": texto}
    citadas = cited_ids(texto)
    ok = bool(citadas) and all(1 <= c <= len(r.evidences) for c in citadas)
    if not ok:
        return {**out, "status": "resposta_invalida", "answer": MSG["resposta_invalida"], "evidences": r.evidences, "raw_answer": texto}
    return {**out, "status": "ok", "answer": texto, "evidences": r.evidences}
