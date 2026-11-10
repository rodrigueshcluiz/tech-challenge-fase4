"""Interface do RAG Voz do Cliente (Olist). Rodar na pasta voc-rag:  streamlit run app.py"""
import os

import pandas as pd
import streamlit as st

from voc_rag.cli import CHROMA, DOCS
from voc_rag.embeddings import E5Embedder
from voc_rag.filters import Filters
from voc_rag.generate import mask_names
from voc_rag.index import open_index
from voc_rag.llm import make_llm
from voc_rag.pipeline import answer

SITUACOES = {"no_prazo": "No prazo", "atrasado": "Atrasado", "nao_entregue": "Não entregue"}
STATUS = {
    "ok": ("success", "Resposta baseada nas avaliações citadas"),
    "sem_evidencia": ("warning", "Sem evidência suficiente: o sistema preferiu não responder"),
    "fora_do_escopo": ("info", "Fora do escopo da base"),
    "resposta_invalida": ("error", "Resposta descartada: as citações não conferem com as evidências"),
}
EXEMPLOS = [
    "Quais são os principais problemas relatados pelos clientes?",
    "O que os clientes elogiam nas entregas no prazo?",
    "Quais reclamações aparecem sobre produtos com defeito?",
    "O que os clientes acham do pagamento via PIX?",
]


@st.cache_resource
def recursos():
    if os.environ.get("VOC_RAG_TESTE"):  # só para o teste automático da tela (tests/test_app.py)
        from tests.test_app import recursos_falsos

        return recursos_falsos()
    return open_index(CHROMA), E5Embedder(), make_llm(), pd.read_parquet(DOCS)


def tabela_evidencias(evidencias) -> pd.DataFrame:
    return pd.DataFrame([{
        "Ref.": f"R{i}",
        "Semelhança": round(e.similarity, 3),
        "Nota": e.score,
        "Entrega": SITUACOES.get(e.delivery_status, e.delivery_status),
        "Categoria": e.categories or "-",
        "Data": e.review_date[:7],
        "Vezes na base": e.count,
        "Texto": mask_names(e.text),
        "review_id (exemplo)": e.review_ids[0],
    } for i, e in enumerate(evidencias, 1)])


st.set_page_config(page_title="Voz do Cliente - Olist", layout="wide")
st.title("Voz do Cliente com RAG")
st.caption("Perguntas respondidas só com avaliações reais de clientes da Olist (2016-2018), sempre com as evidências citadas.")

col, emb, llm, docs = recursos()

with st.sidebar:
    st.header("Filtros")
    notas = st.slider("Nota", 1, 5, (1, 5))
    cats = sorted({c for s in docs["categories"] for c in s.split("|") if c})
    categoria = st.selectbox("Categoria do produto", ["(todas)"] + cats)
    situacao = st.multiselect("Situação da entrega", list(SITUACOES), format_func=SITUACOES.get)
    antes = st.checkbox("Só quem avaliou antes de receber")
    meses = sorted(docs["review_date"].str[:7].unique())
    de, ate = st.select_slider("Período", options=meses, value=(meses[0], meses[-1]))
    st.divider()
    st.caption(f"Modelo: {type(llm).__name__.replace('LLM', '')} {llm.model}")

filtros = Filters(
    min_score=notas[0] if notas[0] > 1 else None, max_score=notas[1] if notas[1] < 5 else None,
    category=None if categoria == "(todas)" else categoria, delivery_status=situacao,
    reviewed_before_delivery=True if antes else None,
    month_from=int(de.replace("-", "")) if de != meses[0] else None,
    month_to=int(ate.replace("-", "")) if ate != meses[-1] else None,
)

st.write("Exemplos:")
cols = st.columns(len(EXEMPLOS))
for c, ex in zip(cols, EXEMPLOS):
    if c.button(ex, use_container_width=True):
        st.session_state["pergunta"] = ex
pergunta = st.text_input("Sua pergunta", key="pergunta")

if pergunta:
    with st.spinner("Buscando avaliações e gerando a resposta..."):
        r = answer(pergunta, filtros, col, emb, llm, docs)
    tipo, rotulo = STATUS[r["status"]]
    getattr(st, tipo)(rotulo)
    st.markdown(r["answer"])
    if r["query"] != pergunta:
        st.caption(f"Busca feita como: \"{r['query']}\"")
    if r["evidences"]:
        titulo = "Evidências usadas" if r["status"] == "ok" else "Avaliações mais próximas encontradas"
        st.subheader(f"{titulo} ({len(r['evidences'])})")
        st.dataframe(tabela_evidencias(r["evidences"]), hide_index=True, use_container_width=True)
    if r["stats"]:
        with st.expander("Estatísticas do recorte (calculadas sobre todas as avaliações filtradas)"):
            st.json(r["stats"])
