"""Etapa 2: grava os vetores das avaliações no Chroma, com os metadados para filtro."""
import re

import pandas as pd

COLECAO = "avaliacoes"


def text_key(t: str) -> str:
    """Chave para reconhecer textos iguais: minúsculas, sem pontuação nas pontas, espaços únicos."""
    return re.sub(r"\s+", " ", t.lower()).strip(" .!?,;")


def metadata_of(row) -> dict:
    m = {
        "score": int(row.score),
        "review_date": row.review_date,
        "review_ym": int(row.review_date[:7].replace("-", "")),
        "categories": row.categories,
        "customer_state": row.customer_state if isinstance(row.customer_state, str) else "",
        "order_status": row.order_status if isinstance(row.order_status, str) else "",
        "delivery_status": row.delivery_status,
        "reviewed_before_delivery": bool(row.reviewed_before_delivery),
        "text_key": text_key(row.text),
    }
    if not pd.isna(row.days_late):
        m["days_late"] = int(row.days_late)
    for c in filter(None, row.categories.split("|")):
        m["cat_" + c] = True
    return m


def build_index(docs: pd.DataFrame, embedder, path, batch=5000):
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    if COLECAO in [c.name for c in client.list_collections()]:
        client.delete_collection(COLECAO)  # recria do zero: rodar duas vezes dá o mesmo índice
    col = client.create_collection(COLECAO, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    keys = docs["text"].map(text_key)
    primeiros = docs.groupby(keys, sort=False)["text"].first()  # texto repetido é calculado uma vez só
    vetores = dict(zip(primeiros.index, embedder.embed_passages(list(primeiros.values))))

    for i in range(0, len(docs), batch):
        part = docs.iloc[i : i + batch]
        col.add(
            ids=list(part["review_id"]),
            embeddings=[vetores[k].tolist() for k in keys.iloc[i : i + batch]],
            documents=list(part["text"]),
            metadatas=[metadata_of(r) for r in part.itertuples()],
        )
    return col


def open_index(path):
    import chromadb

    return chromadb.PersistentClient(path=str(path)).get_collection(COLECAO, embedding_function=None)
