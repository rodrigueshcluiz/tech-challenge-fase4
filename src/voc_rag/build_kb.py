"""Etapa 1: transforma as tabelas da Olist em documentos para o RAG.

Decisões (08/11/2026; definições em docs/definicoes.md):
P1 só avaliações com texto; P2 título + comentário num texto só;
P6 remove só textos sem nenhuma letra (sem corte por tamanho);
P4 uma linha por review_id, com a lista de pedidos;
P5 metadados de nota, data, categoria, estado e status;
P8 situação da entrega, dias de atraso e "respondeu antes de receber"
(pela data da resposta, review_answer_timestamp; revisão de 10/11/2026).
"""
import re
from pathlib import Path

import pandas as pd

TEM_LETRA = r"[^\W\d_]"
ORDEM_SITUACAO = {"no_prazo": 0, "atrasado": 1, "nao_entregue": 2}  # para escolher o pior


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _juntar(titulo: str, msg: str) -> str:
    if titulo and msg:
        sep = " " if titulo[-1] in ".!?" else ". "
        return titulo + sep + msg
    return titulo or msg


def _entrega_por_avaliacao(r: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Situação da entrega de cada linha (avaliação x pedido), comparando só datas."""
    o = orders.set_index("order_id")
    x = r[["review_id", "order_id", "review_answer_timestamp"]].join(o, on="order_id")
    entregue = pd.to_datetime(x["order_delivered_customer_date"]).dt.normalize()
    previsto = pd.to_datetime(x["order_estimated_delivery_date"]).dt.normalize()
    # review_creation_date é o envio da pesquisa; quem conta é a data em que o cliente respondeu
    avaliado = pd.to_datetime(x["review_answer_timestamp"]).dt.normalize()
    x["days_late"] = (entregue - previsto).dt.days.astype("Int64")
    x["delivery_status"] = "no_prazo"
    x.loc[x["days_late"] > 0, "delivery_status"] = "atrasado"
    x.loc[entregue.isna(), "delivery_status"] = "nao_entregue"
    x["reviewed_before_delivery"] = entregue.isna() | (avaliado < entregue)
    return x[["review_id", "order_id", "delivery_status", "days_late", "reviewed_before_delivery"]]


def build_documents(reviews, orders, items, products, customers) -> pd.DataFrame:
    r = reviews.copy()
    titulo = r["review_comment_title"].fillna("").map(clean_text)
    msg = r["review_comment_message"].fillna("").map(clean_text)
    r["text"] = [_juntar(t, m) for t, m in zip(titulo, msg)]
    r = r[r["text"].str.contains(TEM_LETRA, regex=True)]  # P1 e P6: sem texto ou sem letras

    cats = (
        items.merge(products, on="product_id", how="left")
        .dropna(subset=["product_category_name"])
        .groupby("order_id")["product_category_name"]
        .agg(lambda s: "|".join(sorted(set(s))))
        .rename("categories")
    )
    o = orders.merge(customers, on="customer_id", how="left").merge(cats, on="order_id", how="left")
    o["categories"] = o["categories"].fillna("")
    r = r.merge(o[["order_id", "order_status", "customer_state", "categories"]], on="order_id", how="left")
    r = r.merge(_entrega_por_avaliacao(r, orders), on=["review_id", "order_id"])
    r["_pior"] = r["delivery_status"].map(ORDEM_SITUACAO)

    # P4: uma linha por avaliação (o texto é idêntico entre as cópias)
    r = r.sort_values(["review_id", "order_id"])
    docs = r.groupby("review_id", sort=True).agg(
        text=("text", "first"),
        order_ids=("order_id", lambda s: "|".join(s)),
        score=("review_score", "first"),
        review_date=("review_creation_date", lambda s: str(s.iloc[0])[:10]),
        categories=("categories", lambda s: "|".join(sorted(set("|".join(s).split("|")) - {""}))),
        customer_state=("customer_state", "first"),
        order_status=("order_status", "first"),
        _pior=("_pior", "max"),
        days_late=("days_late", "max"),
        reviewed_before_delivery=("reviewed_before_delivery", "any"),
    ).reset_index()
    docs["delivery_status"] = docs.pop("_pior").map({v: k for k, v in ORDEM_SITUACAO.items()})
    return docs


def main(raw_dir="data/raw", out="data/processed/documents.parquet"):
    raw = Path(raw_dir)
    ler = lambda nome: pd.read_csv(raw / f"olist_{nome}_dataset.csv")
    docs = build_documents(
        reviews=ler("order_reviews"),
        orders=ler("orders"),
        items=ler("order_items"),
        products=ler("products"),
        customers=ler("customers"),
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(out, index=False)
    print(f"{len(docs)} documentos -> {out}")


if __name__ == "__main__":
    main()
