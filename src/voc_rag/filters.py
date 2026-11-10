"""Filtros de metadados: uma definição só, usada na busca (Chroma) e nas estatísticas (pandas)."""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Filters:
    min_score: int | None = None
    max_score: int | None = None
    category: str | None = None
    delivery_status: list[str] = field(default_factory=list)  # no_prazo, atrasado, nao_entregue
    reviewed_before_delivery: bool | None = None
    month_from: int | None = None  # AAAAMM, ex.: 201801
    month_to: int | None = None


def to_where(f: Filters):
    c = []
    if f.min_score is not None:
        c.append({"score": {"$gte": f.min_score}})
    if f.max_score is not None:
        c.append({"score": {"$lte": f.max_score}})
    if f.category:
        c.append({"cat_" + f.category: True})
    if f.delivery_status:
        c.append({"delivery_status": {"$in": list(f.delivery_status)}})
    if f.reviewed_before_delivery is not None:
        c.append({"reviewed_before_delivery": f.reviewed_before_delivery})
    if f.month_from is not None:
        c.append({"review_ym": {"$gte": f.month_from}})
    if f.month_to is not None:
        c.append({"review_ym": {"$lte": f.month_to}})
    if not c:
        return None
    return c[0] if len(c) == 1 else {"$and": c}


def apply_filters(docs: pd.DataFrame, f: Filters) -> pd.DataFrame:
    m = pd.Series(True, index=docs.index)
    if f.min_score is not None:
        m &= docs["score"] >= f.min_score
    if f.max_score is not None:
        m &= docs["score"] <= f.max_score
    if f.category:
        m &= docs["categories"].str.split("|").apply(lambda cs: f.category in cs)
    if f.delivery_status:
        m &= docs["delivery_status"].isin(f.delivery_status)
    if f.reviewed_before_delivery is not None:
        m &= docs["reviewed_before_delivery"] == f.reviewed_before_delivery
    ym = docs["review_date"].str[:7].str.replace("-", "").astype(int)
    if f.month_from is not None:
        m &= ym >= f.month_from
    if f.month_to is not None:
        m &= ym <= f.month_to
    return docs[m]
