"""Etapa 3: busca semântica com filtros, agrupando textos iguais e medindo se há evidência suficiente."""
from dataclasses import dataclass, field

from voc_rag.filters import Filters, to_where  # noqa: F401  (Filters reexportado para quem importa daqui)


MAX_FETCH = 2000


@dataclass
class Evidence:
    text: str
    similarity: float
    score: int
    delivery_status: str
    categories: str
    review_date: str
    review_ids: list[str] = field(default_factory=list)  # entre as recuperadas
    total: int | None = None  # quantas avaliações do recorte inteiro têm este mesmo texto

    @property
    def count(self):
        return self.total if self.total is not None else len(self.review_ids)


@dataclass
class Retrieval:
    evidences: list[Evidence]  # as que passaram do limite de semelhança
    closest: list[Evidence]  # as mais próximas, mesmo abaixo do limite (para mostrar na recusa)
    sufficient: bool


def retrieve(col, embedder, query, filters: Filters, k=8, min_sim=0.85, min_evidences=1, fetch_factor=5):
    q = embedder.embed_queries([query])[0].tolist()
    total, n = col.count(), k * fetch_factor
    while True:  # textos muito repetidos podem ocupar a busca: amplia até achar k textos distintos
        res = col.query(query_embeddings=[q], n_results=max(1, min(n, total)), where=to_where(filters),
                        include=["documents", "metadatas", "distances"])
        distintos = {m["text_key"] for m in res["metadatas"][0]}
        if len(distintos) >= k or len(res["ids"][0]) < min(n, total) or n >= min(total, MAX_FETCH):
            break
        n *= 2
    grupos = {}
    for rid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        g = grupos.setdefault(meta["text_key"], {"sim": 1 - dist, "membros": []})
        g["membros"].append((rid, doc, meta))
    evs = []
    for g in grupos.values():
        rid, doc, meta = min(g["membros"], key=lambda m: m[0])  # representante fixo: menor review_id
        evs.append(Evidence(doc, round(g["sim"], 4), meta["score"], meta["delivery_status"], meta["categories"],
                            meta["review_date"], sorted(m[0] for m in g["membros"])))
    evs.sort(key=lambda e: (-e.similarity, e.review_ids[0]))
    evs = evs[:k]
    boas = [e for e in evs if e.similarity >= min_sim]
    return Retrieval(evidences=boas, closest=evs, sufficient=len(boas) >= min_evidences)
