"""Contagens sobre o recorte inteiro (pandas), para perguntas amplas. O LLM não calcula números."""
import pandas as pd


def summarize(sub: pd.DataFrame) -> dict:
    n = len(sub)
    pct = lambda s: {str(k): round(100 * v / n, 1) for k, v in s.value_counts().sort_index().items()} if n else {}
    cats = sub["categories"].str.split("|").explode()
    return {
        "n": n,
        "pct_por_nota": pct(sub["score"]),
        "pct_por_situacao_entrega": pct(sub["delivery_status"]),
        "pct_avaliou_sem_receber": round(100 * sub["reviewed_before_delivery"].mean(), 1) if n else 0.0,
        "top_categorias": cats[cats != ""].value_counts().head(5).to_dict(),
    }


def format_stats(s: dict) -> str:
    return (
        f"Avaliações com texto no recorte: {s['n']}\n"
        f"% por nota: {s['pct_por_nota']}\n"
        f"% por situação da entrega: {s['pct_por_situacao_entrega']}\n"
        f"% que respondeu a pesquisa antes de receber o produto: {s['pct_avaliou_sem_receber']}\n"
        f"Categorias mais frequentes (contagem): {s['top_categorias']}"
    )
