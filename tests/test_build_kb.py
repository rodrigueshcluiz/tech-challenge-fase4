"""Testes da etapa 1: construção da base de conhecimento.

Cada teste corresponde a uma decisão tomada com o Luiz em 08/11/2026
(P1 a P5 na primeira rodada; P6 e P8 revisados depois da revisão crítica).
As definições das métricas estão em docs/definicoes.md.
Usa tabelas pequenas montadas à mão, para que o resultado esperado seja óbvio.
"""
import pandas as pd
import pytest

from voc_rag.build_kb import build_documents, clean_text

# review_id: (order_ids, nota, título, mensagem)
REVIEWS = {
    "r1": (["o1"], 1, None, "Ainda não recebi\r\no produto"),  # não entregue
    "r2": (["o2"], 5, "Ótimo", None),  # só título, entregue no prazo
    "r3": (["o3"], 4, None, None),  # sem texto nenhum
    "r4": (["o4"], 5, None, "..."),  # só pontuação
    "r5": (["o5", "o6"], 2, "Atraso", "chegou depois do prazo"),  # mesma avaliação em 2 pedidos
    "r6": (["o7"], 1, None, "cadê meu pedido"),  # avaliou antes de receber; chegou atrasado depois
    "r7": (["o8"], 1, None, "Péssimo"),  # uma palavra só, mas é queixa
    "r8": (["o9"], 5, None, "10"),  # só número
    "r9": (["o10"], 4, None, "chegou no dia certo"),  # entregue no dia previsto, à noite
}


@pytest.fixture
def tabelas():
    linhas = [
        (rid, oid, nota, tit, msg)
        for rid, (oids, nota, tit, msg) in REVIEWS.items()
        for oid in oids
    ]
    reviews = pd.DataFrame(
        linhas, columns=["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message"]
    )
    reviews["review_creation_date"] = "2018-01-10 00:00:00"
    reviews["review_answer_timestamp"] = "2018-01-10 12:00:00"
    entregas = {  # pedido: (status, entregue em, previsto para)
        "o1": ("shipped", None, "2018-01-08"),
        "o2": ("delivered", "2018-01-05 14:00:00", "2018-01-08"),
        "o3": ("delivered", "2018-01-05 14:00:00", "2018-01-08"),
        "o4": ("delivered", "2018-01-05 14:00:00", "2018-01-08"),
        "o5": ("delivered", "2018-01-09 10:00:00", "2018-01-08"),
        "o6": ("delivered", "2018-01-08 18:00:00", "2018-01-08"),  # no dia previsto, à tarde: no prazo
        "o7": ("delivered", "2018-01-15 09:00:00", "2018-01-08"),
        "o8": ("delivered", "2018-01-05 14:00:00", "2018-01-08"),
        "o9": ("delivered", "2018-01-05 14:00:00", "2018-01-08"),
        "o10": ("delivered", "2018-01-08 21:00:00", "2018-01-08"),
    }
    orders = pd.DataFrame(
        [(o, f"c_{o}", s, e, p) for o, (s, e, p) in entregas.items()],
        columns=["order_id", "customer_id", "order_status", "order_delivered_customer_date", "order_estimated_delivery_date"],
    )
    items = pd.DataFrame(
        {"order_id": ["o1", "o1", "o2", "o5", "o6"], "product_id": ["p1", "p2", "p1", "p3", "p3"]}
    )
    products = pd.DataFrame(
        {"product_id": ["p1", "p2", "p3"], "product_category_name": ["telefonia", "beleza_saude", None]}
    )
    customers = pd.DataFrame({"customer_id": orders["customer_id"], "customer_state": "SP"})
    customers.loc[customers.customer_id == "c_o2", "customer_state"] = "RJ"
    return dict(reviews=reviews, orders=orders, items=items, products=products, customers=customers)


@pytest.fixture
def docs(tabelas):
    return build_documents(**tabelas).set_index("review_id")


def test_clean_text_normaliza_espacos_e_quebras():
    assert clean_text("  Ainda não recebi\r\no   produto ") == "Ainda não recebi o produto"


def test_p1_avaliacao_sem_texto_fica_fora(docs):
    assert "r3" not in docs.index


def test_p2_titulo_e_comentario_viram_um_texto(docs):
    assert docs.loc["r5", "text"] == "Atraso. chegou depois do prazo"
    assert docs.loc["r2", "text"] == "Ótimo"


def test_p6_sai_so_texto_sem_letras(docs):
    assert "r4" not in docs.index  # "..."
    assert "r8" not in docs.index  # "10"
    assert docs.loc["r7", "text"] == "Péssimo"  # uma palavra, mas fica
    assert "searchable" not in docs.columns  # não existe mais corte por tamanho


def test_p4_uma_linha_por_avaliacao_com_lista_de_pedidos(docs):
    assert docs.index.is_unique
    assert docs.loc["r5", "order_ids"] == "o5|o6"


def test_p5_metadados(docs):
    r1 = docs.loc["r1"]
    assert r1["score"] == 1
    assert r1["categories"] == "beleza_saude|telefonia"  # pedido com 2 categorias, em ordem alfabética
    assert r1["customer_state"] == "SP"
    assert docs.loc["r2", "customer_state"] == "RJ"
    assert r1["order_status"] == "shipped"
    assert r1["review_date"] == "2018-01-10"
    assert docs.loc["r5", "categories"] == ""  # produto sem categoria


def test_p8_situacao_da_entrega(docs):
    assert docs.loc["r1", "delivery_status"] == "nao_entregue"
    assert docs.loc["r2", "delivery_status"] == "no_prazo"
    assert docs.loc["r6", "delivery_status"] == "atrasado"
    assert docs.loc["r9", "delivery_status"] == "no_prazo"  # no dia previsto conta como no prazo
    # avaliação em 2 pedidos (um atrasado, um no dia previsto): vale o pior
    assert docs.loc["r5", "delivery_status"] == "atrasado"


def test_p8_dias_de_atraso(docs):
    assert docs.loc["r2", "days_late"] == -3  # 3 dias antes do previsto
    assert docs.loc["r6", "days_late"] == 7
    assert docs.loc["r5", "days_late"] == 1  # maior atraso entre os pedidos
    assert pd.isna(docs.loc["r1", "days_late"])  # não entregue: não inventamos valor


def test_p8_avaliou_sem_ter_recebido(docs):
    assert docs.loc["r1", "reviewed_before_delivery"] == True  # noqa: E712  nunca entregue
    assert docs.loc["r6", "reviewed_before_delivery"] == True  # noqa: E712  avaliou dia 10, chegou dia 15
    assert docs.loc["r2", "reviewed_before_delivery"] == False  # noqa: E712
    assert docs.loc["r5", "reviewed_before_delivery"] == False  # noqa: E712


def test_e_idempotente(tabelas):
    a = build_documents(**tabelas)
    b = build_documents(**tabelas)
    pd.testing.assert_frame_equal(a, b)


def test_respondeu_antes_de_receber_usa_a_data_da_resposta(tabelas):
    """Revisão de 10/11: review_creation_date é o envio da pesquisa; o que importa é quando o cliente respondeu."""
    t = dict(tabelas, reviews=tabelas["reviews"].copy())
    t["reviews"].loc[t["reviews"].review_id == "r6", "review_answer_timestamp"] = "2018-01-20 10:00:00"  # respondeu depois de receber (dia 15)
    docs = build_documents(**t).set_index("review_id")
    assert docs.loc["r6", "reviewed_before_delivery"] == False  # noqa: E712  pesquisa enviada antes, resposta depois
    assert docs.loc["r1", "reviewed_before_delivery"] == True  # noqa: E712  nunca entregue
