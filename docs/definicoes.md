# Definições das métricas e campos

Uma definição por métrica, usada em todo o projeto (código, README e vídeo). Mudou a regra? Muda aqui primeiro.

| Campo | Definição |
|---|---|
| `text` | Título + ". " + comentário, com espaços e quebras de linha normalizados. Se só um existe, só ele. |
| Documento | Uma avaliação (`review_id`) com pelo menos uma letra no texto. Textos só com pontuação ou números ("...", "10") ficam fora. Não há corte por tamanho. |
| `order_ids` | Todos os pedidos ligados à avaliação, separados por `|` (307 avaliações valem para mais de um pedido). |
| `score` | Nota de 1 a 5 dada pelo cliente. |
| `review_date` | Data de criação da avaliação (AAAA-MM-DD). |
| `categories` | Categorias dos produtos do pedido, em português, separadas por `|`. Vazio quando não há categoria. |
| `customer_state` | UF do cliente. |
| `order_status` | Status do pedido na base da Olist. |
| `days_late` | Data de entrega real menos data prevista, em dias, comparando só as datas (sem hora). Negativo = chegou antes. Vazio quando não foi entregue. |
| `delivery_status` | `no_prazo` se `days_late` <= 0 (entregue no dia previsto conta como no prazo); `atrasado` se `days_late` > 0; `nao_entregue` sem data de entrega. |
| `reviewed_before_delivery` | Verdadeiro quando a avaliação foi escrita antes da data de entrega, ou o pedido nunca foi entregue. |
| Avaliação com vários pedidos | Vale o pior caso: pior `delivery_status`, maior `days_late` e `reviewed_before_delivery` verdadeiro se for verdadeiro em algum pedido. |

Números na base completa (08/11/2026): 42.138 documentos; no prazo 36.420 (nota média 4,0), atrasado 3.829 (1,9), não entregue 1.889 (1,5); avaliou sem ter recebido 5.106 (1,7).
