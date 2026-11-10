Limite usado: 0.8. Limite sugerido pelos dados: 0.8933

| Métrica | Resultado |
|---|---|
| Perguntas válidas respondidas (status ok) | 14/20 |
| Recusas corretas (fora do escopo ou sem evidência) | 6/10 |
| Precisão média das evidências, RAG | 0.85 |
| Precisão média das evidências, BM25 | 0.49 |
| Tempo médio por pergunta (s) | 17.4 |

| id   | tipo           | esperado       | status            | acertou   |   sim_max |   n_evid |   segundos | busca                                                                            |   precisao_rag |   precisao_bm25 |
|:-----|:---------------|:---------------|:------------------|:----------|----------:|---------:|-----------:|:---------------------------------------------------------------------------------|---------------:|----------------:|
| Q01  | no_tema        | ok             | resposta_invalida | False     |    0.9031 |       40 |       17.1 | problemas recorrentes nas avaliações                                             |        nan     |           0.25  |
| Q02  | no_tema        | ok             | ok                | True      |    0.9324 |       40 |        2.8 | entrega atrasada, pacote não chega no prazo                                      |          1     |           0.125 |
| Q03  | no_tema        | ok             | resposta_invalida | False     |    0.9355 |       40 |       12.7 | excelente experiência de compra                                                  |        nan     |           0.375 |
| Q04  | no_tema        | ok             | ok                | True      |    0.9362 |       40 |       21.3 | produto com defeito, atraso na entrega, atendimento ruim                         |          1     |           0.75  |
| Q05  | no_tema        | ok             | ok                | True      |    0.9141 |       40 |       27.4 | produto de baixa qualidade, padrão recorrente nas avaliações                     |          0.625 |           0.5   |
| Q06  | no_tema        | ok             | ok                | True      |    0.9133 |       40 |       19.8 | problemas na entrega, atraso e não recebimento                                   |          1     |           0.25  |
| Q07  | no_tema        | ok             | resposta_invalida | False     |    0.9386 |       40 |       23.5 | entrega atrasada, prazo de entrega não cumprido                                  |        nan     |           0.5   |
| Q08  | no_tema        | ok             | ok                | True      |    0.9413 |       40 |       25.4 | produto chegou quebrado ou com defeito                                           |          0.875 |           0.25  |
| Q09  | no_tema        | ok             | ok                | True      |    0.9454 |        8 |       12.4 | pedido chegou incompleto, itens faltando                                         |          1     |           0.875 |
| Q10  | no_tema        | ok             | ok                | True      |    0.9561 |        8 |        9.6 | recebi produto diferente do que comprei                                          |          1     |           0.125 |
| Q11  | no_tema        | ok             | ok                | True      |    0.9219 |       40 |       21.4 | avaliação do atendimento e suporte da loja                                       |          0.375 |           0.5   |
| Q12  | no_tema        | ok             | ok                | True      |    0.9189 |       40 |       30.7 | dificuldade em cancelar pedido e receber reembolso                               |          1     |           0.125 |
| Q13  | no_tema        | ok             | ok                | True      |    0.9216 |       40 |       23.4 | clientes elogiam rapidez da entrega                                              |          1     |           0.875 |
| Q14  | no_tema        | ok             | ok                | True      |    0.9259 |       40 |       23.5 | embalagem dos produtos chegou danificada                                         |          0.625 |           0.25  |
| Q15  | no_tema        | ok             | resposta_invalida | False     |    0.9235 |       40 |       22.5 | frete caro, reclamação                                                           |        nan     |           1     |
| Q16  | filtro         | ok             | resposta_invalida | False     |    0.8811 |       40 |       25.8 | problemas recorrentes em avaliações baixas                                       |        nan     |           0.625 |
| Q17  | filtro         | ok             | ok                | True      |    0.8836 |       40 |       26.8 | os clientes elogiam a qualidade e variedade dos produtos de cama, mesa e banho   |          0.5   |           0.75  |
| Q18  | filtro         | ok             | ok                | True      |    0.9243 |       40 |       22.7 | entrega atrasada, temas recorrentes nas avaliações                               |          1     |           0.625 |
| Q19  | filtro         | ok             | resposta_invalida | False     |    0.9183 |       40 |       30.5 | avaliação antes de receber o produto                                             |        nan     |           0.25  |
| Q20  | filtro         | ok             | ok                | True      |    0.9206 |       40 |       25.7 | produto com defeito, entrega atrasada e suporte ruim em informática e acessórios |          0.875 |           0.875 |
| Q21  | fora_do_escopo | fora_do_escopo | fora_do_escopo    | True      |    0.8719 |        0 |        7.7 | preço do iPhone hoje                                                             |        nan     |         nan     |
| Q22  | fora_do_escopo | fora_do_escopo | fora_do_escopo    | True      |    0.8653 |        0 |        3.2 | previsão do tempo para amanhã em São Paulo                                       |        nan     |         nan     |
| Q23  | fora_do_escopo | fora_do_escopo | fora_do_escopo    | True      |    0.8289 |        0 |        0.8 | quem ganhou a copa do mundo de 2022                                              |        nan     |         nan     |
| Q24  | fora_do_escopo | fora_do_escopo | fora_do_escopo    | True      |    0.8762 |        0 |        3.6 | receita de bolo de cenoura                                                       |        nan     |         nan     |
| Q25  | fora_do_escopo | fora_do_escopo | fora_do_escopo    | True      |    0.9052 |        0 |        5   | atendimento da Amazon é ruim                                                     |        nan     |         nan     |
| Q26  | sem_evidencia  | sem_evidencia  | resposta_invalida | False     |    0.8846 |       40 |       20.5 | pagamento via PIX é fácil e rápido                                               |        nan     |         nan     |
| Q27  | sem_evidencia  | sem_evidencia  | resposta_invalida | False     |    0.8802 |       40 |       21.6 | chatbot de atendimento avaliado pelos clientes                                   |        nan     |         nan     |
| Q28  | sem_evidencia  | sem_evidencia  | resposta_invalida | False     |    0.9055 |       40 |       21.9 | cashback ruim, reclamação                                                        |        nan     |         nan     |
| Q29  | sem_evidencia  | sem_evidencia  | resposta_invalida | False     |    0.8858 |        8 |       10.8 | paguei com criptomoeda                                                           |        nan     |         nan     |
| Q30  | sem_evidencia  | sem_evidencia  | fora_do_escopo    | True      |    0.8684 |        0 |        0.9 | experiência de realidade virtual ao provar roupas                                |        nan     |         nan     |