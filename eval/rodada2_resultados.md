Limite usado: 0.85. Limite sugerido pelos dados: 0.8896

| Métrica | Resultado |
|---|---|
| Perguntas válidas respondidas (status ok) | 20/20 |
| Recusas corretas (fora do escopo ou sem evidência) | 10/10 |
| Precisão média das evidências, RAG | 0.86 |
| Precisão média das evidências, BM25 | 0.49 |
| Tempo médio por pergunta (s) | 21.1 |

| id   | tipo           | esperado       | status         | acertou   |   sim_max |   n_evid |   segundos | busca                                                        |   precisao_rag |   precisao_bm25 |
|:-----|:---------------|:---------------|:---------------|:----------|----------:|---------:|-----------:|:-------------------------------------------------------------|---------------:|----------------:|
| Q01  | no_tema        | ok             | ok             | True      |    0.8947 |       40 |       72.7 | problemas recorrentes nas avaliações dos clientes            |          0.625 |           0.25  |
| Q02  | no_tema        | ok             | ok             | True      |    0.9291 |       40 |        3.6 | entrega atrasada, pacote não chegou, rastreamento falho      |          1     |           0.125 |
| Q03  | no_tema        | ok             | ok             | True      |    0.9355 |       40 |       15.3 | excelente experiência de compra                              |          0.75  |           0.375 |
| Q04  | no_tema        | ok             | ok             | True      |    0.9384 |       40 |       28.1 | produto com defeito, entrega atrasada, atendimento ruim      |          1     |           0.75  |
| Q05  | no_tema        | ok             | ok             | True      |    0.9141 |       40 |       25.2 | produto de baixa qualidade, padrão recorrente nas avaliações |          0.625 |           0.5   |
| Q06  | no_tema        | ok             | ok             | True      |    0.9133 |       40 |       22.6 | problemas na entrega, atraso e não recebimento               |          1     |           0.25  |
| Q07  | no_tema        | ok             | ok             | True      |    0.9386 |       40 |       23.8 | entrega atrasada, prazo de entrega não cumprido              |          1     |           0.5   |
| Q08  | no_tema        | ok             | ok             | True      |    0.9496 |       40 |       24.6 | produto chegou quebrado, defeituoso                          |          1     |           0.25  |
| Q09  | no_tema        | ok             | ok             | True      |    0.9454 |        8 |       13.8 | pedido chegou incompleto, itens faltando                     |          1     |           0.875 |
| Q10  | no_tema        | ok             | ok             | True      |    0.9561 |        8 |       14.2 | recebi produto diferente do que comprei                      |          1     |           0.125 |
| Q11  | no_tema        | ok             | ok             | True      |    0.9219 |       40 |       18   | avaliação do atendimento e suporte da loja                   |          0.375 |           0.5   |
| Q12  | no_tema        | ok             | ok             | True      |    0.9181 |       40 |       28.4 | dificuldade para cancelar pedido e obter reembolso           |          1     |           0.125 |
| Q13  | no_tema        | ok             | ok             | True      |    0.9317 |       40 |       27.4 | entrega rápida, clientes elogiam                             |          1     |           0.875 |
| Q14  | no_tema        | ok             | ok             | True      |    0.9228 |       40 |       26.1 | embalagem dos produtos                                       |          0.5   |           0.25  |
| Q15  | no_tema        | ok             | ok             | True      |    0.9482 |       40 |       19   | valor do frete muito alto                                    |          1     |           1     |
| Q16  | filtro         | ok             | ok             | True      |    0.8751 |       40 |       29.8 | principais problemas nas avaliações com nota baixa           |          0.875 |           0.625 |
| Q17  | filtro         | ok             | ok             | True      |    0.9038 |       40 |       27.1 | os produtos de cama, mesa e banho são muito elogiados        |          0.875 |           0.75  |
| Q18  | filtro         | ok             | ok             | True      |    0.9265 |       40 |       28.1 | entrega atrasada, falta de atualização e serviço ruim        |          1     |           0.625 |
| Q19  | filtro         | ok             | ok             | True      |    0.8968 |       40 |       29.4 | clientes avaliam antes de receber o produto                  |          0.875 |           0.25  |
| Q20  | filtro         | ok             | ok             | True      |    0.8932 |       40 |       36.9 | reclamações frequentes em informática e acessórios           |          0.625 |           0.875 |
| Q21  | fora_do_escopo | fora_do_escopo | fora_do_escopo | True      |    0.8719 |        0 |        9.3 | preço do iPhone hoje                                         |        nan     |         nan     |
| Q22  | fora_do_escopo | fora_do_escopo | fora_do_escopo | True      |    0.8653 |        0 |        3   | previsão do tempo para amanhã em São Paulo                   |        nan     |         nan     |
| Q23  | fora_do_escopo | fora_do_escopo | fora_do_escopo | True      |    0.8289 |        0 |        0.6 | quem ganhou a copa do mundo de 2022                          |        nan     |         nan     |
| Q24  | fora_do_escopo | fora_do_escopo | fora_do_escopo | True      |    0.8848 |        0 |        0.7 | preciso de receita de bolo de cenoura                        |        nan     |         nan     |
| Q25  | fora_do_escopo | fora_do_escopo | fora_do_escopo | True      |    0.9105 |        0 |        5.3 | atendimento da Amazon                                        |        nan     |         nan     |
| Q26  | sem_evidencia  | sem_evidencia  | sem_evidencia  | True      |    0.8948 |       40 |       30.3 | opinião sobre pagamento via PIX                              |        nan     |         nan     |
| Q27  | sem_evidencia  | sem_evidencia  | sem_evidencia  | True      |    0.8984 |       40 |       26.5 | o chatbot de atendimento não resolve minhas dúvidas          |        nan     |         nan     |
| Q28  | sem_evidencia  | sem_evidencia  | sem_evidencia  | True      |    0.9041 |       40 |       21.3 | cashback não funciona, reclamação                            |        nan     |         nan     |
| Q29  | sem_evidencia  | sem_evidencia  | sem_evidencia  | True      |    0.8858 |        8 |       12.4 | paguei com criptomoeda                                       |        nan     |         nan     |
| Q30  | sem_evidencia  | sem_evidencia  | fora_do_escopo | True      |    0.8567 |        0 |       11   | realidade virtual para provar roupas                         |        nan     |         nan     |