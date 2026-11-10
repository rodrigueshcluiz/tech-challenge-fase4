# Voz do Cliente com RAG (Olist)

Tech Challenge da Fase 4 da POSTECH (IA para Devs / AI Scientist).

O sistema responde perguntas de negócio sobre a experiência de compra usando **só as avaliações reais de clientes** da base pública da Olist (2016 a 2018). Toda resposta cita as avaliações que a sustentam. Quando não há evidência suficiente, ou a pergunta foge do assunto, o sistema **recusa em vez de inventar**.

> Exemplo: "Quais são os principais problemas relatados pelos clientes?" → resposta agrupada por tema (atraso, não entrega, defeito...), com cada afirmação ligada a avaliações [R1], [R2]..., que aparecem numa tabela logo abaixo, com nota, situação da entrega, categoria e `review_id`.

## Sumário
1. [Arquitetura](#arquitetura)
2. [Tecnologias](#tecnologias)
3. [Preparação da base](#preparação-da-base-de-conhecimento)
4. [Pipeline RAG](#pipeline-rag)
5. [Perguntas sem evidência e fora do escopo](#perguntas-sem-evidência-e-fora-do-escopo)
6. [Metadados (desafio adicional)](#metadados-desafio-adicional)
7. [Avaliação](#avaliação)
8. [Exemplos de perguntas e respostas](#exemplos-de-perguntas-e-respostas)
9. [Como reproduzir](#como-reproduzir)
10. [Decisões técnicas e limitações](#decisões-técnicas-e-limitações)

## Arquitetura

```
CSV da Olist ──► build_kb.py ──► documents.parquet (42.138 avaliações + metadados)
                                        │
                                        ▼
                     embeddings.py (multilingual-e5-small, local, CPU)
                                        │
                                        ▼
                         index.py ──► Chroma (vetores + metadados para filtro)

Pergunta ──► 1. reescrita + escopo (LLM, JSON) ──► fora do escopo? recusa
                 │
                 ▼
             2. busca semântica com filtros (nota, categoria, entrega, período)
                 │   agrupa textos idênticos e conta repetições no recorte
                 ▼
             3. trava de semelhança mínima ──► nada parecido? recusa
                 │
                 ▼
             4. contexto numerado [R1..Rn] (+ estatísticas do recorte, se a pergunta for ampla)
                 │
                 ▼
             5. geração (LLM) ──► "SEM_EVIDENCIA"? recusa
                 │
                 ▼
             6. conferência das citações ──► citou o que não existe? resposta descartada
                 │
                 ▼
             resposta + tabela de evidências (app Streamlit)
```

| Módulo | Função |
|---|---|
| `src/voc_rag/build_kb.py` | Etapa 1: junta as tabelas da Olist e gera um documento por avaliação |
| `src/voc_rag/embeddings.py` | Embeddings E5 com os prefixos `query:` / `passage:` |
| `src/voc_rag/index.py` | Grava vetores e metadados no Chroma |
| `src/voc_rag/filters.py` | Uma definição de filtro, usada no Chroma e no pandas |
| `src/voc_rag/retrieve.py` | Busca, agrupamento de textos iguais e trava de semelhança |
| `src/voc_rag/stats.py` | Estatísticas exatas do recorte (para perguntas amplas) |
| `src/voc_rag/generate.py` | Prompt, montagem do contexto, mascaramento de nomes e leitura das citações |
| `src/voc_rag/pipeline.py` | Orquestra tudo e aplica as travas de recusa |
| `src/voc_rag/llm.py` | Clientes de LLM (Groq e Gemini), com espera automática no limite por minuto |
| `src/voc_rag/evaluate.py` | Avaliação com 30 perguntas fixas e comparação com BM25 |
| `app.py` | Interface Streamlit |

## Tecnologias

| Tecnologia | Uso | Por quê |
|---|---|---|
| Python 3.12, pandas, pyarrow | Preparação da base | Padrão para dados tabulares; parquet preserva tipos |
| `intfloat/multilingual-e5-small` (sentence-transformers, torch CPU) | Embeddings | Modelo aberto (MIT), multilíngue, bom em português, 384 dimensões; roda localmente em CPU (~6,5 min para a base toda), sem custo e sem enviar dados a terceiros. Versão fixada pelo commit `614241f6` |
| ChromaDB | Banco vetorial | Persistente em disco, sem servidor, com filtros de metadados na própria busca |
| Groq (`openai/gpt-oss-120b`) | Reescrita da pergunta e geração | Modelo aberto, bom em português, rápido, com plano gratuito. Trocável pelo Gemini (`LLM_PROVIDER=gemini`) |
| Streamlit | Interface | Demonstração com filtros e tabela de evidências em poucas linhas |
| rank-bm25 | Baseline | Comparar a busca semântica com a busca por palavra-chave |
| pytest | Testes | 59 testes, com modelos falsos (rodam sem internet nem GPU) |

## Preparação da base de conhecimento

Fonte: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (99.224 avaliações), publicado pela Olist sob a licença [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Os dados não estão neste repositório: baixe-os do Kaggle (ver [Como reproduzir](#como-reproduzir)). Os arquivos de `eval/` trazem trechos de avaliações, com o mesmo crédito e a mesma licença. As definições exatas de cada campo estão em [`docs/definicoes.md`](docs/definicoes.md).

| Decisão | O que foi feito | Justificativa |
|---|---|---|
| Seleção | Só avaliações com texto (título ou comentário): **42.138 documentos** | Sem texto não há "voz do cliente" para recuperar. 77% das notas 1 têm texto, contra 36-38% das notas 5: a base é naturalmente mais crítica, e isso é informado ao modelo |
| Texto | Título + ". " + comentário; quebras de linha e espaços normalizados | O título costuma resumir a queixa ("Atraso") e ajuda a busca |
| Limpeza mínima | Sai só o texto sem nenhuma letra ("...", "10"). **Não há corte por tamanho** | Um corte de 3 palavras eliminava queixas curtas e fortes como "não recebi". Lemos ~100 amostras antes de decidir |
| Valores ausentes | Título/comentário ausentes viram vazio; categoria ausente vira "" (1.079 casos); dias de atraso fica vazio quando não houve entrega (não inventamos valor) | Ausência é informação, não erro |
| Organização | Um documento por `review_id`, com a lista de pedidos (306 avaliações valem para mais de um pedido; vale o pior caso de entrega) | Evita duplicar o mesmo texto no índice |
| Textos repetidos | 35.507 textos distintos. O embedding é calculado uma vez por texto, e na busca textos iguais viram **uma evidência com contagem** ("muito bom", 549x) | Repetições ocupavam todas as vagas do contexto |
| Privacidade | Nomes de clientes no fim dos comentários ("Att, Maria") são mascarados antes de ir ao modelo e à tela | Dados pessoais não são necessários para a análise |
| Lojas | O prompt avisa que os nomes de lojas estão anonimizados com casas de Game of Thrones (lannister, stark...) | Evita o modelo tratar "lannister" como marca ou pessoa |
| Embeddings | E5 com prefixo `passage:` nos documentos e `query:` nas perguntas, vetores normalizados, distância cosseno | É como o E5 foi treinado; sem os prefixos a qualidade cai |
| Indexação | Chroma persistente, com os metadados de cada avaliação para filtro | Filtrar na busca é mais preciso do que filtrar depois |

## Pipeline RAG

1. **Transformação da pergunta.** O LLM devolve um JSON com: a pergunta está no escopo? é ampla ("principais", "padrões")? e uma versão reescrita no vocabulário de um cliente ("o produto não chegou, atraso na entrega"). A reescrita aproxima a pergunta do estilo dos comentários, numa ideia parecida com o HyDE (*Hypothetical Document Embeddings*): buscamos com um texto que se parece com a resposta, e não com a pergunta. A mesma chamada faz o roteamento: decide se a pergunta vai para a busca, para as estatísticas ou para a recusa. Se o JSON vier quebrado, o sistema segue com a pergunta original e uma regra simples para "ampla".
2. **Recuperação semântica** com os filtros escolhidos na tela, aplicados dentro do Chroma.
3. **Seleção.** A busca amplia a janela até achar *k* textos distintos (8 para perguntas específicas, 40 para amplas). Textos idênticos viram uma evidência, com a contagem de quantas vezes aparecem no recorte filtrado.
4. **Contexto.** Cada evidência entra numerada, com metadados: `[R1] (nota 1 | nao_entregue | telefonia | 2018-01 | 3x) texto`. Em perguntas amplas entra também um bloco de **estatísticas exatas** do recorte inteiro, calculadas em pandas. O modelo é proibido de calcular porcentagens a partir da amostra: os números só podem vir desse bloco.
5. **Geração** com regras explícitas: usar só as evidências, citar cada afirmação, não inventar, agrupar por tema, até 250 palavras. O prompt traz um exemplo de resposta bem citada (*few-shot*) e tem versão (`PROMPT_VERSION`, gravada em cada linha da avaliação).
   - **Segurança contra *prompt injection*:** as avaliações são texto escrito por clientes. Elas entram entre as marcas `<evidencias>` e `</evidencias>`, o prompt diz que esse conteúdo é dado e nunca instrução, e o código neutraliza qualquer marca falsa escrita dentro de uma avaliação.
6. **Conferência.** O código lê as citações da resposta e descarta a resposta se ela não citar nada ou citar uma evidência que não existe.

## Perguntas sem evidência e fora do escopo

Quatro travas, da mais barata para a mais cara:

| Trava | Quando age | O que acontece |
|---|---|---|
| Escopo | A pergunta não é sobre a experiência de compra (ex.: "preço do iPhone hoje") | Recusa sem buscar nem gerar |
| Semelhança mínima | Nenhuma avaliação passa do limite de semelhança | Recusa sem chamar o LLM, e mostra as mais próximas |
| Leitura pelo modelo | O modelo lê as evidências e nenhuma trata do assunto (responde `SEM_EVIDENCIA`) | Recusa, e mostra as mais próximas |
| Citações | A resposta não cita nada ou cita evidência inexistente | Resposta descartada; as evidências continuam visíveis |

Por que quatro? A avaliação mostrou que **a semelhança sozinha não separa** perguntas sem evidência das válidas: na rodada 4, "chatbot de atendimento" (sem evidência) teve semelhança máxima de 0,896, enquanto a pergunta válida com menor semelhança (Q20, informática com filtro) ficou em 0,886. Embeddings pequenos dão notas altas para qualquer texto do mesmo domínio. Por isso a decisão final é do modelo, que lê o conteúdo, e o limite de semelhança fica só como filtro barato.

## Metadados (desafio adicional)

Filtros disponíveis na tela e na busca: **nota**, **categoria do produto**, **situação da entrega** (no prazo, atrasado, não entregue), **"respondeu antes de receber"** e **período**.

Campos criados por nós:
- `delivery_status` e `days_late`: comparam a data real de entrega com a prevista, só pela data.
- `reviewed_before_delivery`: 4.721 clientes (nota média 1,6) responderam a pesquisa antes de o produto chegar. A data usada é a da resposta (`review_answer_timestamp`), e não a do envio da pesquisa (`review_creation_date`). É um recorte forte para entender a ansiedade com prazo.

| Situação | Avaliações | Nota média |
|---|---|---|
| No prazo | 36.420 | 4,0 |
| Atrasado | 3.829 | 1,9 |
| Não entregue | 1.889 | 1,5 |

Como os metadados melhoram a recuperação: perguntas como "o que reclamam os clientes de telefonia com nota baixa?" buscam só dentro do recorte certo, em vez de esperar que a semelhança do texto acerte a categoria e a nota. As estatísticas das perguntas amplas usam o mesmo filtro, então os números batem com o que foi buscado.

## Avaliação

30 perguntas fixas em [`eval/perguntas.json`](eval/perguntas.json):
- 15 no tema;
- 5 com filtro;
- 5 fora do escopo;
- 5 sem evidência, sobre assuntos cuja palavra-chave tem 0 menções na base: PIX, chatbot, cashback, cripto e realidade virtual.

Medimos se o sistema responde ou recusa como esperado, a **precisão das evidências** (fração das evidências usadas que é de fato relevante, por um padrão de palavras definido para cada pergunta) e a mesma precisão para uma **busca por palavra-chave (BM25)**, como linha de base, com o mesmo número de evidências.

As 30 perguntas funcionam como um *golden set*, e a precisão das evidências é uma forma simples de *context precision*.

| Métrica | Rodada 1 | Rodada 2 | Rodada 3 (prompt v3) | Rodada 4 (prompt v4) |
|---|---|---|---|---|
| Precisão das evidências, RAG | 0,85 | 0,86 | 0,81 | **0,87** |
| Precisão, BM25 com a pergunta original | 0,49 | 0,49 | 0,49 | 0,50 |
| Precisão, BM25 com a pergunta reescrita | – | 0,80 (recalculada) | 0,83 | **0,71** |
| Perguntas válidas respondidas | 14/20 | 20/20 | 20/20 | 20/20 |
| Recusas corretas | 6/10 | 10/10 | 9/10 | 10/10 |
| Respostas que usaram as estatísticas | – | 6 de 21 | 0 de 21 | 13 de 16 amplas |
| Recusas pela trava de semelhança | – | – | – | 0 |
| Tempo médio por pergunta | 17 s | 21 s | 20 s | 18 s |

Rodada 1: limite de semelhança 0,80. Rodadas 2 a 4: 0,85. Todas usam o Groq (`openai/gpt-oss-120b`); as rodadas 1 a 3 são de 09/11/2026 e a 4 é de 10/11/2026.

**Atenção à comparação:** nas rodadas 1 a 3 a precisão era medida só nas 8 primeiras evidências. Na rodada 4 ela passou a ser medida em **todas** as evidências que o modelo recebe (40 nas perguntas amplas), e o BM25 passou a buscar o mesmo número. Por isso a rodada 4 não é diretamente comparável às anteriores. Os resultados da rodada 3 ficaram guardados em `eval/rodada3_resultados.md` e `eval/rodada3_resultados.csv`.

**O que os números dizem, com honestidade:**
- **A reescrita da pergunta ajuda muito.** Com a pergunta original, o BM25 fica em 0,50. Com a mesma pergunta reescrita que o RAG usa, ele sobe para 0,71. A comparação justa é entre as duas buscas com a mesma entrada.
- **Com 40 evidências, a busca densa se destaca:** 0,87 contra 0,71 do BM25 com a pergunta reescrita. Nas 8 primeiras (rodadas 2 e 3) as duas ficavam próximas; a diferença aparece mais fundo na lista, onde a busca por palavra começa a trazer textos que só compartilham termos.
- **A métrica favorece a busca por palavra**, porque mede relevância por um padrão de palavras. Ela também pode errar: na Q14 (embalagem) a precisão do RAG foi 0,25, provavelmente porque muitas avaliações falam de embalagem com palavras que o padrão não pega (não conferimos uma a uma). Uma avaliação humana tende a valorizar mais a busca densa, que acha sinônimos ("não veio", "nunca recebi").
- **As estatísticas agora entram na resposta.** Na rodada 3 nenhuma resposta usava o bloco de estatísticas. Com o prompt v4, 13 das 16 perguntas amplas começam com uma linha "Panorama:" citando [Estatísticas]. As 3 que não usaram (Q08, Q14 e Q18) responderam só com as avaliações.
- **A trava de semelhança nunca disparou** nas 30 perguntas: as recusas vieram da trava de escopo (5) e da leitura pelo modelo, `SEM_EVIDENCIA` (5). A menor semelhança de uma pergunta válida foi 0,886 e a maior de uma sem evidência foi 0,896, então nenhum limite separa os dois grupos. A trava fica como proteção barata, mas quem decide de fato é o modelo.
- **Variação entre rodadas.** O LLM não é determinístico e são só 20 perguntas válidas, então diferenças de ±0,05 são ruído.
- **Erro da rodada 3, e o que ele ensina:** a pergunta sobre chatbot devia ser recusada, mas foi respondida. Ao ler a resposta, vimos que o modelo achou reclamações reais sobre atendimento automático ("só um robô do outro lado da linha") e as citou corretamente. O rótulo "sem evidência" tinha sido conferido só pela palavra "chatbot", então o erro é em parte do gabarito, e não só do sistema. A lição é que um *golden set* também precisa de revisão. Na rodada 4 a mesma pergunta foi recusada.
- **Limitação: viés da busca para elogios.** Na Q11 (atendimento e suporte), todas as evidências citadas na resposta são elogios, e ela termina dizendo que "não há evidências de críticas" ao atendimento, o que é falso para a base inteira (a Q01 mostra muitas reclamações de contato). A resposta é fiel às evidências que recebeu, mas não à base. Isso acontece porque a pergunta é neutra e os elogios curtos ("excelente atendimento") são os textos mais parecidos com ela. Uma correção seria buscar separadamente elogios e reclamações nas perguntas amplas.
- **Próximo passo natural:** busca híbrida, juntando a densa e o BM25 por *Reciprocal Rank Fusion*.

O que a 1ª rodada ensinou, e foi corrigido:
1. **Formato das citações:** o modelo escrevia "(R1, R4)" em vez de "[R1]", e o código descartava respostas boas. Agora os dois formatos são aceitos, e a regra foi reforçada no prompt.
2. **Perguntas sem evidência:** passavam pela trava de semelhança (ver a seção acima). Entrou a 3ª trava, de leitura pelo modelo.

Para reproduzir: `python -m voc_rag.evaluate 0.85`, que gera `eval/resultados.md` e `eval/resultados.csv`, com o texto de cada resposta.

## Exemplos de perguntas e respostas

Respostas da rodada 4 (prompt v4-2026-11-10), copiadas de `eval/resultados.csv` sem edição.

### Quais são os principais problemas relatados pelos clientes em suas avaliações?

Pergunta ampla (Q01). Status: `ok`.

> Panorama: 9.1 % das avaliações apontam entregas atrasadas e 4.5 % relatam não‑entrega [Estatísticas].
>
> **Atrasos e não‑entregas** – A maior parte das reclamações refere‑se a problemas com os Correios ou com a logística da loja, como entregas fora do prazo (R3, R4, R6, R7, R27, R38), não‑entregas (R1, R9, R10, R15, R17, R28) e entregas parciais (R14, R18, R40).  
>
> **Dificuldade de contato / comunicação** – Clientes denunciam a impossibilidade de falar com atendentes ou obter respostas, seja por telefone, e‑mail ou WhatsApp (R4, R5, R11, R13, R20, R22, R23, R31, R34, R36, R38).  
>
> **Problemas com o produto** – Há relatos de itens errados, incompletos ou com defeito, como tesoura sem afiar (R2), produto diferente do pedido (R25, R32, R36), falta de peças ou manual (R18, R34) e necessidade de buscar o item nos Correios (R19, R40).  
>
> **Falta de respeito e tratamento inadequado** – Sentimentos de desrespeito ao consumidor são frequentes, citando “falta de compromisso” (R6), “desrespeito” (R11, R15, R17, R37, R38) e atendimento péssimo (R23, R31).  
>
> **Problemas de pagamento** – Poucas avaliações tratam de questões financeiras, como cartão clonado (R39).  
>
> Em síntese, os principais problemas apontados pelos clientes são atrasos ou ausência de entrega, comunicação ineficaz com o vendedor, defeitos ou incongruências nos produtos e percepções de desrespeito no atendimento.

### Quais são os principais problemas relatados nas avaliações com nota baixa?

Pergunta com filtro (Q16). Filtro: `max_score = 2` (notas 1 e 2). Status: `ok`.

> Panorama: 80.3 % das avaliações de nota baixa são de 1 estrela e 25.6 % relatam atraso na entrega [Estatísticas].
>
> **Falta ou quantidade incorreta de produtos** – Vários clientes recebem menos itens que compraram ou quantidades divergentes na nota fiscal: “faltando 1” (R1), “compri 2 cortinas e só veio 1” (R3, R26), “recebi somente 01 unidade de 04 compradas” (R16), “veio apenas 1 de 5 unidades” (R18), “recebi 1 caneta de 2 compradas” (R30), “recebi 1 vaso de 2” (R39).
>
> **Problemas com nota fiscal** – Ausência, valores errados ou informações inválidas são citados: “entregue sem nota fiscal nem manual” (R5), “nota fiscal com valor muito mais barato” (R7), “NF com valor extremamente inferior ao pago” (R10), “chave de NF‑e inválida” (R12), “NF rasurada na quantidade” (R22), “nota fiscal incorreta quanto à quantidade” (R33).
>
> **Entrega atrasada ou não entregue** – Reclamações de não recebimento ou atraso significativo: “não recebi o produto” (R8), “entrega não realizada” (R23), “produto ainda não chegou após quase um mês” (R32), “agora não recebi o material” (R35), “esperando solução há mais de 1 mês” (R37).
>
> **Produto diferente do anunciado ou de baixa qualidade** – Divergência entre descrição e item recebido ou defeitos: “modelo 2+2 ao invés de 4+4” (R2), “novo vs remold” (R4), “produto de péssima qualidade” (R19), “problema na palmilha” (R21), “lentes com defeito” (R24), “produto falso” (R38), “marca diferente e inferior” (R40).
>
> **Dificuldade de cancelamento, troca e atendimento** – Falta de resposta e processos complicados: “dificuldade para trocar” (R6), “não respondem email, telefone não funciona” (R11), “cancelamento impossível após despacho” (R28), “sem atendimento útil” (R23), “nenhum retorno sobre cancelamento” (R17).

### O que os clientes acham do pagamento via PIX?

Pergunta sem evidência (Q26). Status: `sem_evidencia`.

> SEM_EVIDENCIA  
> Não há avaliações que abordem o pagamento via PIX nas evidências fornecidas.

### Qual é o preço do iPhone hoje?

Pergunta fora do escopo (Q21). Status: `fora_do_escopo`.

> Essa pergunta está fora do escopo da base: ela só tem avaliações de clientes da Olist (2016-2018).

## Como reproduzir

Requisitos: Python 3.11 ou superior (testado com 3.12.10), cerca de 4 GB livres e uma chave gratuita do [Groq](https://console.groq.com) (ou do Gemini). Não precisa de GPU.

```bash
git clone https://github.com/rodrigueshcluiz/tech-challenge-fase4.git
cd tech-challenge-fase4
python -m venv .venv
```

Ative o ambiente: no Windows, `.venv\Scripts\activate`; no Linux ou Mac, `source .venv/bin/activate`. Depois instale:

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

1. Baixe a base do [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) e extraia os CSVs em `data/raw/`.
2. Crie um arquivo `.env` na pasta do projeto com `GROQ_API_KEY=sua_chave`. O `.env` está no `.gitignore` e nunca vai para o repositório.
3. Rode:

```bash
python -m pytest -q                      # testes com modelos falsos, sem internet
python -m voc_rag.cli base               # etapa 1: data/processed/documents.parquet
python -m voc_rag.cli indice             # embeddings + Chroma (~7 min em CPU)
python -m voc_rag.cli testar-llm         # testa o LLM configurado (não mostra a chave)
streamlit run app.py                     # interface
python -m voc_rag.evaluate 0.85          # avaliação
```

Variáveis opcionais no `.env`: `LLM_PROVIDER=groq|gemini`, `GROQ_MODEL`, `GEMINI_MODEL`.

## Decisões técnicas e limitações

**Decisões**
- **Embeddings locais e LLM por API.** Os embeddings rodam uma vez, offline e de graça. A geração precisa de um modelo grande, e um modelo local de 7B na CPU levaria cerca de 1 minuto por resposta.
- **Troca de LLM durante o projeto.** O plano gratuito do Gemini liberava só 20 chamadas por dia, menos do que uma rodada de avaliação (cerca de 60). Por isso migramos para o Groq. O código isola o LLM em um único arquivo, e a troca não mexeu no resto do pipeline.
- **Agrupar textos iguais na busca, e não no índice.** O mesmo texto aparece com notas e datas diferentes. Se fosse agrupado no índice, os filtros perderiam informação.
- **Estatísticas em pandas, não pelo LLM.** O modelo vê no máximo 40 avaliações. Porcentagens sobre essa amostra seriam enganosas.
- **Testes com modelos falsos e testes de mutação.** Quebramos o código de propósito para confirmar que os testes pegam o erro. Assim, os testes provam a lógica do pipeline sem depender da internet.

**Limitações**
- A base é de 2016 a 2018: assuntos atuais (PIX, por exemplo) são recusados por falta de evidência.
- Clientes insatisfeitos escrevem mais, então as respostas tendem a destacar problemas.
- A precisão é medida por um padrão de palavras por pergunta, uma aproximação. Uma conferência manual de fidelidade (*faithfulness*) complementa a métrica: cada frase de uma amostra de respostas é comparada com as evidências que ela cita (`python -m voc_rag.fidelidade` gera a planilha).
- A recusa de perguntas sem evidência depende do LLM: em uma das três rodadas, uma pergunta sem evidência passou.
- No plano gratuito do Groq há um limite por minuto, e o sistema espera e tenta de novo quando bate nele.
