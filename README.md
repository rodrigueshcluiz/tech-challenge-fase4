# Voz do Cliente com RAG (Olist)

Tech Challenge da Fase 4 da POSTECH (IA para Devs / AI Scientist).

Objetivo: responder perguntas de negócio sobre a experiência de compra usando só as avaliações reais de clientes da base pública da Olist, citando as evidências e recusando quando não houver evidência ou a pergunta fugir do assunto.

Estrutura planejada:

| Pasta | Conteúdo |
|---|---|
| `src/voc_rag/` | Preparação da base, embeddings, índice, recuperação e geração |
| `tests/` | Testes automáticos com modelos falsos |
| `eval/` | Perguntas fixas e resultados da avaliação |
| `docs/` | Definições dos campos da base |
| `data/` | Dados brutos e processados (fora do repositório; ver `.gitignore`) |
