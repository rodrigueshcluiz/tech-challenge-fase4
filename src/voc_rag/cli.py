"""Comandos do projeto (rodar dentro da pasta voc-rag):

  python -m voc_rag.cli testar-gemini        # 1 chamada de teste e lista de modelos (não mostra a chave)
  python -m voc_rag.cli base                 # etapa 1: data/raw -> data/processed/documents.parquet
  python -m voc_rag.cli indice               # etapa 2: embeddings + índice Chroma em data/chroma
  python -m voc_rag.cli perguntar "texto"    # etapas 3 e 4 no terminal
"""
import os
import sys
import time

import pandas as pd

DOCS, CHROMA = "data/processed/documents.parquet", "data/chroma"
REVISAO_E5 = os.environ.get("E5_REVISION")  # fixar o commit do modelo para reprodutibilidade


def testar_gemini():
    from voc_rag.llm import make_llm

    llm = make_llm()
    print("Provedor:", type(llm).__name__, "| modelo:", llm.model)
    if hasattr(llm, "client"):  # só o Gemini lista modelos
        nomes = [m.name for m in llm.client.models.list()]
        print("Modelos com 'flash' liberados:", [n for n in nomes if "flash" in n][:10])
    if hasattr(llm, "models"):
        print("Modelos do Groq liberados:", llm.models())
    print("Resposta:", llm("Responda só com a palavra OK.", "Teste de conexão."))


def indice():
    from voc_rag.embeddings import E5Embedder
    from voc_rag.index import build_index

    t = time.time()
    col = build_index(pd.read_parquet(DOCS), E5Embedder(revision=REVISAO_E5), CHROMA)
    print(f"{col.count()} vetores em {CHROMA} ({time.time() - t:.0f}s)")


def perguntar(pergunta):
    from voc_rag.embeddings import E5Embedder
    from voc_rag.filters import Filters
    from voc_rag.index import open_index
    from voc_rag.llm import make_llm
    from voc_rag.pipeline import answer

    r = answer(pergunta, Filters(), open_index(CHROMA), E5Embedder(revision=REVISAO_E5), make_llm(), pd.read_parquet(DOCS))
    print(f"[{r['status']}] busca: {r['query']}\n\n{r['answer']}\n")
    for i, e in enumerate(r["evidences"], 1):
        print(f"[R{i}] sim={e.similarity:.3f} nota={e.score} {e.delivery_status} {e.count}x {e.review_ids[0]}: {e.text}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "testar-gemini":
        testar_gemini()
    elif cmd == "base":
        from voc_rag.build_kb import main

        main()
    elif cmd == "indice":
        indice()
    elif cmd == "perguntar":
        perguntar(" ".join(sys.argv[2:]))
    else:
        print(__doc__)
