"""Embeddings com o multilingual-e5 (Microsoft, licença MIT).

O E5 foi treinado com prefixos: "query: " na pergunta e "passage: " no texto buscado.
Sem eles a qualidade da busca cai.
"""
import numpy as np

MODELO_PADRAO = "intfloat/multilingual-e5-small"
REVISAO_PADRAO = "614241f622f53c4eeff9890bdc4f31cfecc418b3"  # commit fixo usado no índice (08/11/2026)


class E5Embedder:
    def __init__(self, model=None, model_name=MODELO_PADRAO, revision=None, device="cpu", batch_size=64):
        self._model = model
        self.model_name, self.revision, self.device, self.batch_size = model_name, revision or REVISAO_PADRAO, device, batch_size

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # import pesado: só quando precisa

            self._model = SentenceTransformer(self.model_name, revision=self.revision, device=self.device)
        return self._model

    def _encode(self, texts):
        return np.asarray(
            self.model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 1000
            )
        )

    def embed_passages(self, texts):
        return self._encode(["passage: " + t for t in texts])

    def embed_queries(self, texts):
        return self._encode(["query: " + t for t in texts])
