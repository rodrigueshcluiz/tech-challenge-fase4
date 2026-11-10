"""Clientes de LLM (Groq e Gemini). As chaves vêm do arquivo .env, nunca do código."""
import os
from pathlib import Path

MODELO_PADRAO = "gemini-3.5-flash"  # testado em 08/11/2026; o 2.5-flash não aceita mais contas novas


def load_env(path=".env"):
    p = Path(path)
    if p.exists():
        for linha in p.read_text(encoding="utf-8").splitlines():
            if "=" in linha and not linha.strip().startswith("#"):
                k, v = linha.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


class GeminiLLM:
    def __init__(self, model=None, temperature=0.2):
        from google import genai

        load_env()
        model = model or os.environ.get("GEMINI_MODEL", MODELO_PADRAO)
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model, self.temperature = model, temperature

    def __call__(self, system, prompt, json_mode=False):
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        return self.client.models.generate_content(model=self.model, contents=prompt, config=cfg).text


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELO_PADRAO = "openai/gpt-oss-120b"  # modelo aberto; o llama-3.3-70b saiu da conta em 09/11/2026


class GroqLLM:
    """Cliente do Groq (API no formato da OpenAI) só com a biblioteca padrão: nada novo para instalar.

    A chave vem de GROQ_API_KEY no .env. `post` pode ser trocado nos testes para não usar a internet.
    """

    def __init__(self, model=None, temperature=0.2, post=None):
        load_env()
        self.model = model or os.environ.get("GROQ_MODEL", GROQ_MODELO_PADRAO)
        self.temperature = temperature
        self._key = os.environ["GROQ_API_KEY"]
        self._post = post or self._http_post

    def payload(self, system, prompt, json_mode=False):
        p = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if json_mode:
            p["response_format"] = {"type": "json_object"}
        return p

    def _http_post(self, payload):
        import json
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            GROQ_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "User-Agent": "voc-rag"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # mostra o motivo que o Groq deu (o corpo não traz a chave)
            raise RuntimeError(f"Groq HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from None

    def models(self):
        """Modelos liberados para esta chave (GET /models)."""
        import json
        import urllib.request

        req = urllib.request.Request(GROQ_URL.replace("chat/completions", "models"),
                                     headers={"Authorization": f"Bearer {self._key}", "User-Agent": "voc-rag"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return sorted(m["id"] for m in json.loads(r.read().decode("utf-8"))["data"])

    def __call__(self, system, prompt, json_mode=False, tentativas=6, dormir=None):
        """Com limite por minuto (429), espera o tempo que o Groq pede e tenta de novo."""
        import re
        import time

        dormir = dormir or time.sleep
        for i in range(tentativas):
            try:
                return self._post(self.payload(system, prompt, json_mode))["choices"][0]["message"]["content"]
            except RuntimeError as e:
                if "HTTP 429" not in str(e) or i == tentativas - 1:
                    raise
                m = re.search(r"try again in ([\d.]+)s", str(e))
                dormir(float(m.group(1)) + 1 if m else 10 * (i + 1))


def make_llm():
    """LLM_PROVIDER=groq|gemini no .env. Sem essa linha: Groq se houver GROQ_API_KEY, senão Gemini."""
    load_env()
    prov = os.environ.get("LLM_PROVIDER") or ("groq" if os.environ.get("GROQ_API_KEY") else "gemini")
    return GroqLLM() if prov == "groq" else GeminiLLM()
