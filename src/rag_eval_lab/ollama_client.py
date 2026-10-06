"""Minimal HTTP client for a local Ollama server (standard library only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_HOST = "http://127.0.0.1:11434"

Transport = Callable[[str, str, dict | None], dict]


class OllamaError(RuntimeError):
    """Ollama is unreachable, the model is missing, or the response is unusable."""


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float
    eval_count: int
    prompt_eval_count: int


def _http_transport(host: str, timeout_s: float) -> Transport:
    def transport(method: str, path: str, payload: dict | None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{host}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    return transport


def _error_body(exc: urllib.error.HTTPError, limit: int = 300) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:  # the body is best-effort context only
        body = ""
    return body[:limit] or "<empty body>"


class OllamaClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        timeout_s: float = 180.0,
        retries: int = 2,
        transport: Transport | None = None,
    ) -> None:
        self.host = host
        self.timeout_s = timeout_s
        self.retries = retries
        self._transport = transport or _http_transport(host, timeout_s)

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                return self._transport(method, path, payload)
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    raise OllamaError(
                        f"Ollama returned HTTP {exc.code} for {path}: {_error_body(exc)}"
                    ) from exc
                last = exc  # 5xx: transient server-side failure, retry
            except OSError as exc:  # URLError and TimeoutError are OSError subclasses
                last = exc
        raise OllamaError(
            f"cannot reach Ollama at {self.host} ({last}). Start it with `ollama serve`."
        ) from last

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 300,
        json_mode: bool = False,
        seed: int = 0,
    ) -> Completion:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
        }
        if json_mode:
            payload["format"] = "json"
        start = time.perf_counter()
        data = self._call("POST", "/api/generate", payload)
        latency_ms = (time.perf_counter() - start) * 1000
        if "response" not in data:
            raise OllamaError(f"unexpected response from Ollama for model '{model}': {data}")
        return Completion(
            text=str(data["response"]).strip(),
            latency_ms=latency_ms,
            eval_count=int(data.get("eval_count", 0)),
            prompt_eval_count=int(data.get("prompt_eval_count", 0)),
        )

    def model_digest(self, model: str) -> str:
        data = self._call("GET", "/api/tags")
        for entry in data.get("models", []):
            if model in (entry.get("name"), entry.get("model")):
                return str(entry.get("digest", "unknown"))[:12]
        raise OllamaError(f"model '{model}' is not installed. Run: ollama pull {model}")
