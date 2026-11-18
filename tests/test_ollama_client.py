import io
import urllib.error

import pytest

from rag_eval_lab.ollama_client import Completion, OllamaClient, OllamaError


class FakeTransport:
    def __init__(self, responses, fail_times=0, error=None):
        self.responses = responses
        self.fail_times = fail_times
        self.error = error if error is not None else TimeoutError("timeout")
        self.calls = []

    def __call__(self, method, path, payload):
        self.calls.append((method, path, payload))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        return self.responses[path]


def test_generate_builds_payload_and_parses_response():
    transport = FakeTransport(
        {
            "/api/generate": {
                "response": "  une réponse  ",
                "eval_count": 12,
                "prompt_eval_count": 345,
            }
        }
    )
    client = OllamaClient(transport=transport)
    completion = client.generate("llama3.2:latest", "prompt", max_tokens=50)
    assert isinstance(completion, Completion)
    assert completion.text == "une réponse"
    assert (completion.eval_count, completion.prompt_eval_count) == (12, 345)
    assert completion.latency_ms >= 0
    method, path, payload = transport.calls[0]
    assert (method, path) == ("POST", "/api/generate")
    assert payload["model"] == "llama3.2:latest"
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.0, "num_predict": 50, "seed": 0}
    assert "format" not in payload


def test_generate_sends_num_ctx_when_given():
    transport = FakeTransport({"/api/generate": {"response": "ok"}})
    OllamaClient(transport=transport).generate("m", "p", num_ctx=8192)
    assert transport.calls[0][2]["options"]["num_ctx"] == 8192


def test_generate_omits_num_ctx_by_default():
    transport = FakeTransport({"/api/generate": {"response": "ok"}})
    OllamaClient(transport=transport).generate("m", "p")
    assert "num_ctx" not in transport.calls[0][2]["options"]


def test_generate_json_mode_sets_format():
    transport = FakeTransport({"/api/generate": {"response": "{}"}})
    OllamaClient(transport=transport).generate("m", "p", json_mode=True)
    assert transport.calls[0][2]["format"] == "json"


def test_generate_retries_then_succeeds():
    transport = FakeTransport({"/api/generate": {"response": "ok"}}, fail_times=2)
    client = OllamaClient(transport=transport, retries=2)
    assert client.generate("m", "p").text == "ok"
    assert len(transport.calls) == 3


def test_generate_raises_after_exhausting_retries():
    transport = FakeTransport({"/api/generate": {"response": "ok"}}, fail_times=5)
    with pytest.raises(OllamaError, match="ollama serve"):
        OllamaClient(transport=transport, retries=1).generate("m", "p")


def _http_error(code, body=b""):
    return urllib.error.HTTPError(
        url="http://x/api/generate",
        code=code,
        msg="error",
        hdrs=None,
        fp=io.BytesIO(body),
    )


def test_http_404_is_not_retried_and_reports_the_body():
    error = _http_error(404, b'{"error":"model not found"}')
    transport = FakeTransport({}, fail_times=5, error=error)
    with pytest.raises(OllamaError) as excinfo:
        OllamaClient(transport=transport, retries=2).generate("m", "p")
    message = str(excinfo.value)
    assert "404" in message
    assert "model not found" in message
    assert "ollama serve" not in message
    assert len(transport.calls) == 1


def test_http_500_is_retried_then_reports_unreachable():
    error = _http_error(500, b"boom")
    transport = FakeTransport({}, fail_times=10, error=error)
    with pytest.raises(OllamaError, match="ollama serve"):
        OllamaClient(transport=transport, retries=2).generate("m", "p")
    assert len(transport.calls) == 3


def test_generate_rejects_a_response_without_text():
    transport = FakeTransport({"/api/generate": {"error": "boom"}})
    with pytest.raises(OllamaError, match="unexpected response"):
        OllamaClient(transport=transport).generate("m", "p")


def test_model_digest_finds_the_model():
    transport = FakeTransport(
        {
            "/api/tags": {
                "models": [
                    {"name": "llama3.1:8B", "digest": "abcdef0123456789"},
                    {"name": "llama3.2:latest", "digest": "0123456789abcdef"},
                ]
            }
        }
    )
    assert OllamaClient(transport=transport).model_digest("llama3.2:latest") == "0123456789ab"


def test_model_digest_ignores_case():
    # configs/v2.yaml says `llama3.1:8B`; the README pulls `llama3.1:8b`
    transport = FakeTransport(
        {"/api/tags": {"models": [{"name": "llama3.1:8b", "digest": "f" * 16}]}}
    )
    assert OllamaClient(transport=transport).model_digest("llama3.1:8B") == "f" * 12


def test_model_digest_names_the_pull_command():
    transport = FakeTransport({"/api/tags": {"models": []}})
    with pytest.raises(OllamaError, match="ollama pull mistral"):
        OllamaClient(transport=transport).model_digest("mistral")
