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


def test_model_digest_names_the_pull_command():
    transport = FakeTransport({"/api/tags": {"models": []}})
    with pytest.raises(OllamaError, match="ollama pull mistral"):
        OllamaClient(transport=transport).model_digest("mistral")
