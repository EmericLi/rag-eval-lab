from rag_eval_lab.generation import OllamaGenerator
from rag_eval_lab.ollama_client import Completion, OllamaClient
from rag_eval_lab.types import Chunk


class RecordingTransport:
    def __init__(self):
        self.payloads = []

    def __call__(self, method, path, payload):
        self.payloads.append(payload)
        return {"response": "réponse générée", "eval_count": 7, "prompt_eval_count": 100}


def _generator(transport, **kwargs):
    return OllamaGenerator(OllamaClient(transport=transport), "llama3.2:latest", **kwargs)


def test_generator_sends_the_passages_and_returns_a_completion():
    transport = RecordingTransport()
    completion = _generator(transport, max_tokens=120).answer(
        "Qu'est-ce qu'un volcan ?", [Chunk("d1#0", "d1", "Un volcan est une ouverture.")]
    )
    assert isinstance(completion, Completion)
    assert completion.text == "réponse générée"
    prompt = transport.payloads[0]["prompt"]
    assert "Un volcan est une ouverture." in prompt
    assert "Qu'est-ce qu'un volcan ?" in prompt
    assert transport.payloads[0]["options"]["num_predict"] == 120
    assert transport.payloads[0]["options"]["temperature"] == 0.0


def test_generator_without_passages_uses_the_memory_prompt():
    transport = RecordingTransport()
    _generator(transport).answer("Question ?", [])
    assert "connaissances" in transport.payloads[0]["prompt"]


def test_generator_exposes_its_model():
    assert _generator(RecordingTransport()).model == "llama3.2:latest"
