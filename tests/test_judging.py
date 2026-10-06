import pytest

from rag_eval_lab.judging import JudgeParseError, OllamaJudge, Verdict, extract_json
from rag_eval_lab.ollama_client import OllamaClient
from rag_eval_lab.types import Chunk

PASSAGES = [Chunk("d1#0", "d1", "Un volcan est une ouverture de la croûte terrestre.")]


class ScriptedTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, method, path, payload):
        self.prompts.append(payload["prompt"])
        return {"response": self.responses.pop(0)}


def _judge(transport, **kwargs):
    return OllamaJudge(OllamaClient(transport=transport, retries=0), "llama3.1:8B", **kwargs)


def test_extract_json_tolerates_surrounding_text():
    assert extract_json('Voici : {"correct": true} merci') == {"correct": True}


def test_extract_json_rejects_garbage():
    with pytest.raises(JudgeParseError):
        extract_json("pas de json ici")


def test_verdict_combines_both_calls():
    transport = ScriptedTransport(
        [
            '{"faithful": true, "relevant": true, "reason": "appuyé par [1]"}',
            '{"correct": false, "reason": "contredit la référence"}',
        ]
    )
    verdict = _judge(transport).verdict("Q ?", PASSAGES, "une réponse", "la référence")
    assert isinstance(verdict, Verdict)
    assert (verdict.faithfulness, verdict.relevance, verdict.correctness) == (True, True, False)
    assert verdict.rationales["faithfulness"] == "appuyé par [1]"
    assert verdict.rationales["correctness"] == "contredit la référence"
    assert verdict.latency_ms > 0
    assert len(transport.prompts) == 2


def test_verdict_without_passages_skips_grounding():
    transport = ScriptedTransport(['{"correct": true, "reason": "ok"}'])
    verdict = _judge(transport).verdict("Q ?", [], "une réponse", "la référence")
    assert (verdict.faithfulness, verdict.relevance) == (None, None)
    assert verdict.correctness is True
    assert len(transport.prompts) == 1


def test_verdict_without_reference_skips_correctness():
    transport = ScriptedTransport(['{"faithful": false, "relevant": true, "reason": "r"}'])
    verdict = _judge(transport).verdict("Q ?", PASSAGES, "une réponse", "")
    assert verdict.correctness is None
    assert verdict.faithfulness is False


def test_judge_retries_on_invalid_json_then_succeeds():
    transport = ScriptedTransport(
        [
            "je ne sais pas",
            '{"faithful": true, "relevant": false, "reason": "r"}',
            '{"correct": true, "reason": "r"}',
        ]
    )
    verdict = _judge(transport, retries=1).verdict("Q ?", PASSAGES, "a", "ref")
    assert verdict.relevance is False
    assert len(transport.prompts) == 3


def test_judge_raises_with_the_raw_output_after_retries():
    transport = ScriptedTransport(["bruit", "encore du bruit"])
    with pytest.raises(JudgeParseError) as excinfo:
        _judge(transport, retries=1).verdict("Q ?", PASSAGES, "a", "ref")
    assert excinfo.value.raw_output == "encore du bruit"


def test_judge_coerces_string_booleans():
    transport = ScriptedTransport(
        [
            '{"faithful": "true", "relevant": "false", "reason": "r"}',
            '{"correct": "true", "reason": "r"}',
        ]
    )
    verdict = _judge(transport).verdict("Q ?", PASSAGES, "a", "ref")
    assert (verdict.faithfulness, verdict.relevance, verdict.correctness) == (True, False, True)
