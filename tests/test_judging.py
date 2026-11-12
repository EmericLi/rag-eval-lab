import pytest

from rag_eval_lab.judging import JudgeParseError, OllamaJudge, Verdict, extract_json
from rag_eval_lab.ollama_client import OllamaClient
from rag_eval_lab.types import Chunk

PASSAGES = [Chunk("d1#0", "d1", "Un volcan est une ouverture de la croûte terrestre.")]


class ScriptedTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.options = []

    def __call__(self, method, path, payload):
        self.prompts.append(payload["prompt"])
        self.options.append(payload["options"])
        return {"response": self.responses.pop(0), "prompt_eval_count": 7}


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
    assert verdict.prompt_eval_count == 14


def test_every_judge_call_sends_the_configured_context_window():
    transport = ScriptedTransport(
        [
            '{"faithful": true, "relevant": true, "reason": "ok"}',
            '{"correct": true, "reason": "ok"}',
        ]
    )
    _judge(transport, num_ctx=4096).verdict("Q ?", PASSAGES, "une réponse", "la référence")
    assert [o["num_ctx"] for o in transport.options] == [4096, 4096]


def test_judge_defaults_to_an_explicit_8192_window():
    transport = ScriptedTransport(['{"correct": true, "reason": "ok"}'])
    _judge(transport).verdict("Q ?", [], "une réponse", "la référence")
    assert transport.options[0]["num_ctx"] == 8192


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


def test_judge_retries_on_non_boolean_value_then_succeeds():
    transport = ScriptedTransport(
        [
            '{"faithful": "maybe", "relevant": true, "reason": "r"}',
            '{"faithful": true, "relevant": true, "reason": "r"}',
        ]
    )
    verdict = _judge(transport, retries=1).verdict("Q ?", PASSAGES, "a", "")
    assert (verdict.faithfulness, verdict.relevance) == (True, True)
    assert len(transport.prompts) == 2


def test_judge_raises_last_raw_output_when_every_value_is_non_boolean():
    transport = ScriptedTransport(
        [
            '{"faithful": null, "relevant": true, "reason": "r"}',
            '{"faithful": "maybe", "relevant": true, "reason": "r"}',
        ]
    )
    with pytest.raises(JudgeParseError) as excinfo:
        _judge(transport, retries=1).verdict("Q ?", PASSAGES, "a", "")
    assert excinfo.value.raw_output == '{"faithful": "maybe", "relevant": true, "reason": "r"}'
    assert len(transport.prompts) == 2


def test_judge_retries_on_missing_key_then_succeeds():
    transport = ScriptedTransport(
        [
            '{"faithful": true, "reason": "r"}',
            '{"faithful": true, "relevant": true, "reason": "r"}',
        ]
    )
    verdict = _judge(transport, retries=1).verdict("Q ?", PASSAGES, "a", "")
    assert verdict.relevance is True
    assert len(transport.prompts) == 2


def test_judge_rejects_negative_retries():
    with pytest.raises(ValueError):
        _judge(ScriptedTransport([]), retries=-1)


def test_verdict_with_whitespace_only_reference_skips_correctness():
    transport = ScriptedTransport(['{"faithful": true, "relevant": true, "reason": "r"}'])
    verdict = _judge(transport).verdict("Q ?", PASSAGES, "a", "   \n")
    assert verdict.correctness is None
    assert len(transport.prompts) == 1
