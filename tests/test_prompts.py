from rag_eval_lab.prompts import (
    ANSWER_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    render_answer_prompt,
    render_correctness_prompt,
    render_grounding_prompt,
)


def test_answer_prompt_numbers_passages_and_carries_the_question():
    prompt = render_answer_prompt("Qu'est-ce qu'un volcan ?", ["premier extrait", "second extrait"])
    assert "Qu'est-ce qu'un volcan ?" in prompt
    assert "[1] premier extrait" in prompt
    assert "[2] second extrait" in prompt
    assert prompt.index("[1]") < prompt.index("[2]")


def test_answer_prompt_without_passages_asks_for_memory():
    prompt = render_answer_prompt("Question ?", [])
    assert "connaissances" in prompt
    assert "[1]" not in prompt


def test_grounding_prompt_contains_answer_passages_and_json_keys():
    prompt = render_grounding_prompt("Q ?", ["extrait"], "la réponse")
    assert "extrait" in prompt and "la réponse" in prompt
    for key in ("faithful", "relevant", "reason"):
        assert key in prompt


def test_correctness_prompt_contains_the_reference():
    prompt = render_correctness_prompt("Q ?", "réponse générée", "réponse du prof")
    assert "réponse du prof" in prompt
    assert "correct" in prompt
    assert "réponse générée" in prompt


def test_prompt_versions_are_short_and_distinct():
    assert len(ANSWER_PROMPT_VERSION) == 8
    assert len(JUDGE_PROMPT_VERSION) == 8
    assert ANSWER_PROMPT_VERSION != JUDGE_PROMPT_VERSION
