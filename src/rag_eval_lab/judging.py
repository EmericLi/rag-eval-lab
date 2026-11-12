"""LLM-as-a-judge: three binary verdicts per answer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from rag_eval_lab.ollama_client import OllamaClient
from rag_eval_lab.prompts import render_correctness_prompt, render_grounding_prompt
from rag_eval_lab.types import Chunk


class JudgeParseError(RuntimeError):
    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


@dataclass(frozen=True)
class Verdict:
    faithfulness: bool | None
    relevance: bool | None
    correctness: bool | None
    rationales: dict[str, str]
    latency_ms: float
    prompt_eval_count: int = 0


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise JudgeParseError("no JSON object in judge output", text)
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"invalid JSON in judge output: {exc}", text) from exc
    if not isinstance(value, dict):
        raise JudgeParseError("judge output is not a JSON object", text)
    return value


def _as_bool(value: object, key: str, raw: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise JudgeParseError(f"key '{key}' is not a boolean: {value!r}", raw)


class Judge(Protocol):
    model: str

    def verdict(
        self, question: str, passages: list[Chunk], answer: str, reference: str
    ) -> Verdict: ...


class OllamaJudge:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        max_tokens: int = 200,
        retries: int = 2,
        num_ctx: int = 8192,
    ) -> None:
        if retries < 0:
            raise ValueError(f"Invalid judge: retries={retries}")
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.retries = retries
        self.num_ctx = num_ctx

    def _ask(
        self, prompt: str, keys: tuple[str, ...], bool_keys: tuple[str, ...] = ()
    ) -> tuple[dict, float, int]:
        last: JudgeParseError | None = None
        latency = 0.0
        prompt_tokens = 0
        for _ in range(self.retries + 1):
            completion = self.client.generate(
                self.model,
                prompt,
                temperature=0.0,
                max_tokens=self.max_tokens,
                json_mode=True,
                num_ctx=self.num_ctx,
            )
            latency += completion.latency_ms
            prompt_tokens += completion.prompt_eval_count
            try:
                payload = extract_json(completion.text)
                result = {key: payload[key] for key in keys}
                for key in bool_keys:
                    result[key] = _as_bool(result[key], key, completion.text)
                return (result, latency, prompt_tokens)
            except (JudgeParseError, KeyError) as exc:
                last = (
                    exc
                    if isinstance(exc, JudgeParseError)
                    else JudgeParseError(f"missing key {exc}", completion.text)
                )
        raise last  # type: ignore[misc]

    def verdict(self, question: str, passages: list[Chunk], answer: str, reference: str) -> Verdict:
        faithfulness: bool | None = None
        relevance: bool | None = None
        correctness: bool | None = None
        rationales: dict[str, str] = {}
        latency = 0.0
        prompt_tokens = 0

        if passages:
            prompt = render_grounding_prompt(question, [p.text for p in passages], answer)
            payload, took, used = self._ask(
                prompt, ("faithful", "relevant", "reason"), bool_keys=("faithful", "relevant")
            )
            latency += took
            prompt_tokens += used
            faithfulness = payload["faithful"]
            relevance = payload["relevant"]
            rationales["faithfulness"] = str(payload["reason"])

        if reference.strip():
            prompt = render_correctness_prompt(question, answer, reference)
            payload, took, used = self._ask(prompt, ("correct", "reason"), bool_keys=("correct",))
            latency += took
            prompt_tokens += used
            correctness = payload["correct"]
            rationales["correctness"] = str(payload["reason"])

        return Verdict(faithfulness, relevance, correctness, rationales, latency, prompt_tokens)
