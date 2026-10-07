"""Answer generation on top of a context source."""

from __future__ import annotations

from typing import Protocol

from rag_eval_lab.ollama_client import Completion, OllamaClient
from rag_eval_lab.prompts import render_answer_prompt
from rag_eval_lab.types import Chunk


class Generator(Protocol):
    model: str

    def answer(self, question: str, passages: list[Chunk]) -> Completion: ...


class OllamaGenerator:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
        num_ctx: int = 8192,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_ctx = num_ctx

    def answer(self, question: str, passages: list[Chunk]) -> Completion:
        prompt = render_answer_prompt(question, [p.text for p in passages])
        return self.client.generate(
            self.model,
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            num_ctx=self.num_ctx,
        )
