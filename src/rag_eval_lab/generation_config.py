"""Configuration for the generation benchmark, validated before any model call."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_eval_lab.cache import make_key
from rag_eval_lab.chunking import parse_chunker
from rag_eval_lab.config import DatasetConfig
from rag_eval_lab.factory import build_retriever
from rag_eval_lab.prompts import ANSWER_PROMPT_VERSION


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = 0.0
    max_tokens: int = Field(default=300, gt=0)
    timeout_s: float = Field(default=180.0, gt=0)


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = 0.0
    max_tokens: int = Field(default=200, gt=0)
    timeout_s: float = Field(default=300.0, gt=0)


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["empty", "retrieved", "oracle"]
    chunker: str | None = None
    retriever: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ContextConfig:
        if self.kind == "retrieved":
            if not self.chunker or not self.retriever:
                raise ValueError(
                    f"context '{self.name}': kind 'retrieved' requires 'chunker' and 'retriever'"
                )
            parse_chunker(self.chunker)
            build_retriever(self.retriever)
        elif self.chunker or self.retriever:
            raise ValueError(
                f"context '{self.name}': kind '{self.kind}' takes no chunker or retriever"
            )
        return self


class AnnotationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(default=60, gt=0)
    seed: int = 7


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    contexts: list[ContextConfig] = Field(min_length=1)
    generator: GeneratorConfig
    judge: JudgeConfig
    n_passages: int = Field(default=5, gt=0)
    oracle_chunker: str = "fixed_1000_200"
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    cache_dir: str = ".cache/embeddings"

    @model_validator(mode="after")
    def _check(self) -> GenerationConfig:
        names = [c.name for c in self.contexts]
        if len(set(names)) != len(names):
            raise ValueError(f"context names must be unique, got {names}")
        parse_chunker(self.oracle_chunker)
        return self

    def answer_id(self, context_name: str, query_id: str) -> str:
        """Resume key covering everything that shapes the answer, not just the context name.

        Editing a context's definition (or the generator settings) without renaming it
        therefore yields new ids, so a resumed run regenerates instead of reusing stale answers.
        """
        context = next((c for c in self.contexts if c.name == context_name), None)
        if context is None:
            raise ValueError(f"unknown context '{context_name}'")
        return make_key(
            context_name,
            context.kind,
            str(context.chunker),
            str(context.retriever),
            str(self.n_passages),
            self.oracle_chunker,
            query_id,
            self.generator.model,
            str(self.generator.temperature),
            str(self.generator.max_tokens),
            ANSWER_PROMPT_VERSION,
        )[:12]


def load_generation_config(path: Path) -> GenerationConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return GenerationConfig.model_validate(yaml.safe_load(handle))
