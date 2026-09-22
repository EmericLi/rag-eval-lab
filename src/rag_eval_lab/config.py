"""Experiment configuration, validated before any computation."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_eval_lab.cache import make_key
from rag_eval_lab.chunking import parse_chunker
from rag_eval_lab.factory import build_reranker, build_retriever

SpecField = Literal["chunker", "retriever", "reranker"]


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["alloprof", "jsonl"]
    path: str | None = None
    split: str = "test"
    n_queries: int | None = Field(default=None, gt=0)
    n_docs: int | None = Field(default=None, gt=0)
    seed: int = 42

    @model_validator(mode="after")
    def _check_path(self) -> DatasetConfig:
        if self.name == "jsonl" and not self.path:
            raise ValueError("dataset 'jsonl' requires a 'path'")
        return self


class GridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunker: list[str] = Field(min_length=1)
    retriever: list[str] = Field(min_length=1)
    reranker: list[str] = Field(default_factory=lambda: ["none"], min_length=1)


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunker: str
    retriever: str
    reranker: str = "none"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    grid: GridConfig
    k: int = Field(default=10, gt=0)
    ks: list[int] = Field(default_factory=lambda: [5, 10], min_length=1)
    candidates: int = Field(default=100, gt=0)
    cache_dir: str = ".cache/embeddings"
    exclude: list[dict[SpecField, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if any(k <= 0 or k > self.k for k in self.ks):
            raise ValueError(f"all ks must be in [1, k={self.k}], got {self.ks}")
        for spec in self.grid.chunker:
            parse_chunker(spec)
        for spec in self.grid.retriever:
            build_retriever(spec)
        for spec in self.grid.reranker:
            build_reranker(spec)
        return self

    def expand(self) -> list[RunSpec]:
        specs = []
        for chunker, retriever, reranker in itertools.product(
            self.grid.chunker, self.grid.retriever, self.grid.reranker
        ):
            spec = RunSpec(chunker=chunker, retriever=retriever, reranker=reranker)
            values = spec.model_dump()
            if any(all(values[f] == v for f, v in rule.items()) for rule in self.exclude):
                continue
            specs.append(spec)
        return specs

    def run_id(self, spec: RunSpec) -> str:
        return make_key(
            self.dataset.model_dump_json(),
            str(self.k),
            str(self.candidates),
            spec.chunker,
            spec.retriever,
            spec.reranker,
        )[:12]


def load_config(path: Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as f:
        return ExperimentConfig.model_validate(yaml.safe_load(f))
