# rag-eval-lab v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire un outil CLI Python qui benchmarke des pipelines de retrieval RAG (chunking × BM25/dense/hybride × reranker) sur AlloprofRetrieval et produit un rapport qualité/latence.

**Architecture:** Briques indépendantes à interface commune (`Chunker`, `Retriever`, `Encoder`, `Reranker`) assemblées par `RetrievalPipeline`. Un runner lit une grille YAML validée par pydantic, exécute chaque configuration (avec reprise et cache d'embeddings), écrit `results.jsonl`, puis un module de rapport produit Markdown + graphiques.

**Tech Stack:** Python ≥ 3.11 (dev en 3.13), uv, pydantic v2, typer, numpy, rank-bm25, matplotlib, pytest, ruff ; extra `models` : sentence-transformers, datasets (Hugging Face).

**Spec :** `docs/design/specs/2026-09-21-rag-eval-lab-design.md`

## Global Constraints

- Code, docstrings, README et rapport en **anglais** ; plans et specs en français.
- Tout tourne en **CPU local**, aucun appel réseau pendant une expérience (hors téléchargement initial modèles/données).
- Les tests (`pytest`) ne doivent **jamais** télécharger de modèle ni de données : ils utilisent l'encodeur `hashing` et des faux modèles injectés.
- `sentence-transformers` et `datasets` ne sont importés **qu'à l'intérieur des fonctions** qui en ont besoin (imports paresseux) ; ils sont dans l'extra `models`.
- Tailles de chunks en **caractères**.
- Dataset : `lyon-nlp/alloprof`, configs `documents` (split `test`, champs `uuid`, `title`, `text`) et `queries` (split `test`, champs `id`, `text`, `relevant`).
- Reproduction : `uv sync --extra models`, `uv run rel run configs/v1.yaml`, `uv run rel report results/v1`.
- Grille v1 complète < 2 h sur Ryzen 7 8845HS CPU, sinon réduire `n_queries`.
- Commits : messages conventionnels (`feat:`, `test:`, `docs:`, `chore:`), **sans** ligne `Co-Authored-By` (auteur unique : l'utilisateur). Identité git du dépôt : `EmericLi <145478297+EmericLi@users.noreply.github.com>` (déjà configurée, ne pas modifier).
- Toutes les commandes se lancent depuis la racine `C:\Users\Emeric\travail\portofolio\rag-eval-lab`.

## File Structure

```
rag-eval-lab/
├── pyproject.toml              # deps, extra "models", script "rel", ruff/pytest config
├── .python-version             # 3.13
├── .gitignore
├── README.md                   # README recruteur (Task 13)
├── .github/workflows/ci.yml    # ruff + pytest
├── configs/
│   ├── smoke.yaml              # petite grille pour valider (Task 12)
│   └── v1.yaml                 # grille complète (Task 13)
├── docs/results-v1.md          # analyse détaillée (Task 13)
├── src/rag_eval_lab/
│   ├── __init__.py             # __version__
│   ├── types.py                # Document, Query, Qrels, Dataset, Chunk, Hit
│   ├── text.py                 # tokenize()
│   ├── metrics.py              # recall/mrr/ndcg + evaluate()
│   ├── chunking.py             # NoChunker, FixedChunker, RecursiveChunker, parse_chunker()
│   ├── cache.py                # make_key(), EmbeddingCache
│   ├── encoders.py             # Encoder protocol, HashingEncoder, SentenceTransformerEncoder
│   ├── retrieval.py            # BM25Retriever, DenseRetriever, HybridRetriever
│   ├── reranking.py            # CrossEncoderReranker
│   ├── pipeline.py             # RetrievalPipeline, aggregate_to_docs()
│   ├── factory.py              # build_encoder/retriever/reranker à partir des chaînes de spec
│   ├── config.py               # pydantic: DatasetConfig, GridConfig, RunSpec, ExperimentConfig, load_config()
│   ├── data.py                 # load_jsonl, load_alloprof, sample_dataset, load_benchmark
│   ├── experiment.py           # run_experiment() + environment_info()
│   ├── report.py               # load_results, pareto_front, build_report
│   └── cli.py                  # typer app: run, report
└── tests/
    ├── conftest.py             # fixtures mini_dataset, mini_jsonl
    ├── test_types.py
    ├── test_metrics.py
    ├── test_chunking.py
    ├── test_cache_encoders.py
    ├── test_retrieval.py
    ├── test_pipeline.py
    ├── test_config.py
    ├── test_data.py
    ├── test_experiment.py
    ├── test_report.py
    └── test_cli.py
```

---

### Task 0: Installer uv (prérequis, demande l'accord de l'utilisateur)

`uv` n'est pas installé sur la machine. **Demander l'accord de l'utilisateur avant d'installer.**

- [ ] **Step 1: Installer uv**

Run (PowerShell) : `winget install --id=astral-sh.uv -e`
Alternative si winget indisponible : `python -m pip install --user uv`

- [ ] **Step 2: Vérifier (nouveau terminal si le PATH a changé)**

Run: `uv --version`
Expected: `uv 0.x.y` (toute version ≥ 0.5)

---

### Task 1: Scaffolding du projet, types et CI

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `README.md`, `.github/workflows/ci.yml`
- Create: `src/rag_eval_lab/__init__.py`, `src/rag_eval_lab/types.py`
- Create: `tests/conftest.py`, `tests/test_types.py`

**Interfaces:**
- Produces (`rag_eval_lab.types`) :
  - `Document(id: str, text: str)` (dataclass frozen)
  - `Query(id: str, text: str)` (dataclass frozen)
  - `Qrels = dict[str, dict[str, int]]` (query_id → {doc_id: gain})
  - `Dataset(name: str, documents: list[Document], queries: list[Query], qrels: Qrels)` (dataclass frozen)
  - `Chunk(id: str, doc_id: str, text: str)` (dataclass frozen)
  - `Hit(chunk: Chunk, score: float)` (dataclass frozen)
- Produces (fixtures pytest) : `mini_dataset -> Dataset` (20 docs `d0..d19`, 5 requêtes `q0..q4`, `qrels["qi"] == {"di": 1}`), `mini_jsonl -> Path` (dossier contenant `corpus.jsonl` et `queries.jsonl`), constante `TOPICS` dans `tests/conftest.py`.

- [ ] **Step 1: Créer `pyproject.toml`**

```toml
[project]
name = "rag-eval-lab"
version = "0.1.0"
description = "Benchmark RAG retrieval pipelines (chunking, BM25, dense, hybrid, reranking) on French data."
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [{ name = "EmericLi" }]
dependencies = [
    "numpy>=1.26",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "rank-bm25>=0.2.2",
    "typer>=0.12",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
models = [
    "sentence-transformers>=3.0",
    "datasets>=2.19",
]

[project.scripts]
rel = "rag_eval_lab.cli:app"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rag_eval_lab"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Créer `.python-version`, `.gitignore`, `README.md` minimal**

`.python-version` :
```
3.13
```

`.gitignore` :
```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
.cache/
results/*/runs/
```

`README.md` (provisoire, remplacé en Task 13) :
```markdown
# rag-eval-lab

Benchmark RAG retrieval pipelines on French data. Work in progress.
```

- [ ] **Step 3: Créer `src/rag_eval_lab/__init__.py` et `src/rag_eval_lab/types.py`**

`src/rag_eval_lab/__init__.py` :
```python
"""Benchmark RAG retrieval pipelines on French data."""

__version__ = "0.1.0"
```

`src/rag_eval_lab/types.py` :
```python
"""Core data types shared by every module."""

from __future__ import annotations

from dataclasses import dataclass

Qrels = dict[str, dict[str, int]]
"""Relevance judgments: query_id -> {doc_id: gain}."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: list[Document]
    queries: list[Query]
    qrels: Qrels


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
```

- [ ] **Step 4: Créer `tests/conftest.py`**

```python
import json
from pathlib import Path

import pytest

from rag_eval_lab.types import Dataset, Document, Query

TOPICS = [
    "volcan", "photosynthese", "fraction", "adverbe", "revolution",
    "atome", "cellule", "triangle", "poeme", "electricite",
    "gravite", "democratie", "molecule", "equation", "verbe",
    "climat", "pyramide", "magnetisme", "planete", "ecosysteme",
]


def _doc_text(topic: str) -> str:
    sentences = " ".join(f"Le {topic} est un sujet important numero {j}." for j in range(8))
    return f"Fiche {topic}. {sentences}"


@pytest.fixture
def mini_dataset() -> Dataset:
    documents = [Document(id=f"d{i}", text=_doc_text(t)) for i, t in enumerate(TOPICS)]
    queries = [Query(id=f"q{i}", text=f"Comment expliquer le {TOPICS[i]} ?") for i in range(5)]
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(5)}
    return Dataset(name="mini", documents=documents, queries=queries, qrels=qrels)


@pytest.fixture
def mini_jsonl(tmp_path: Path, mini_dataset: Dataset) -> Path:
    directory = tmp_path / "mini"
    directory.mkdir()
    with (directory / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for d in mini_dataset.documents:
            f.write(json.dumps({"id": d.id, "text": d.text}, ensure_ascii=False) + "\n")
    with (directory / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q in mini_dataset.queries:
            relevant = list(mini_dataset.qrels[q.id])
            f.write(json.dumps({"id": q.id, "text": q.text, "relevant": relevant}) + "\n")
    return directory
```

- [ ] **Step 5: Écrire le test de fumée `tests/test_types.py`**

```python
import rag_eval_lab
from rag_eval_lab.types import Chunk, Hit


def test_version():
    assert rag_eval_lab.__version__ == "0.1.0"


def test_hit_is_hashable_and_comparable():
    chunk = Chunk(id="d1#0", doc_id="d1", text="bonjour")
    assert Hit(chunk, 1.0) == Hit(chunk, 1.0)
    assert len({Hit(chunk, 1.0), Hit(chunk, 1.0)}) == 1


def test_mini_dataset_fixture(mini_dataset):
    assert len(mini_dataset.documents) == 20
    assert mini_dataset.qrels["q0"] == {"d0": 1}
```

- [ ] **Step 6: Installer et lancer les tests**

Run: `uv sync`
Expected: création de `.venv` et `uv.lock`, sans erreur.

Run: `uv run pytest -q`
Expected: `3 passed`

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: `All checks passed!` puis aucun fichier à reformater (sinon lancer `uv run ruff format .` et relancer).

- [ ] **Step 7: Créer `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.12"
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest -q
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore README.md .github src tests
git commit -m "chore: scaffold project with core types, fixtures and CI"
```

---

### Task 2: Métriques de retrieval

**Files:**
- Create: `src/rag_eval_lab/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Qrels` de `rag_eval_lab.types`
- Produces (`rag_eval_lab.metrics`) :
  - `Run = dict[str, list[str]]` (query_id → doc_ids classés, meilleur en premier)
  - `recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float`
  - `mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float`
  - `ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float`
  - `evaluate(run: Run, qrels: Qrels, ks: Sequence[int]) -> dict[str, float]` — clés `"recall@{k}"`, `"mrr@{k}"`, `"ndcg@{k}"` pour chaque k, moyennées sur les requêtes de `qrels` qui ont au moins un document pertinent (gain > 0) ; une requête absente de `run` compte comme une liste vide.

- [ ] **Step 1: Écrire les tests**

```python
import math

import pytest

from rag_eval_lab.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k

RANKED = ["a", "b", "c"]
RELEVANT = {"b": 1, "d": 1}


def test_recall_at_k():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 2) == 0.5
    assert recall_at_k(RANKED, RELEVANT, 3) == 0.5


def test_recall_ignores_zero_gain_and_empty_relevant():
    assert recall_at_k(RANKED, {"a": 0}, 3) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(RANKED, RELEVANT, 3) == 0.5
    assert mrr_at_k(RANKED, RELEVANT, 1) == 0.0
    assert mrr_at_k(RANKED, {"a": 1}, 3) == 1.0


def test_ndcg_at_k_hand_computed():
    # DCG = 1/log2(3); IDCG = 1 + 1/log2(3)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(expected)
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.386853, abs=1e-6)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["b", "d", "a"], RELEVANT, 3) == pytest.approx(1.0)


def test_evaluate_averages_over_queries():
    run = {"q1": ["a", "b"], "q2": ["x", "y"]}
    qrels = {"q1": {"a": 1}, "q2": {"y": 1}, "q3": {"z": 0}}
    scores = evaluate(run, qrels, ks=[1, 2])
    # q3 has no relevant doc -> ignored; q1 perfect, q2 hit at rank 2
    assert scores["recall@1"] == pytest.approx(0.5)
    assert scores["recall@2"] == pytest.approx(1.0)
    assert scores["mrr@2"] == pytest.approx((1.0 + 0.5) / 2)
    assert set(scores) == {"recall@1", "recall@2", "mrr@1", "mrr@2", "ndcg@1", "ndcg@2"}


def test_evaluate_missing_query_counts_as_empty():
    scores = evaluate({}, {"q1": {"a": 1}}, ks=[10])
    assert scores["recall@10"] == 0.0
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_metrics.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.metrics'`

- [ ] **Step 3: Implémenter `src/rag_eval_lab/metrics.py`**

```python
"""Information-retrieval metrics computed from ranked document ids."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rag_eval_lab.types import Qrels

Run = dict[str, list[str]]
"""Retrieval output: query_id -> ranked doc ids (best first)."""


def _relevant_ids(relevant: dict[str, int]) -> set[str]:
    return {doc_id for doc_id, gain in relevant.items() if gain > 0}


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    rel = _relevant_ids(relevant)
    if not rel:
        return 0.0
    return len(rel.intersection(ranked[:k])) / len(rel)


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(run: Run, qrels: Qrels, ks: Sequence[int]) -> dict[str, float]:
    """Average recall/MRR/nDCG at each k over queries with at least one relevant doc."""
    query_ids = [qid for qid, rel in qrels.items() if _relevant_ids(rel)]
    totals: dict[str, float] = {}
    for k in ks:
        for name in ("recall", "mrr", "ndcg"):
            totals[f"{name}@{k}"] = 0.0
    for qid in query_ids:
        ranked = run.get(qid, [])
        relevant = qrels[qid]
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
            totals[f"mrr@{k}"] += mrr_at_k(ranked, relevant, k)
            totals[f"ndcg@{k}"] += ndcg_at_k(ranked, relevant, k)
    n = len(query_ids)
    return {key: (value / n if n else 0.0) for key, value in totals.items()}
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_metrics.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/metrics.py tests/test_metrics.py
git commit -m "feat: add recall, MRR and nDCG metrics"
```

---

### Task 3: Chunking

**Files:**
- Create: `src/rag_eval_lab/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `Document`, `Chunk` de `rag_eval_lab.types`
- Produces (`rag_eval_lab.chunking`) :
  - `Chunker` (Protocol) : attribut `name: str`, méthode `chunk(doc: Document) -> list[Chunk]`
  - `NoChunker()` (`name == "none"`)
  - `FixedChunker(size: int, overlap: int)` (`name == f"fixed_{size}_{overlap}"`, `ValueError` si `size <= 0` ou `overlap` hors `[0, size)`)
  - `RecursiveChunker(size: int)` (`name == f"recursive_{size}"`)
  - `parse_chunker(spec: str) -> Chunker` : `"none"`, `"fixed_<size>_<overlap>"`, `"recursive_<size>"`, sinon `ValueError("Unknown chunker spec ...")`
  - Ids de chunks : `f"{doc.id}#{i}"` (i à partir de 0) ; les morceaux vides ou uniquement blancs sont ignorés.

- [ ] **Step 1: Écrire les tests**

```python
import pytest

from rag_eval_lab.chunking import (
    FixedChunker,
    NoChunker,
    RecursiveChunker,
    parse_chunker,
)
from rag_eval_lab.types import Chunk, Document


def test_no_chunker_returns_whole_document():
    chunks = NoChunker().chunk(Document("d1", "hello world"))
    assert chunks == [Chunk(id="d1#0", doc_id="d1", text="hello world")]


def test_fixed_chunker_sizes_and_overlap():
    chunks = FixedChunker(size=4, overlap=1).chunk(Document("d1", "abcdefghij"))
    assert [c.text for c in chunks] == ["abcd", "defg", "ghij"]
    assert [c.id for c in chunks] == ["d1#0", "d1#1", "d1#2"]


def test_fixed_chunker_short_document():
    chunks = FixedChunker(size=4, overlap=1).chunk(Document("d1", "abc"))
    assert [c.text for c in chunks] == ["abc"]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, -1)])
def test_fixed_chunker_rejects_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        FixedChunker(size=size, overlap=overlap)


def test_recursive_chunker_respects_size_and_keeps_text():
    text = (
        "Premier paragraphe court.\n\n"
        "Deuxième paragraphe un peu plus long que le premier. Il a deux phrases.\n\n"
        + "mot " * 60
    )
    chunks = RecursiveChunker(size=80).chunk(Document("d1", text))
    assert all(len(c.text) <= 80 for c in chunks)
    assert "".join(c.text for c in chunks) == text
    assert all(c.doc_id == "d1" for c in chunks)
    assert chunks[0].text == "Premier paragraphe court.\n\n"


def test_recursive_chunker_hard_splits_long_words():
    chunks = RecursiveChunker(size=10).chunk(Document("d1", "x" * 25))
    assert [len(c.text) for c in chunks] == [10, 10, 5]


def test_empty_document_gives_no_chunks():
    assert NoChunker().chunk(Document("d1", "   ")) == []


def test_parse_chunker():
    assert isinstance(parse_chunker("none"), NoChunker)
    fixed = parse_chunker("fixed_1000_200")
    assert isinstance(fixed, FixedChunker)
    assert (fixed.size, fixed.overlap, fixed.name) == (1000, 200, "fixed_1000_200")
    recursive = parse_chunker("recursive_800")
    assert isinstance(recursive, RecursiveChunker)
    assert recursive.name == "recursive_800"
    with pytest.raises(ValueError, match="Unknown chunker"):
        parse_chunker("banana")
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_chunking.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.chunking'`

- [ ] **Step 3: Implémenter `src/rag_eval_lab/chunking.py`**

```python
"""Document chunking strategies. Sizes are expressed in characters."""

from __future__ import annotations

import re
from typing import Protocol

from rag_eval_lab.types import Chunk, Document

_SEPARATORS = ("\n\n", "\n", ". ", " ")


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def _make_chunks(doc: Document, pieces: list[str]) -> list[Chunk]:
    kept = [p for p in pieces if p.strip()]
    return [Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=p) for i, p in enumerate(kept)]


class NoChunker:
    name = "none"

    def chunk(self, doc: Document) -> list[Chunk]:
        return _make_chunks(doc, [doc.text])


class FixedChunker:
    def __init__(self, size: int, overlap: int) -> None:
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError(f"Invalid fixed chunker: size={size}, overlap={overlap}")
        self.size = size
        self.overlap = overlap
        self.name = f"fixed_{size}_{overlap}"

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        step = self.size - self.overlap
        pieces: list[str] = []
        start = 0
        while True:
            pieces.append(text[start : start + self.size])
            if start + self.size >= len(text):
                break
            start += step
        return _make_chunks(doc, pieces)


def _split(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """Split text into pieces of at most `size` chars, keeping separators attached."""
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split(text, size, rest)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + sep if i < len(parts) - 1 else part
        pieces.extend(_split(piece, size, rest))
    return pieces


class RecursiveChunker:
    """Split on paragraphs, then lines, sentences and words; merge greedily up to `size`."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError(f"Invalid recursive chunker: size={size}")
        self.size = size
        self.name = f"recursive_{size}"

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces: list[str] = []
        current = ""
        for part in _split(doc.text, self.size, _SEPARATORS):
            if current and len(current) + len(part) > self.size:
                pieces.append(current)
                current = part
            else:
                current += part
        if current:
            pieces.append(current)
        return _make_chunks(doc, pieces)


_FIXED_RE = re.compile(r"^fixed_(\d+)_(\d+)$")
_RECURSIVE_RE = re.compile(r"^recursive_(\d+)$")


def parse_chunker(spec: str) -> Chunker:
    if spec == "none":
        return NoChunker()
    if m := _FIXED_RE.match(spec):
        return FixedChunker(size=int(m.group(1)), overlap=int(m.group(2)))
    if m := _RECURSIVE_RE.match(spec):
        return RecursiveChunker(size=int(m.group(1)))
    raise ValueError(
        f"Unknown chunker spec '{spec}'. Expected 'none', 'fixed_<size>_<overlap>' "
        "or 'recursive_<size>'."
    )
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_chunking.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/chunking.py tests/test_chunking.py
git commit -m "feat: add none, fixed and recursive chunkers"
```

---

### Task 4: Tokenisation, cache d'embeddings et encodeurs

**Files:**
- Create: `src/rag_eval_lab/text.py`, `src/rag_eval_lab/cache.py`, `src/rag_eval_lab/encoders.py`
- Test: `tests/test_cache_encoders.py`

**Interfaces:**
- Produces (`rag_eval_lab.text`) : `tokenize(text: str) -> list[str]` (minuscules, regex `\w+` Unicode).
- Produces (`rag_eval_lab.cache`) :
  - `make_key(*parts: str) -> str` (sha256 hexadécimal tronqué à 32 caractères, parties séparées par `\x00`)
  - `EmbeddingCache(directory: Path)` avec `get_or_compute(key: str, compute: Callable[[], np.ndarray]) -> np.ndarray` (fichier `<directory>/<key>.npy`, écriture atomique)
- Produces (`rag_eval_lab.encoders`) :
  - `Encoder` (Protocol) : `name: str`, `encode_queries(texts: list[str]) -> np.ndarray`, `encode_passages(texts: list[str]) -> np.ndarray` — lignes normalisées L2, `float32`, forme `(len(texts), dim)`.
  - `HashingEncoder(dim: int = 1024)` (`name == f"hashing-{dim}"`), sans modèle, déterministe (crc32).
  - `SentenceTransformerEncoder(name: str, model_id: str, query_prefix: str = "", passage_prefix: str = "", batch_size: int = 32, max_seq_length: int = 512, model: object | None = None)` ; attributs publics `name`, `model_id` ; modèle chargé paresseusement au premier encodage (`_model is None` avant).

- [ ] **Step 1: Écrire les tests**

```python
import numpy as np

from rag_eval_lab.cache import EmbeddingCache, make_key
from rag_eval_lab.encoders import HashingEncoder, SentenceTransformerEncoder
from rag_eval_lab.text import tokenize


def test_tokenize_lowercases_and_keeps_accents():
    assert tokenize("Le Volcan, l'ÉTÉ!") == ["le", "volcan", "l", "été"]


def test_make_key_is_deterministic_and_unambiguous():
    assert make_key("a", "b") == make_key("a", "b")
    assert make_key("ab", "") != make_key("a", "b")
    assert len(make_key("x")) == 32


def test_cache_computes_once(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    calls = []

    def compute():
        calls.append(1)
        return np.ones((2, 3), dtype=np.float32)

    first = cache.get_or_compute("k", compute)
    second = cache.get_or_compute("k", compute)
    assert len(calls) == 1
    np.testing.assert_array_equal(first, second)
    assert (tmp_path / "cache" / "k.npy").exists()


def test_hashing_encoder_is_normalized_and_deterministic():
    enc = HashingEncoder(dim=64)
    vectors = enc.encode_passages(["le volcan", "la photosynthese"])
    assert vectors.shape == (2, 64)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(vectors, enc.encode_passages(["le volcan", "la photosynthese"]))


def test_hashing_encoder_similarity_follows_word_overlap():
    enc = HashingEncoder(dim=1024)
    query = enc.encode_queries(["volcan actif"])[0]
    docs = enc.encode_passages(["un volcan actif", "la photosynthese des plantes"])
    assert docs[0] @ query > docs[1] @ query


def test_hashing_encoder_empty_text_is_zero_vector():
    vector = HashingEncoder(dim=8).encode_passages([""])[0]
    assert not np.isnan(vector).any()
    assert np.count_nonzero(vector) == 0


class FakeSentenceTransformer:
    def __init__(self):
        self.seen: list[str] = []

    def encode(self, texts, **kwargs):
        self.seen = list(texts)
        return np.ones((len(texts), 2), dtype=np.float32)


def test_sentence_transformer_encoder_is_lazy():
    enc = SentenceTransformerEncoder(name="e5-small", model_id="intfloat/multilingual-e5-small")
    assert enc._model is None
    assert enc.model_id == "intfloat/multilingual-e5-small"


def test_sentence_transformer_encoder_applies_prefixes():
    fake = FakeSentenceTransformer()
    enc = SentenceTransformerEncoder(
        name="e5", model_id="x", query_prefix="query: ", passage_prefix="passage: ", model=fake
    )
    enc.encode_queries(["a"])
    assert fake.seen == ["query: a"]
    out = enc.encode_passages(["b", "c"])
    assert fake.seen == ["passage: b", "passage: c"]
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_cache_encoders.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.cache'`

- [ ] **Step 3: Implémenter `src/rag_eval_lab/text.py`**

```python
"""Text normalization shared by lexical components."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
```

- [ ] **Step 4: Implémenter `src/rag_eval_lab/cache.py`**

```python
"""On-disk cache for embedding matrices."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np


def make_key(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


class EmbeddingCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def get_or_compute(self, key: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
        path = self.directory / f"{key}.npy"
        if path.exists():
            return np.load(path)
        array = compute()
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / f"{key}.tmp.npy"
        np.save(tmp, array)
        tmp.replace(path)
        return array
```

- [ ] **Step 5: Implémenter `src/rag_eval_lab/encoders.py`**

```python
"""Text encoders producing L2-normalized float32 embeddings."""

from __future__ import annotations

import zlib
from typing import Any, Protocol

import numpy as np

from rag_eval_lab.text import tokenize


class Encoder(Protocol):
    name: str

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...


class HashingEncoder:
    """Bag-of-words hashing trick. Model-free baseline, also used in tests."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return out

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class SentenceTransformerEncoder:
    """Wrapper around a sentence-transformers model, loaded lazily."""

    def __init__(
        self,
        name: str,
        model_id: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        max_seq_length: int = 512,
        model: Any | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device="cpu")
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        vectors = self._get_model().encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)
```

- [ ] **Step 6: Vérifier le succès**

Run: `uv run pytest tests/test_cache_encoders.py -q`
Expected: `8 passed`

- [ ] **Step 7: Commit**

```bash
git add src/rag_eval_lab/text.py src/rag_eval_lab/cache.py src/rag_eval_lab/encoders.py tests/test_cache_encoders.py
git commit -m "feat: add tokenizer, embedding cache and encoders"
```

---

### Task 5: Retrievers (BM25, dense, hybride RRF)

**Files:**
- Create: `src/rag_eval_lab/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Chunk`, `Hit` (types) ; `tokenize` (text) ; `Encoder` (encoders) ; `EmbeddingCache`, `make_key` (cache)
- Produces (`rag_eval_lab.retrieval`) :
  - `Retriever` (Protocol) : `index(chunks: list[Chunk]) -> None`, `search(query: str, k: int) -> list[Hit]` (score décroissant, au plus `k` hits)
  - `BM25Retriever()`
  - `DenseRetriever(encoder: Encoder, cache: EmbeddingCache | None = None)` ; attribut public `encoder` ; clé de cache = `make_key(encoder.name, *chunk_ids, *chunk_texts)`
  - `HybridRetriever(retrievers: list[Retriever], rrf_k: int = 60, depth: int = 100)` ; attribut public `retrievers` ; `ValueError` si moins de 2 retrievers ; score RRF = Σ 1/(rrf_k + rang), rang commençant à 1.

- [ ] **Step 1: Écrire les tests**

```python
import numpy as np
import pytest

from rag_eval_lab.cache import EmbeddingCache
from rag_eval_lab.encoders import HashingEncoder
from rag_eval_lab.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from rag_eval_lab.types import Chunk, Hit

CHUNKS = [
    Chunk("a#0", "a", "le volcan est actif"),
    Chunk("b#0", "b", "la photosynthese des plantes vertes"),
    Chunk("c#0", "c", "une fraction et un denominateur"),
]


def test_bm25_ranks_matching_chunk_first():
    retriever = BM25Retriever()
    retriever.index(CHUNKS)
    hits = retriever.search("volcan", k=2)
    assert len(hits) == 2
    assert hits[0].chunk.id == "a#0"
    assert hits[0].score >= hits[1].score


def test_search_k_larger_than_corpus():
    retriever = BM25Retriever()
    retriever.index(CHUNKS)
    assert len(retriever.search("volcan", k=50)) == 3


def test_dense_retriever_ranks_and_caches(tmp_path):
    cache = EmbeddingCache(tmp_path)
    for _ in range(2):
        retriever = DenseRetriever(HashingEncoder(dim=256), cache=cache)
        retriever.index(CHUNKS)
        hits = retriever.search("une fraction", k=3)
        assert hits[0].chunk.id == "c#0"
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(list(tmp_path.glob("*.npy"))) == 1


class FixedRetriever:
    def __init__(self, ranked_ids: list[str]) -> None:
        self.ranked_ids = ranked_ids

    def index(self, chunks):
        self.by_id = {c.id: c for c in chunks}

    def search(self, query, k):
        return [Hit(self.by_id[i], 1.0) for i in self.ranked_ids[:k]]


def test_hybrid_rrf_fusion():
    hybrid = HybridRetriever(
        [FixedRetriever(["a#0", "b#0", "c#0"]), FixedRetriever(["b#0", "c#0", "a#0"])], rrf_k=60
    )
    hybrid.index(CHUNKS)
    hits = hybrid.search("anything", k=3)
    assert [h.chunk.id for h in hits] == ["b#0", "a#0", "c#0"]
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_hybrid_requires_two_retrievers():
    with pytest.raises(ValueError):
        HybridRetriever([BM25Retriever()])


def test_hybrid_with_real_retrievers():
    hybrid = HybridRetriever([BM25Retriever(), DenseRetriever(HashingEncoder(dim=256))])
    hybrid.index(CHUNKS)
    assert hybrid.search("photosynthese", k=1)[0].chunk.id == "b#0"
    assert isinstance(hybrid.search("photosynthese", k=1)[0].score, float)
    assert np.isfinite(hybrid.search("photosynthese", k=1)[0].score)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_retrieval.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.retrieval'`

- [ ] **Step 3: Implémenter `src/rag_eval_lab/retrieval.py`**

```python
"""First-stage retrievers over chunks: lexical, dense and hybrid (RRF)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from rag_eval_lab.cache import EmbeddingCache, make_key
from rag_eval_lab.encoders import Encoder
from rag_eval_lab.text import tokenize
from rag_eval_lab.types import Chunk, Hit


class Retriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[Hit]: ...


def _top_k(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [Hit(chunks[i], float(scores[i])) for i in order]


class BM25Retriever:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.search called before index")
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        return _top_k(self._chunks, scores, k)


class DenseRetriever:
    def __init__(self, encoder: Encoder, cache: EmbeddingCache | None = None) -> None:
        self.encoder = encoder
        self.cache = cache
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]

        def compute() -> np.ndarray:
            return self.encoder.encode_passages(texts)

        if self.cache is None:
            self._embeddings = compute()
        else:
            key = make_key(self.encoder.name, *(c.id for c in self._chunks), *texts)
            self._embeddings = self.cache.get_or_compute(key, compute)

    def search(self, query: str, k: int) -> list[Hit]:
        if self._embeddings is None:
            raise RuntimeError("DenseRetriever.search called before index")
        query_vector = self.encoder.encode_queries([query])[0]
        scores = self._embeddings @ query_vector
        return _top_k(self._chunks, scores, k)


class HybridRetriever:
    """Reciprocal Rank Fusion of several retrievers."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 100) -> None:
        if len(retrievers) < 2:
            raise ValueError("HybridRetriever needs at least two retrievers")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth

    def index(self, chunks: list[Chunk]) -> None:
        for retriever in self.retrievers:
            retriever.index(chunks)

    def search(self, query: str, k: int) -> list[Hit]:
        depth = max(k, self.depth)
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, depth), start=1):
                chunks[hit.chunk.id] = hit.chunk
                scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
        return [Hit(chunks[cid], scores[cid]) for cid in ranked]
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_retrieval.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/retrieval.py tests/test_retrieval.py
git commit -m "feat: add BM25, dense and hybrid RRF retrievers"
```

---

### Task 6: Reranker et pipeline

**Files:**
- Create: `src/rag_eval_lab/reranking.py`, `src/rag_eval_lab/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Chunk`, `Hit`, `Document` (types) ; `Chunker` (chunking) ; `Retriever` (retrieval)
- Produces (`rag_eval_lab.reranking`) :
  - `Reranker` (Protocol) : `load() -> None`, `rerank(query: str, hits: list[Hit], k: int) -> list[Hit]`
  - `CrossEncoderReranker(model_id: str, batch_size: int = 32, max_length: int = 512, model: object | None = None)` ; attribut public `model_id` ; le modèle doit exposer `predict(pairs, batch_size=..., show_progress_bar=...)`.
- Produces (`rag_eval_lab.pipeline`) :
  - `aggregate_to_docs(hits: list[Hit], k: int) -> list[str]` : score du doc = max de ses chunks, dédoublonné, trié décroissant (égalités : ordre de première apparition).
  - `RetrievalPipeline(chunker: Chunker, retriever: Retriever, reranker: Reranker | None = None, candidates: int = 100)` ; `index(documents: list[Document]) -> None` (définit `n_chunks: int`, appelle `reranker.load()`) ; `search(query: str, k: int) -> list[str]` (doc ids).

- [ ] **Step 1: Écrire les tests**

```python
import numpy as np

from rag_eval_lab.chunking import FixedChunker
from rag_eval_lab.pipeline import RetrievalPipeline, aggregate_to_docs
from rag_eval_lab.reranking import CrossEncoderReranker
from rag_eval_lab.retrieval import BM25Retriever
from rag_eval_lab.types import Chunk, Hit


class LengthModel:
    def predict(self, pairs, **kwargs):
        return np.array([len(text) for _, text in pairs], dtype=float)


class KeywordModel:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    def predict(self, pairs, **kwargs):
        return np.array([1.0 if self.keyword in text else 0.0 for _, text in pairs])


def test_cross_encoder_reranker_reorders():
    hits = [
        Hit(Chunk("a#0", "a", "court"), 0.9),
        Hit(Chunk("b#0", "b", "beaucoup plus long"), 0.1),
    ]
    reranker = CrossEncoderReranker("fake", model=LengthModel())
    out = reranker.rerank("q", hits, k=2)
    assert [h.chunk.id for h in out] == ["b#0", "a#0"]
    assert out[0].score == len("beaucoup plus long")


def test_cross_encoder_reranker_empty_and_truncation():
    reranker = CrossEncoderReranker("fake", model=LengthModel())
    assert reranker.rerank("q", [], k=5) == []
    hits = [Hit(Chunk(f"x#{i}", "x", "t" * i), 0.0) for i in range(1, 5)]
    assert len(reranker.rerank("q", hits, k=2)) == 2


def test_cross_encoder_reranker_is_lazy():
    assert CrossEncoderReranker("some/model")._model is None


def test_aggregate_to_docs_uses_max_score_and_dedups():
    hits = [
        Hit(Chunk("a#1", "a", "x"), 0.9),
        Hit(Chunk("b#0", "b", "y"), 0.8),
        Hit(Chunk("a#0", "a", "z"), 0.7),
        Hit(Chunk("c#0", "c", "w"), 0.95),
    ]
    assert aggregate_to_docs(hits, k=2) == ["c", "a"]
    assert aggregate_to_docs(hits, k=10) == ["c", "a", "b"]


def test_pipeline_bm25_on_mini_dataset(mini_dataset):
    pipeline = RetrievalPipeline(FixedChunker(size=200, overlap=50), BM25Retriever())
    pipeline.index(mini_dataset.documents)
    assert pipeline.n_chunks > len(mini_dataset.documents)
    assert pipeline.search("Comment expliquer le volcan ?", k=3)[0] == "d0"


def test_pipeline_with_reranker_overrides_first_stage(mini_dataset):
    pipeline = RetrievalPipeline(
        FixedChunker(size=200, overlap=50),
        BM25Retriever(),
        reranker=CrossEncoderReranker("fake", model=KeywordModel("atome")),
        candidates=1000,
    )
    pipeline.index(mini_dataset.documents)
    assert pipeline.search("Comment expliquer le volcan ?", k=3)[0] == "d5"
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.pipeline'`

- [ ] **Step 3: Implémenter `src/rag_eval_lab/reranking.py`**

```python
"""Second-stage rerankers."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from rag_eval_lab.types import Hit


class Reranker(Protocol):
    def load(self) -> None: ...

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


class CrossEncoderReranker:
    """Cross-encoder scoring (query, chunk) pairs; model loaded lazily."""

    def __init__(
        self,
        model_id: str,
        batch_size: int = 32,
        max_length: int = 512,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = model

    def load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, max_length=self.max_length, device="cpu")

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        self.load()
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = np.asarray(
            self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False),
            dtype=np.float64,
        )
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(hits[i].chunk, float(scores[i])) for i in order]
```

- [ ] **Step 4: Implémenter `src/rag_eval_lab/pipeline.py`**

```python
"""Assemble chunker, retriever and optional reranker into a document ranker."""

from __future__ import annotations

from rag_eval_lab.chunking import Chunker
from rag_eval_lab.reranking import Reranker
from rag_eval_lab.retrieval import Retriever
from rag_eval_lab.types import Document, Hit


def aggregate_to_docs(hits: list[Hit], k: int) -> list[str]:
    """Map chunk hits to doc ids: a document scores as its best chunk."""
    best: dict[str, float] = {}
    for hit in hits:
        doc_id = hit.chunk.doc_id
        if doc_id not in best or hit.score > best[doc_id]:
            best[doc_id] = hit.score
    return sorted(best, key=lambda d: best[d], reverse=True)[:k]


class RetrievalPipeline:
    def __init__(
        self,
        chunker: Chunker,
        retriever: Retriever,
        reranker: Reranker | None = None,
        candidates: int = 100,
    ) -> None:
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker
        self.candidates = candidates
        self.n_chunks = 0

    def index(self, documents: list[Document]) -> None:
        chunks = [chunk for doc in documents for chunk in self.chunker.chunk(doc)]
        self.retriever.index(chunks)
        self.n_chunks = len(chunks)
        if self.reranker is not None:
            self.reranker.load()

    def search(self, query: str, k: int) -> list[str]:
        hits = self.retriever.search(query, max(self.candidates, k))
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits, len(hits))
        return aggregate_to_docs(hits, k)
```

- [ ] **Step 5: Vérifier le succès**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add src/rag_eval_lab/reranking.py src/rag_eval_lab/pipeline.py tests/test_pipeline.py
git commit -m "feat: add cross-encoder reranker and retrieval pipeline"
```

---

### Task 7: Configuration pydantic et factory de composants

**Files:**
- Create: `src/rag_eval_lab/factory.py`, `src/rag_eval_lab/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `parse_chunker` (chunking) ; `HashingEncoder`, `SentenceTransformerEncoder`, `Encoder` (encoders) ; `BM25Retriever`, `DenseRetriever`, `HybridRetriever`, `Retriever` (retrieval) ; `CrossEncoderReranker`, `Reranker` (reranking) ; `EmbeddingCache`, `make_key` (cache)
- Produces (`rag_eval_lab.factory`) :
  - `ENCODERS: dict[str, dict[str, str]]` : alias → kwargs (`model_id`, préfixes). Alias : `e5-small`, `e5-base`, `bge-m3`.
  - `RERANKERS: dict[str, str]` : `minilm` → `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `bge-v2-m3` → `BAAI/bge-reranker-v2-m3`.
  - `build_encoder(name: str) -> Encoder` (`"hashing"` ou un alias ; sinon `ValueError("Unknown encoder '...'")`)
  - `build_retriever(spec: str, cache: EmbeddingCache | None = None) -> Retriever` : `"bm25"`, `"dense:<encoder>"`, `"hybrid:<a>+<b>[+...]"` où chaque partie est `bm25` ou un nom d'encodeur ; sinon `ValueError("Unknown retriever spec ...")`
  - `build_reranker(spec: str) -> Reranker | None` : `"none"` → `None` ; alias de `RERANKERS` ; sinon `ValueError("Unknown reranker '...'")`
  - Aucun modèle n'est chargé par ces fonctions (tout est paresseux).
- Produces (`rag_eval_lab.config`) :
  - `DatasetConfig(name: Literal["alloprof", "jsonl"], path: str | None = None, n_queries: int | None = None, n_docs: int | None = None, seed: int = 42, split: str = "test")` ; `path` obligatoire si `name == "jsonl"`.
  - `GridConfig(chunker: list[str], retriever: list[str], reranker: list[str] = ["none"])`
  - `RunSpec(chunker: str, retriever: str, reranker: str = "none")` (pydantic, frozen)
  - `ExperimentConfig(dataset, grid, k: int = 10, ks: list[int] = [5, 10], candidates: int = 100, cache_dir: str = ".cache/embeddings", exclude: list[dict[Literal["chunker","retriever","reranker"], str]] = [])` avec `expand() -> list[RunSpec]` et `run_id(spec: RunSpec) -> str` (12 caractères hexadécimaux, dépend du dataset, de `k`, `candidates` et du spec).
  - `load_config(path: Path) -> ExperimentConfig`
  - Toutes les classes rejettent les champs inconnus (`extra="forbid"`).

- [ ] **Step 1: Écrire les tests**

```python
import pytest
import yaml
from pydantic import ValidationError

from rag_eval_lab.config import ExperimentConfig, RunSpec, load_config
from rag_eval_lab.encoders import SentenceTransformerEncoder
from rag_eval_lab.factory import build_reranker, build_retriever
from rag_eval_lab.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

BASE = {
    "dataset": {"name": "jsonl", "path": "data/mini"},
    "grid": {
        "chunker": ["none", "fixed_100_20"],
        "retriever": ["bm25", "dense:hashing"],
        "reranker": ["none", "minilm"],
    },
    "exclude": [{"chunker": "none", "reranker": "minilm"}],
}


def _config(**overrides) -> ExperimentConfig:
    data = {**BASE, **overrides}
    return ExperimentConfig.model_validate(data)


def test_expand_grid_applies_exclude():
    specs = _config().expand()
    assert len(specs) == 6
    assert RunSpec(chunker="none", retriever="bm25", reranker="minilm") not in specs
    assert RunSpec(chunker="fixed_100_20", retriever="bm25", reranker="minilm") in specs


def test_unknown_encoder_rejected():
    grid = {**BASE["grid"], "retriever": ["dense:unknown"]}
    with pytest.raises(ValidationError, match="Unknown encoder"):
        _config(grid=grid)


def test_unknown_chunker_rejected():
    grid = {**BASE["grid"], "chunker": ["banana"]}
    with pytest.raises(ValidationError, match="Unknown chunker"):
        _config(grid=grid)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        _config(typo_field=1)


def test_jsonl_dataset_requires_path():
    with pytest.raises(ValidationError, match="path"):
        _config(dataset={"name": "jsonl"})


def test_ks_must_not_exceed_k():
    with pytest.raises(ValidationError, match="ks"):
        _config(k=5, ks=[5, 10])


def test_run_id_is_stable_and_depends_on_dataset():
    spec = RunSpec(chunker="none", retriever="bm25")
    first = _config().run_id(spec)
    assert first == _config().run_id(spec)
    assert len(first) == 12
    other = _config(dataset={"name": "jsonl", "path": "data/mini", "seed": 7}).run_id(spec)
    assert other != first


def test_load_config_from_yaml(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    assert load_config(path).grid.retriever == ["bm25", "dense:hashing"]


def test_build_retriever_variants():
    assert isinstance(build_retriever("bm25"), BM25Retriever)
    dense = build_retriever("dense:e5-small")
    assert isinstance(dense, DenseRetriever)
    assert isinstance(dense.encoder, SentenceTransformerEncoder)
    assert dense.encoder.model_id == "intfloat/multilingual-e5-small"
    assert dense.encoder._model is None
    hybrid = build_retriever("hybrid:bm25+hashing")
    assert isinstance(hybrid, HybridRetriever)
    assert len(hybrid.retrievers) == 2
    with pytest.raises(ValueError, match="Unknown retriever"):
        build_retriever("sparse:splade")
    with pytest.raises(ValueError):
        build_retriever("hybrid:bm25")


def test_build_reranker():
    assert build_reranker("none") is None
    reranker = build_reranker("minilm")
    assert reranker.model_id == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    with pytest.raises(ValueError, match="Unknown reranker"):
        build_reranker("nope")
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.config'`

- [ ] **Step 3: Implémenter `src/rag_eval_lab/factory.py`**

```python
"""Build pipeline components from compact spec strings used in configs."""

from __future__ import annotations

from rag_eval_lab.cache import EmbeddingCache
from rag_eval_lab.encoders import Encoder, HashingEncoder, SentenceTransformerEncoder
from rag_eval_lab.reranking import CrossEncoderReranker, Reranker
from rag_eval_lab.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever

ENCODERS: dict[str, dict[str, str]] = {
    "e5-small": {
        "model_id": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "e5-base": {
        "model_id": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "bge-m3": {"model_id": "BAAI/bge-m3"},
}

RERANKERS: dict[str, str] = {
    "minilm": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
}


def build_encoder(name: str) -> Encoder:
    if name == "hashing":
        return HashingEncoder()
    if name in ENCODERS:
        return SentenceTransformerEncoder(name=name, **ENCODERS[name])
    known = ", ".join(["hashing", *ENCODERS])
    raise ValueError(f"Unknown encoder '{name}'. Known encoders: {known}")


def build_retriever(spec: str, cache: EmbeddingCache | None = None) -> Retriever:
    kind, _, arg = spec.partition(":")
    if spec == "bm25":
        return BM25Retriever()
    if kind == "dense" and arg:
        return DenseRetriever(build_encoder(arg), cache=cache)
    if kind == "hybrid" and arg:
        parts = arg.split("+")
        if len(parts) < 2:
            raise ValueError(f"Hybrid retriever '{spec}' needs at least two parts")
        return HybridRetriever(
            [build_retriever(p if p == "bm25" else f"dense:{p}", cache) for p in parts]
        )
    raise ValueError(
        f"Unknown retriever spec '{spec}'. Expected 'bm25', 'dense:<encoder>' "
        "or 'hybrid:<a>+<b>'."
    )


def build_reranker(spec: str) -> Reranker | None:
    if spec == "none":
        return None
    if spec in RERANKERS:
        return CrossEncoderReranker(RERANKERS[spec])
    known = ", ".join(["none", *RERANKERS])
    raise ValueError(f"Unknown reranker '{spec}'. Known rerankers: {known}")
```

- [ ] **Step 4: Implémenter `src/rag_eval_lab/config.py`**

```python
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
```

- [ ] **Step 5: Vérifier le succès**

Run: `uv run pytest tests/test_config.py -q`
Expected: `10 passed`

- [ ] **Step 6: Commit**

```bash
git add src/rag_eval_lab/factory.py src/rag_eval_lab/config.py tests/test_config.py
git commit -m "feat: add validated experiment config and component factory"
```

---

### Task 8: Chargement et échantillonnage des datasets

**Files:**
- Create: `src/rag_eval_lab/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `Dataset`, `Document`, `Query` (types) ; `DatasetConfig` (config)
- Produces (`rag_eval_lab.data`) :
  - `load_jsonl(directory: Path) -> Dataset` : `corpus.jsonl` (`{"id", "text"}`), `queries.jsonl` (`{"id", "text", "relevant": [doc_id, ...]}`) ; `name` = nom du dossier.
  - `alloprof_from_rows(doc_rows: Iterable[dict], query_rows: Iterable[dict]) -> Dataset` : texte = `f"{title}\n\n{text}"`, doc id = `uuid`, query id = `str(id)`, gains à 1, requêtes aux ids dupliqués ignorées.
  - `load_alloprof(split: str = "test") -> Dataset` (télécharge via `datasets`, import paresseux)
  - `sample_dataset(dataset: Dataset, n_queries: int | None, n_docs: int | None, seed: int) -> Dataset` : retire les jugements vers des docs absents et les requêtes sans doc pertinent ; échantillonne les requêtes ; si `n_docs`, garde tous les docs pertinents + docs aléatoires (ordre d'origine conservé) ; `ValueError` si `n_docs` < nombre de docs pertinents requis.
  - `load_benchmark(cfg: DatasetConfig) -> Dataset` : charge puis échantillonne.

- [ ] **Step 1: Écrire les tests**

```python
import pytest

from rag_eval_lab.config import DatasetConfig
from rag_eval_lab.data import alloprof_from_rows, load_benchmark, load_jsonl, sample_dataset
from rag_eval_lab.types import Dataset, Document, Query


def test_load_jsonl(mini_jsonl):
    dataset = load_jsonl(mini_jsonl)
    assert dataset.name == "mini"
    assert len(dataset.documents) == 20
    assert len(dataset.queries) == 5
    assert dataset.qrels["q0"] == {"d0": 1}


def test_alloprof_from_rows():
    docs = [{"uuid": "u1", "title": "Le volcan", "text": "Un volcan est...", "topic": "x"}]
    queries = [
        {"id": 7, "text": "c'est quoi un volcan", "relevant": ["u1"], "subject": "s"},
        {"id": 7, "text": "doublon", "relevant": ["u1"]},
    ]
    dataset = alloprof_from_rows(docs, queries)
    assert dataset.documents == [Document("u1", "Le volcan\n\nUn volcan est...")]
    assert dataset.queries == [Query("7", "c'est quoi un volcan")]
    assert dataset.qrels == {"7": {"u1": 1}}


def test_sample_queries_is_deterministic(mini_dataset):
    a = sample_dataset(mini_dataset, n_queries=3, n_docs=None, seed=1)
    b = sample_dataset(mini_dataset, n_queries=3, n_docs=None, seed=1)
    assert [q.id for q in a.queries] == [q.id for q in b.queries]
    assert len(a.queries) == 3
    assert set(a.qrels) == {q.id for q in a.queries}
    assert len(a.documents) == 20


def test_sample_docs_keeps_relevant_documents(mini_dataset):
    sampled = sample_dataset(mini_dataset, n_queries=2, n_docs=6, seed=0)
    doc_ids = {d.id for d in sampled.documents}
    assert len(doc_ids) == 6
    for rel in sampled.qrels.values():
        assert set(rel) <= doc_ids


def test_sample_drops_queries_without_relevant_documents():
    dataset = Dataset(
        name="t",
        documents=[Document("d1", "texte")],
        queries=[Query("q1", "a"), Query("q2", "b")],
        qrels={"q1": {"d1": 1}, "q2": {"missing": 1}},
    )
    sampled = sample_dataset(dataset, n_queries=None, n_docs=None, seed=0)
    assert [q.id for q in sampled.queries] == ["q1"]
    assert sampled.qrels == {"q1": {"d1": 1}}


def test_sample_rejects_too_small_n_docs(mini_dataset):
    with pytest.raises(ValueError, match="n_docs"):
        sample_dataset(mini_dataset, n_queries=5, n_docs=2, seed=0)


def test_load_benchmark_jsonl(mini_jsonl):
    cfg = DatasetConfig(name="jsonl", path=str(mini_jsonl), n_queries=2, seed=3)
    dataset = load_benchmark(cfg)
    assert len(dataset.queries) == 2
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_data.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.data'`

- [ ] **Step 3: Implémenter `src/rag_eval_lab/data.py`**

```python
"""Dataset loaders and deterministic sub-sampling."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

from rag_eval_lab.config import DatasetConfig
from rag_eval_lab.types import Dataset, Document, Qrels, Query

ALLOPROF_REPO = "lyon-nlp/alloprof"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl(directory: Path) -> Dataset:
    directory = Path(directory)
    documents = [Document(str(r["id"]), r["text"]) for r in _read_jsonl(directory / "corpus.jsonl")]
    queries: list[Query] = []
    qrels: Qrels = {}
    for row in _read_jsonl(directory / "queries.jsonl"):
        qid = str(row["id"])
        queries.append(Query(qid, row["text"]))
        qrels[qid] = {str(doc_id): 1 for doc_id in row["relevant"]}
    return Dataset(name=directory.name, documents=documents, queries=queries, qrels=qrels)


def alloprof_from_rows(doc_rows: Iterable[dict], query_rows: Iterable[dict]) -> Dataset:
    documents = [Document(r["uuid"], f"{r['title']}\n\n{r['text']}") for r in doc_rows]
    queries: list[Query] = []
    qrels: Qrels = {}
    for row in query_rows:
        qid = str(row["id"])
        if qid in qrels:
            continue
        queries.append(Query(qid, row["text"]))
        qrels[qid] = {doc_id: 1 for doc_id in row["relevant"]}
    return Dataset(name="alloprof", documents=documents, queries=queries, qrels=qrels)


def load_alloprof(split: str = "test") -> Dataset:
    from datasets import load_dataset

    doc_rows = load_dataset(ALLOPROF_REPO, "documents", split="test")
    query_rows = load_dataset(ALLOPROF_REPO, "queries", split=split)
    return alloprof_from_rows(doc_rows, query_rows)


def sample_dataset(
    dataset: Dataset, n_queries: int | None, n_docs: int | None, seed: int
) -> Dataset:
    rng = random.Random(seed)
    doc_ids = {d.id for d in dataset.documents}
    qrels_all = {
        qid: {d: g for d, g in rel.items() if d in doc_ids and g > 0}
        for qid, rel in dataset.qrels.items()
    }
    queries = [q for q in dataset.queries if qrels_all.get(q.id)]
    if n_queries is not None and n_queries < len(queries):
        queries = rng.sample(queries, n_queries)
    qrels = {q.id: qrels_all[q.id] for q in queries}

    documents = dataset.documents
    if n_docs is not None and n_docs < len(documents):
        needed = {d for rel in qrels.values() for d in rel}
        if len(needed) > n_docs:
            raise ValueError(
                f"n_docs={n_docs} is smaller than the {len(needed)} relevant documents required"
            )
        others = [d.id for d in documents if d.id not in needed]
        keep = needed | set(rng.sample(others, n_docs - len(needed)))
        documents = [d for d in documents if d.id in keep]
    return Dataset(name=dataset.name, documents=documents, queries=queries, qrels=qrels)


def load_benchmark(cfg: DatasetConfig) -> Dataset:
    if cfg.name == "alloprof":
        dataset = load_alloprof(cfg.split)
    else:
        dataset = load_jsonl(Path(cfg.path))
    return sample_dataset(dataset, cfg.n_queries, cfg.n_docs, cfg.seed)
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_data.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/data.py tests/test_data.py
git commit -m "feat: add jsonl and Alloprof loaders with deterministic sampling"
```

---

### Task 9: Runner d'expériences (reprise, erreurs, chronométrage)

**Files:**
- Create: `src/rag_eval_lab/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `RunSpec` (config) ; `load_benchmark` (data) ; `parse_chunker` (chunking) ; `build_retriever`, `build_reranker` (factory) ; `RetrievalPipeline` (pipeline) ; `evaluate` (metrics) ; `EmbeddingCache` (cache) ; `Dataset` (types)
- Produces (`rag_eval_lab.experiment`) :
  - `run_experiment(config: ExperimentConfig, out_dir: Path, dataset: Dataset | None = None) -> list[dict]` : renvoie les enregistrements produits pendant cet appel (les configurations déjà `ok` sont sautées).
  - `environment_info() -> dict` : `python`, `platform`, `processor`, `git_commit`, `packages`.
  - Format d'un enregistrement (une ligne de `results.jsonl`) :
    - toujours : `config_id`, `status` (`"ok"` | `"error"`), `chunker`, `retriever`, `reranker`, `timestamp` (ISO 8601 UTC)
    - si `ok` : `metrics` (dict de `evaluate`), `latency_ms` (`{"p50": float, "p95": float}`), `index_time_s`, `n_chunks`, `n_docs`, `n_queries`, `dataset` (nom), `env`
    - si `error` : `error` (`"<ExceptionType>: <message>"`)
  - Classements bruts : `<out_dir>/runs/<config_id>.json` (dict query_id → liste de doc ids).

- [ ] **Step 1: Écrire les tests**

```python
import json

from rag_eval_lab import experiment
from rag_eval_lab.config import ExperimentConfig
from rag_eval_lab.experiment import environment_info, run_experiment


def _config(mini_jsonl, tmp_path, retrievers=("bm25", "dense:hashing")) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "dataset": {"name": "jsonl", "path": str(mini_jsonl)},
            "k": 5,
            "ks": [1, 5],
            "candidates": 20,
            "cache_dir": str(tmp_path / "cache"),
            "grid": {"chunker": ["none"], "retriever": list(retrievers)},
        }
    )


def _lines(out_dir):
    return (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()


def test_run_experiment_writes_results(tmp_path, mini_jsonl):
    out = tmp_path / "out"
    records = run_experiment(_config(mini_jsonl, tmp_path), out)
    assert [r["status"] for r in records] == ["ok", "ok"]
    assert len(_lines(out)) == 2
    record = records[0]
    assert record["metrics"]["recall@5"] >= 0.8
    assert set(record["latency_ms"]) == {"p50", "p95"}
    assert record["n_queries"] == 5
    assert record["n_chunks"] == 20
    run = json.loads((out / "runs" / f"{record['config_id']}.json").read_text(encoding="utf-8"))
    assert set(run) == {"q0", "q1", "q2", "q3", "q4"}


def test_run_experiment_resumes(tmp_path, mini_jsonl):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_experiment(cfg, out)
    assert run_experiment(cfg, out) == []
    assert len(_lines(out)) == 2


def test_run_experiment_records_errors_and_retries(tmp_path, mini_jsonl, monkeypatch):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    real_build = experiment.build_retriever

    def flaky(spec, cache=None):
        if spec == "bm25":
            raise RuntimeError("boom")
        return real_build(spec, cache)

    monkeypatch.setattr(experiment, "build_retriever", flaky)
    records = run_experiment(cfg, out)
    statuses = {r["retriever"]: r["status"] for r in records}
    assert statuses == {"bm25": "error", "dense:hashing": "ok"}
    error = next(r for r in records if r["status"] == "error")
    assert error["error"] == "RuntimeError: boom"

    monkeypatch.undo()
    retried = run_experiment(cfg, out)
    assert [(r["retriever"], r["status"]) for r in retried] == [("bm25", "ok")]


def test_environment_info_keys():
    info = environment_info()
    assert {"python", "platform", "processor", "git_commit", "packages"} <= set(info)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_experiment.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.experiment'`

- [ ] **Step 3: Implémenter `src/rag_eval_lab/experiment.py`**

```python
"""Run a grid of retrieval configurations and record results as JSON lines."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from rag_eval_lab.cache import EmbeddingCache
from rag_eval_lab.chunking import parse_chunker
from rag_eval_lab.config import ExperimentConfig, RunSpec
from rag_eval_lab.data import load_benchmark
from rag_eval_lab.factory import build_reranker, build_retriever
from rag_eval_lab.metrics import evaluate
from rag_eval_lab.pipeline import RetrievalPipeline
from rag_eval_lab.types import Dataset

logger = logging.getLogger(__name__)

_TRACKED_PACKAGES = ("rag-eval-lab", "sentence-transformers", "torch", "rank-bm25", "numpy")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": _git_commit(),
        "packages": {name: _package_version(name) for name in _TRACKED_PACKAGES},
    }


def _completed_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") == "ok":
                done.add(record["config_id"])
    return done


def _append(results_path: Path, record: dict) -> None:
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_one(
    config: ExperimentConfig,
    spec: RunSpec,
    dataset: Dataset,
    cache: EmbeddingCache,
    out_dir: Path,
    config_id: str,
) -> dict:
    pipeline = RetrievalPipeline(
        chunker=parse_chunker(spec.chunker),
        retriever=build_retriever(spec.retriever, cache),
        reranker=build_reranker(spec.reranker),
        candidates=config.candidates,
    )
    start = time.perf_counter()
    pipeline.index(dataset.documents)
    index_time_s = time.perf_counter() - start

    run: dict[str, list[str]] = {}
    latencies_ms: list[float] = []
    for i, query in enumerate(dataset.queries, start=1):
        start = time.perf_counter()
        run[query.id] = pipeline.search(query.text, config.k)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        if i % 100 == 0:
            logger.info("  %d/%d queries", i, len(dataset.queries))

    (out_dir / "runs" / f"{config_id}.json").write_text(json.dumps(run), encoding="utf-8")
    return {
        "metrics": evaluate(run, dataset.qrels, config.ks),
        "latency_ms": {
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
        },
        "index_time_s": index_time_s,
        "n_chunks": pipeline.n_chunks,
        "n_docs": len(dataset.documents),
        "n_queries": len(dataset.queries),
        "dataset": dataset.name,
        "env": environment_info(),
    }


def run_experiment(
    config: ExperimentConfig, out_dir: Path, dataset: Dataset | None = None
) -> list[dict]:
    out_dir = Path(out_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = _completed_ids(results_path)
    specs = config.expand()
    if dataset is None:
        dataset = load_benchmark(config.dataset)
    cache = EmbeddingCache(Path(config.cache_dir))

    records = []
    for n, spec in enumerate(specs, start=1):
        config_id = config.run_id(spec)
        label = f"{spec.chunker} | {spec.retriever} | {spec.reranker}"
        if config_id in done:
            logger.info("[%d/%d] skip (already done): %s", n, len(specs), label)
            continue
        logger.info("[%d/%d] run: %s", n, len(specs), label)
        record = {"config_id": config_id, **spec.model_dump()}
        try:
            record.update(status="ok", **_run_one(config, spec, dataset, cache, out_dir, config_id))
        except Exception as exc:  # one failing config must not stop the grid
            logger.exception("Configuration failed: %s", label)
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        record["timestamp"] = datetime.now(UTC).isoformat()
        _append(results_path, record)
        records.append(record)
    return records
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_experiment.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/experiment.py tests/test_experiment.py
git commit -m "feat: add resumable experiment runner with error isolation"
```

---

### Task 10: Rapport (Markdown + graphiques)

**Files:**
- Create: `src/rag_eval_lab/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: le format d'enregistrement de `results.jsonl` (Task 9)
- Produces (`rag_eval_lab.report`) :
  - `load_results(results_dir: Path) -> list[dict]` : dernier enregistrement par `config_id`, uniquement `status == "ok"`.
  - `pareto_front(points: list[tuple[float, float]]) -> set[int]` : points `(latence, qualité)` non dominés (latence plus basse et qualité plus haute).
  - `build_report(results_dir: Path) -> Path` : écrit `report.md`, `pareto.png`, `ranking.png` dans `results_dir` ; renvoie le chemin de `report.md` ; `ValueError` si aucun résultat `ok`.
  - Métrique principale = `ndcg@<plus grand k disponible>`.

- [ ] **Step 1: Écrire les tests**

```python
import json

import pytest

from rag_eval_lab.report import build_report, load_results, pareto_front


def _record(cid, retriever, ndcg, p50, status="ok"):
    record = {"config_id": cid, "status": status, "chunker": "none",
              "retriever": retriever, "reranker": "none"}
    if status == "ok":
        record.update(
            metrics={"ndcg@10": ndcg, "mrr@10": ndcg, "recall@5": ndcg, "recall@10": ndcg},
            latency_ms={"p50": p50, "p95": p50 * 2},
            index_time_s=1.5,
            n_chunks=10,
            n_queries=5,
        )
    else:
        record["error"] = "RuntimeError: boom"
    return record


def _write(results_dir, records):
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_pareto_front():
    points = [(10.0, 0.5), (20.0, 0.6), (30.0, 0.55), (5.0, 0.3)]
    assert pareto_front(points) == {0, 1, 3}


def test_load_results_keeps_last_ok_record(tmp_path):
    _write(tmp_path, [
        _record("a", "bm25", 0.0, 1.0, status="error"),
        _record("a", "bm25", 0.4, 1.0),
        _record("b", "dense:x", 0.6, 30.0),
        _record("c", "dense:y", 0.0, 0.0, status="error"),
    ])
    results = load_results(tmp_path)
    assert sorted(r["config_id"] for r in results) == ["a", "b"]


def test_build_report(tmp_path):
    _write(tmp_path, [
        _record("a", "bm25", 0.4, 2.0),
        _record("b", "dense:x", 0.6, 30.0),
        _record("c", "hybrid:bm25+x", 0.55, 60.0),
    ])
    path = build_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path.name == "report.md"
    assert text.index("dense:x") < text.index("hybrid:bm25+x") < text.index("| bm25")
    assert "ndcg@10" in text
    assert (tmp_path / "pareto.png").exists()
    assert (tmp_path / "ranking.png").exists()


def test_build_report_without_results_raises(tmp_path):
    _write(tmp_path, [_record("a", "bm25", 0.0, 1.0, status="error")])
    with pytest.raises(ValueError, match="No successful"):
        build_report(tmp_path)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_report.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.report'`

- [ ] **Step 3: Implémenter `src/rag_eval_lab/report.py`**

```python
"""Turn results.jsonl into a Markdown report with charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_results(results_dir: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    path = Path(results_dir) / "results.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["config_id"]] = record
    return [r for r in latest.values() if r.get("status") == "ok"]


def pareto_front(points: list[tuple[float, float]]) -> set[int]:
    """Indices of (latency, quality) points not dominated by any other point."""
    front = set()
    for i, (lat_i, q_i) in enumerate(points):
        dominated = any(
            lat_j <= lat_i and q_j >= q_i and (lat_j < lat_i or q_j > q_i)
            for j, (lat_j, q_j) in enumerate(points)
            if j != i
        )
        if not dominated:
            front.add(i)
    return front


def _metric_sort_key(name: str) -> tuple[str, int]:
    base, _, k = name.partition("@")
    return base, int(k) if k.isdigit() else 0


def _label(record: dict) -> str:
    return f"{record['chunker']} | {record['retriever']} | {record['reranker']}"


def _plot_pareto(results: list[dict], primary: str, front: set[int], path: Path) -> None:
    latencies = [r["latency_ms"]["p50"] for r in results]
    scores = [r["metrics"][primary] for r in results]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(latencies, scores, color="#9aa5b1", label="configuration")
    front_sorted = sorted(front, key=lambda i: latencies[i])
    ax.plot(
        [latencies[i] for i in front_sorted],
        [scores[i] for i in front_sorted],
        "o-",
        color="#2563eb",
        label="Pareto front",
    )
    for i in front_sorted:
        ax.annotate(_label(results[i]), (latencies[i], scores[i]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    if min(latencies) > 0 and max(latencies) / min(latencies) > 20:
        ax.set_xscale("log")
    ax.set_xlabel("Query latency p50 (ms)")
    ax.set_ylabel(primary)
    ax.set_title(f"Quality vs latency ({primary})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ranking(results: list[dict], primary: str, path: Path) -> None:
    ordered = sorted(results, key=lambda r: r["metrics"][primary])
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(ordered))))
    ax.barh([_label(r) for r in ordered], [r["metrics"][primary] for r in ordered],
            color="#2563eb")
    ax.set_xlabel(primary)
    ax.set_title(f"Configurations ranked by {primary}")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    results = load_results(results_dir)
    if not results:
        raise ValueError(f"No successful results in {results_dir / 'results.jsonl'}")

    metric_names = sorted(results[0]["metrics"], key=_metric_sort_key)
    ndcg_names = [m for m in metric_names if m.startswith("ndcg@")]
    primary = max(ndcg_names, key=_metric_sort_key)
    results.sort(key=lambda r: r["metrics"][primary], reverse=True)
    front = pareto_front([(r["latency_ms"]["p50"], r["metrics"][primary]) for r in results])

    header = ["Rank", "Chunker", "Retriever", "Reranker", *metric_names,
              "p50 (ms)", "p95 (ms)", "Index (s)", "Chunks", "Pareto"]
    lines = [
        "# Retrieval benchmark report",
        "",
        f"- Configurations: {len(results)}",
        f"- Queries: {results[0].get('n_queries', '?')}",
        f"- Primary metric: `{primary}`",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for i, r in enumerate(results):
        cells = [
            str(i + 1), r["chunker"], r["retriever"], r["reranker"],
            *(f"{r['metrics'][m]:.3f}" for m in metric_names),
            f"{r['latency_ms']['p50']:.1f}", f"{r['latency_ms']['p95']:.1f}",
            f"{r['index_time_s']:.1f}", str(r["n_chunks"]), "★" if i in front else "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "![Quality vs latency](pareto.png)",
        "",
        "![Ranking](ranking.png)",
        "",
        "## Notes",
        "",
        "- Latency is measured per query on CPU, including query encoding and reranking.",
        "- Index time includes embedding computation only when the embedding cache was cold.",
        "- ★ marks configurations on the quality/latency Pareto front.",
        "",
    ]

    _plot_pareto(results, primary, front, results_dir / "pareto.png")
    _plot_ranking(results, primary, results_dir / "ranking.png")
    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_report.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/report.py tests/test_report.py
git commit -m "feat: add Markdown report with Pareto and ranking charts"
```

---

### Task 11: CLI `rel` et test de bout en bout

**Files:**
- Create: `src/rag_eval_lab/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_config` (config) ; `run_experiment` (experiment) ; `build_report` (report)
- Produces (`rag_eval_lab.cli`) : `app` (Typer) avec les commandes :
  - `rel run CONFIG [--out DIR] [--verbose]` : sortie par défaut `results/<nom du fichier config sans extension>` ; config invalide → message sur stderr, code 1.
  - `rel report RESULTS_DIR` : écrit le rapport ; aucun résultat → message, code 1.

- [ ] **Step 1: Écrire les tests**

```python
import yaml
from typer.testing import CliRunner

from rag_eval_lab.cli import app

runner = CliRunner()


def _write_config(tmp_path, mini_jsonl, retrievers):
    config = {
        "dataset": {"name": "jsonl", "path": str(mini_jsonl)},
        "k": 5,
        "ks": [1, 5],
        "candidates": 20,
        "cache_dir": str(tmp_path / "cache"),
        "grid": {"chunker": ["none", "fixed_200_50"], "retriever": retrievers},
    }
    path = tmp_path / "mini.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_cli_run_and_report_end_to_end(tmp_path, mini_jsonl):
    config = _write_config(tmp_path, mini_jsonl, ["bm25", "dense:hashing", "hybrid:bm25+hashing"])
    out = tmp_path / "results"
    result = runner.invoke(app, ["run", str(config), "--out", str(out)])
    assert result.exit_code == 0, result.output
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert '"status": "error"' not in "".join(lines)

    result = runner.invoke(app, ["report", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "report.md").exists()


def test_cli_rejects_invalid_config(tmp_path, mini_jsonl):
    config = _write_config(tmp_path, mini_jsonl, ["dense:nope"])
    result = runner.invoke(app, ["run", str(config)])
    assert result.exit_code == 1
    assert "Unknown encoder" in result.output


def test_cli_report_without_results(tmp_path):
    (tmp_path / "results.jsonl").write_text("", encoding="utf-8")
    result = runner.invoke(app, ["report", str(tmp_path)])
    assert result.exit_code == 1
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.cli'`

- [ ] **Step 3: Implémenter `src/rag_eval_lab/cli.py`**

```python
"""Command-line interface: `rel run` and `rel report`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from rag_eval_lab.config import load_config
from rag_eval_lab.experiment import run_experiment
from rag_eval_lab.report import build_report

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


@app.command()
def run(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="YAML config")],
    out: Annotated[Path | None, typer.Option(help="Output directory")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run every configuration of the grid (already finished ones are skipped)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        cfg = load_config(config)
    except ValidationError as exc:
        typer.echo(f"Invalid config {config}:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc
    out_dir = out or Path("results") / config.stem
    records = run_experiment(cfg, out_dir)
    failed = [r for r in records if r["status"] == "error"]
    typer.echo(f"{len(records)} configuration(s) run, {len(failed)} failed -> {out_dir}")


@app.command()
def report(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Build report.md and charts from results.jsonl."""
    try:
        path = build_report(results_dir)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Report written to {path}")
```

- [ ] **Step 4: Vérifier le succès et la suite complète**

Run: `uv run pytest tests/test_cli.py -q`
Expected: `3 passed`

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `68 passed`, `All checks passed!`, aucun fichier à reformater (sinon `uv run ruff format .` puis relancer).

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/cli.py tests/test_cli.py
git commit -m "feat: add rel CLI with run and report commands"
```

---

### Task 12: Validation sur Alloprof avec une grille réduite

Première exécution réelle : télécharge le dataset et les modèles (quelques Go au total), vérifie que tout fonctionne et mesure les temps pour dimensionner la grille complète.

**Files:**
- Create: `configs/smoke.yaml`

**Interfaces:**
- Consumes: CLI `rel` (Task 11), alias de `factory.ENCODERS` / `factory.RERANKERS`.

- [ ] **Step 1: Installer l'extra `models`**

Run: `uv sync --extra models`
Expected: installation de `sentence-transformers`, `torch` (CPU sous Windows), `datasets`, sans erreur.

- [ ] **Step 2: Vérifier le chargement Alloprof**

Run :
```bash
uv run python -c "from rag_eval_lab.data import load_alloprof, sample_dataset; d = load_alloprof(); s = sample_dataset(d, None, None, 0); print(len(d.documents), len(d.queries), len(s.queries))"
```
Expected : environ `2556 2316 N` avec N proche de 2316 (requêtes dont au moins un doc pertinent est dans le corpus). Si N est très inférieur (< 1500), s'arrêter et signaler le problème à l'utilisateur avant de continuer.

- [ ] **Step 3: Créer `configs/smoke.yaml`**

```yaml
# Small grid to validate the setup and measure timings before the full run.
dataset: { name: alloprof, n_queries: 50, seed: 42 }
k: 10
ks: [5, 10]
candidates: 30
grid:
  chunker:   [none, recursive_1000]
  retriever: [bm25, dense:e5-small, hybrid:bm25+e5-small]
  reranker:  [none, minilm]
exclude:
  - { chunker: none, reranker: minilm }
```

- [ ] **Step 4: Lancer la grille réduite**

Run: `uv run rel run configs/smoke.yaml`
Expected : `9 configuration(s) run, 0 failed -> results\smoke`. Durée attendue : environ 10 à 20 min, surtout pour encoder le corpus découpé avec e5-small.

Puis : `uv run rel report results/smoke`
Expected : `Report written to results\smoke\report.md`. Vérifier dans le rapport que BM25 a un `ndcg@10` > 0.2 et que les configs denses ne sont pas à ~0, ce qui signalerait un bug de préfixes ou d'ids.

- [ ] **Step 5: Estimer la durée de la grille complète**

Noter dans le compte rendu à l'utilisateur : le temps d'indexation `recursive_1000 + dense:e5-small` (encodage à froid) et les latences p50 avec/sans reranker. Estimer la grille v1 (Task 13) : l'encodage bge-m3 est environ 5 fois plus lent qu'e5-small ; la recherche est proportionnelle à `n_queries` (500 au lieu de 50). Si l'estimation dépasse 2 h, réduire `n_queries` à 300 dans `configs/v1.yaml`.

- [ ] **Step 6: Commit**

```bash
git add configs/smoke.yaml
git commit -m "chore: add smoke config validated on Alloprof"
```

(Les résultats de la grille réduite ne sont pas commités.)

---

### Task 13: Benchmark complet, README et analyse

**Files:**
- Create: `configs/v1.yaml`, `docs/results-v1.md`
- Modify: `README.md` (remplacement complet)
- Commit: `results/v1/results.jsonl`, `results/v1/report.md`, `results/v1/pareto.png`, `results/v1/ranking.png`

**Interfaces:**
- Consumes: CLI `rel`, rapport de la Task 10.

- [ ] **Step 1: Créer `configs/v1.yaml`**

```yaml
# Full v1 grid: chunking x first-stage retriever x reranker on AlloprofRetrieval.
dataset: { name: alloprof, n_queries: 500, seed: 42 }
k: 10
ks: [5, 10]
candidates: 30
grid:
  chunker:   [none, fixed_1000_200, recursive_1000]
  retriever: [bm25, dense:e5-small, dense:bge-m3, hybrid:bm25+bge-m3]
  reranker:  [none, minilm]
exclude:
  - { chunker: none, reranker: minilm }
```
(Si la Task 12 a conclu qu'il fallait réduire, remplacer `n_queries: 500` par `300`.)

- [ ] **Step 2: Lancer la grille complète en arrière-plan**

Run : `uv run rel run configs/v1.yaml` (outil Bash avec `run_in_background: true` ; durée estimée en Task 12).
Expected : `20 configuration(s) run, 0 failed -> results\v1`. En cas d'interruption, relancer la même commande : les configurations terminées sont sautées.

- [ ] **Step 3: Générer le rapport**

Run: `uv run rel report results/v1`
Expected: `Report written to results\v1\report.md`

- [ ] **Step 4: Extraire les conclusions chiffrées**

Lire `results/v1/report.md` et `results/v1/results.jsonl`, puis calculer au moins :
1. le gain de `ndcg@10` du meilleur dense par rapport à BM25 (même chunker) ;
2. le gain apporté par le reranker `minilm` et son coût en latence p50 (même chunker et retriever) ;
3. l'effet du chunking (`none` vs `recursive_1000` vs `fixed_1000_200`) pour bge-m3 ;
4. la configuration recommandée sur le front de Pareto (meilleur compromis qualité/latence).

Examiner aussi 3 requêtes où la meilleure configuration échoue (lire `results/v1/runs/<config_id>.json` et le texte des requêtes via `load_alloprof`), pour la section « Failure analysis ».

- [ ] **Step 5: Écrire `docs/results-v1.md`**

Structure imposée (en anglais), avec les **valeurs réelles** issues de l'étape 4 :

```markdown
# Results v1 — Retrieval on AlloprofRetrieval

## Setup
Dataset (documents, sampled queries, seed), hardware (CPU, RAM), models with Hugging Face ids,
chunk sizes in characters, candidates=30, commit hash (from `env.git_commit`).

## Main results
Copy of the ranking table from results/v1/report.md, plus the two charts.

## Findings
1. Dense vs lexical — measured gain and when BM25 still wins.
2. Reranking — measured gain vs latency cost.
3. Chunking — effect of whole-document vs chunked indexing (long Alloprof documents are
   truncated to 512 tokens without chunking).
4. Recommendation — the Pareto configuration to pick, and why.

## Failure analysis
Three failed queries: query text, expected document title, retrieved titles, likely cause.

## Limitations
Single dataset, 500 sampled queries, CPU-only latency, no answer-generation evaluation (v2).
```

- [ ] **Step 6: Réécrire `README.md`**

Structure imposée (en anglais), avec les valeurs réelles :

```markdown
# rag-eval-lab

[![CI](https://github.com/EmericLi/rag-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/EmericLi/rag-eval-lab/actions/workflows/ci.yml)

**Which RAG retrieval setup should you pick for French documents, and what does each
component actually buy you?** rag-eval-lab benchmarks chunking strategies, lexical, dense and
hybrid retrievers, and cross-encoder reranking on a French dataset, measuring quality
(nDCG, MRR, Recall) against latency. Everything runs locally on a laptop CPU.

## Key results
Short table: top 5 configurations + BM25 baseline (from results/v1/report.md),
the Pareto chart (results/v1/pareto.png), and the 3 numbered findings in one sentence each.

## Quickstart
    uv sync --extra models
    uv run rel run configs/v1.yaml
    uv run rel report results/v1

## How it works
The pipeline diagram (Dataset → Chunker → Retriever → Reranker → doc ids), one line per module,
and how a YAML grid becomes a set of runs (resumable, embeddings cached).

## Configuration
The annotated content of configs/v1.yaml and the list of available specs
(chunkers, retrievers, encoders, rerankers).

## Development
    uv sync
    uv run pytest
    uv run ruff check .

## Roadmap
- v2: answer generation + LLM-as-a-judge (local Ollama vs free-tier API), synthetic questions
  on any corpus (example: French Python docs).
- v3: Streamlit demo / API.

## License
MIT. Dataset: AlloprofRetrieval (lyon-nlp/alloprof, Apache-2.0).
```

Ajouter aussi un fichier `LICENSE` (texte MIT standard, `Copyright (c) 2026 EmericLi`).

- [ ] **Step 7: Vérification finale**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `68 passed`, aucune erreur ruff.

Vérifier que les liens d'images du README (`results/v1/pareto.png`) pointent vers des fichiers existants et qu'aucune valeur n'est laissée entre crochets ou en exemple.

- [ ] **Step 8: Commit**

```bash
git add configs/v1.yaml docs/results-v1.md README.md LICENSE results/v1/results.jsonl results/v1/report.md results/v1/pareto.png results/v1/ranking.png
git commit -m "docs: add v1 benchmark results, analysis and README"
```

La publication sur GitHub (création du dépôt `EmericLi/rag-eval-lab` et `git push`) n'est **pas** dans ce plan : elle se fait uniquement avec l'accord explicite de l'utilisateur.
