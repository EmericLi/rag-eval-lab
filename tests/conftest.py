import json
from pathlib import Path

import pytest

from rag_eval_lab.types import Dataset, Document, Query

TOPICS = [
    "volcan",
    "photosynthese",
    "fraction",
    "adverbe",
    "revolution",
    "atome",
    "cellule",
    "triangle",
    "poeme",
    "electricite",
    "gravite",
    "democratie",
    "molecule",
    "equation",
    "verbe",
    "climat",
    "pyramide",
    "magnetisme",
    "planete",
    "ecosysteme",
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
