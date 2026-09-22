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
