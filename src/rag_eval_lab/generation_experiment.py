"""Generation stage: one answer per (context, query), resumable and error-isolated."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from rag_eval_lab.cache import EmbeddingCache
from rag_eval_lab.chunking import parse_chunker
from rag_eval_lab.context import ContextSource, EmptyContext, OracleContext, RetrievedContext
from rag_eval_lab.data import load_benchmark
from rag_eval_lab.factory import build_retriever
from rag_eval_lab.generation import OllamaGenerator
from rag_eval_lab.generation_config import GenerationConfig
from rag_eval_lab.ollama_client import OllamaClient, OllamaError
from rag_eval_lab.pipeline import RetrievalPipeline
from rag_eval_lab.prompts import PROMPT_VERSION
from rag_eval_lab.types import Dataset

logger = logging.getLogger(__name__)

V1_SAMPLE_SIZE = 300
"""The v1 benchmark sampled this many queries; v2 keeps a prefix of that sample."""


def build_dataset(cfg: GenerationConfig) -> Dataset:
    requested = cfg.dataset.n_queries
    base = load_benchmark(cfg.dataset.model_copy(update={"n_queries": V1_SAMPLE_SIZE}))
    queries = base.queries if requested is None else base.queries[:requested]
    qrels = {q.id: base.qrels[q.id] for q in queries}
    return Dataset(name=base.name, documents=base.documents, queries=queries, qrels=qrels)


def build_context_sources(
    cfg: GenerationConfig, dataset: Dataset, only: set[str] | None = None
) -> list[ContextSource]:
    """Build the context sources in config order; `only` restricts it to those names.

    Retrieved contexts are indexed here, which is the expensive step, so callers with
    nothing left to generate for a context should leave it out of `only`.
    """
    cache = EmbeddingCache(Path(cfg.cache_dir))
    sources: list[ContextSource] = []
    for context in cfg.contexts:
        if only is not None and context.name not in only:
            continue
        if context.kind == "empty":
            sources.append(EmptyContext(context.name))
        elif context.kind == "oracle":
            sources.append(
                OracleContext(
                    context.name,
                    dataset.qrels,
                    dataset.documents,
                    parse_chunker(cfg.oracle_chunker),
                    cfg.n_passages,
                )
            )
        else:
            pipeline = RetrievalPipeline(
                chunker=parse_chunker(context.chunker or ""),
                retriever=build_retriever(context.retriever or "", cache),
            )
            logger.info("indexing context '%s'", context.name)
            pipeline.index(dataset.documents)
            sources.append(RetrievedContext(context.name, pipeline, cfg.n_passages))
    return sources


def _completed_ids(path: Path) -> set[str]:
    """Ids of the answers already stored with status "ok".

    A crash mid-write can leave a truncated line; it is skipped with a warning
    rather than making the whole run unresumable.
    """
    if not path.exists():
        return set()
    done = set()
    text = path.read_text(encoding="utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record.get("status") == "ok":
                done.add(record["answer_id"])
        except (json.JSONDecodeError, AttributeError, KeyError):
            logger.warning("skipping unreadable line %d of %s", number, path)
    return done


def _append(path: Path, record: dict) -> None:
    prefix = ""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                prefix = "\n"  # a crash left an unterminated fragment: do not merge into it
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + json.dumps(record, ensure_ascii=False) + "\n")


def run_generation(
    cfg: GenerationConfig,
    out_dir: Path,
    client: OllamaClient | None = None,
    dataset: Dataset | None = None,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"
    done = _completed_ids(answers_path)

    client = client or OllamaClient(timeout_s=cfg.generator.timeout_s)
    digest = client.model_digest(cfg.generator.model)
    generator = OllamaGenerator(
        client, cfg.generator.model, cfg.generator.max_tokens, cfg.generator.temperature
    )
    dataset = dataset or build_dataset(cfg)

    # Resume check first: only contexts with pending items are built (and indexed).
    pending = {
        context.name
        for context in cfg.contexts
        if any(cfg.answer_id(context.name, q.id) not in done for q in dataset.queries)
    }
    sources = build_context_sources(cfg, dataset, only=pending) if pending else []

    records: list[dict] = []
    for source in sources:
        for n, query in enumerate(dataset.queries, start=1):
            answer_id = cfg.answer_id(source.name, query.id)
            if answer_id in done:
                continue
            record = {
                "answer_id": answer_id,
                "context": source.name,
                "query_id": query.id,
                "question": query.text,
                "reference": query.answer,
                "prompt_version": PROMPT_VERSION,
                "generator": {
                    "model": cfg.generator.model,
                    "digest": digest,
                    "max_tokens": cfg.generator.max_tokens,
                    "temperature": cfg.generator.temperature,
                },
            }
            try:
                passages = source.passages(query)
                completion = generator.answer(query.text, passages)
                record.update(
                    status="ok",
                    answer=completion.text,
                    passages=[{"id": p.id, "text": p.text} for p in passages],
                    n_passage_chars=sum(len(p.text) for p in passages),
                    latency_ms=completion.latency_ms,
                    eval_count=completion.eval_count,
                )
            except Exception as exc:  # noqa: BLE001 - one bad item must not stop an unattended run
                if isinstance(exc, OllamaError):
                    logger.error("generation failed for %s / %s: %s", source.name, query.id, exc)
                else:
                    logger.exception("generation failed for %s / %s", source.name, query.id)
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            record["timestamp"] = datetime.now(UTC).isoformat()
            _append(answers_path, record)
            records.append(record)
            if n % 10 == 0:
                logger.info("context %s: %d/%d", source.name, n, len(dataset.queries))
    return records
