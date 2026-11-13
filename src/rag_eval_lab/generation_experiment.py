"""Generation stage: one answer per (context, query), resumable and error-isolated."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from rag_eval_lab.cache import EmbeddingCache
from rag_eval_lab.chunking import parse_chunker
from rag_eval_lab.context import ContextSource, EmptyContext, OracleContext, RetrievedContext
from rag_eval_lab.data import load_benchmark
from rag_eval_lab.factory import build_retriever
from rag_eval_lab.generation import OllamaGenerator
from rag_eval_lab.generation_config import GenerationConfig
from rag_eval_lab.judging import JudgeParseError, OllamaJudge
from rag_eval_lab.ollama_client import OllamaClient, OllamaError
from rag_eval_lab.pipeline import RetrievalPipeline
from rag_eval_lab.prompts import ANSWER_PROMPT_VERSION, JUDGE_PROMPT_VERSION
from rag_eval_lab.types import Chunk, Dataset

logger = logging.getLogger(__name__)

V1_SAMPLE_SIZE = 300
"""The v1 benchmark sampled this many queries; v2 keeps a prefix of that sample."""


def build_dataset(cfg: GenerationConfig) -> Dataset:
    requested = cfg.dataset.n_queries
    # Up to the v1 sample size, take a prefix of that sample; beyond it, sample directly.
    size = None if requested is None else max(V1_SAMPLE_SIZE, requested)
    base = load_benchmark(cfg.dataset.model_copy(update={"n_queries": size}))
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


def _guarded_append_json_line(path: Path, record: dict) -> None:
    """Append a JSON record as a line, guarding against torn fragments.

    If the file is non-empty and doesn't end with a newline (from a previous crash),
    prepend a newline before writing the new record. This ensures each record is
    on its own line and protects against merging into a torn fragment.
    """
    prefix = ""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                prefix = "\n"  # a crash left an unterminated fragment: do not merge into it
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + json.dumps(record, ensure_ascii=False) + "\n")


def _completed_ids(path: Path, keep: Callable[[dict], bool] | None = None) -> set[str]:
    """Ids of the records already stored with status "ok" (and accepted by `keep`).

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
            if record.get("status") == "ok" and (keep is None or keep(record)):
                done.add(record["answer_id"])
        except (json.JSONDecodeError, AttributeError, KeyError):
            logger.warning("skipping unreadable line %d of %s", number, path)
    return done


def _append(path: Path, record: dict) -> None:
    """Append a JSON record to a file, using the guarded append helper."""
    _guarded_append_json_line(path, record)


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
        client,
        cfg.generator.model,
        cfg.generator.max_tokens,
        cfg.generator.temperature,
        cfg.generator.num_ctx,
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
                "prompt_version": ANSWER_PROMPT_VERSION,
                "generator": {
                    "model": cfg.generator.model,
                    "digest": digest,
                    "max_tokens": cfg.generator.max_tokens,
                    "temperature": cfg.generator.temperature,
                    "num_ctx": cfg.generator.num_ctx,
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
                    prompt_eval_count=completion.prompt_eval_count,
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


def load_answers(out_dir: Path) -> list[dict]:
    """Last record per answer_id, successful ones only; unreadable lines are skipped."""
    path = Path(out_dir) / "answers.jsonl"
    latest: dict[str, dict] = {}
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                latest[record["answer_id"]] = record
            except (json.JSONDecodeError, TypeError, KeyError):
                logger.warning("skipping unreadable line %d of %s", number, path)
    return [r for r in latest.values() if r.get("status") == "ok"]


def run_judging(
    cfg: GenerationConfig, out_dir: Path, client: OllamaClient | None = None
) -> list[dict]:
    """Judge every successful answer not yet judged; the reference comes from the answer record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / "judgments.jsonl"

    def same_judge(record: dict) -> bool:
        # A judgment made by another model, prompt or window is stale: judge again.
        judge_info = record.get("judge") or {}
        return (
            record.get("prompt_version") == JUDGE_PROMPT_VERSION
            and judge_info.get("model") == cfg.judge.model
            and judge_info.get("num_ctx") == cfg.judge.num_ctx
        )

    done = _completed_ids(judgments_path, keep=same_judge)

    client = client or OllamaClient(timeout_s=cfg.judge.timeout_s)
    digest = client.model_digest(cfg.judge.model)
    judge = OllamaJudge(
        client,
        cfg.judge.model,
        cfg.judge.max_tokens,
        num_ctx=cfg.judge.num_ctx,
        temperature=cfg.judge.temperature,
    )

    records: list[dict] = []
    answers = [a for a in load_answers(out_dir) if a["answer_id"] not in done]
    for n, answer in enumerate(answers, start=1):
        record = {
            "answer_id": answer["answer_id"],
            "judge": {"model": cfg.judge.model, "digest": digest, "num_ctx": cfg.judge.num_ctx},
            "prompt_version": JUDGE_PROMPT_VERSION,
        }
        try:
            passages = [
                Chunk(p["id"], p["id"].split("#")[0], p["text"]) for p in answer["passages"]
            ]
            verdict = judge.verdict(
                answer["question"], passages, answer["answer"], answer.get("reference", "")
            )
            record.update(
                status="ok",
                faithfulness=verdict.faithfulness,
                relevance=verdict.relevance,
                correctness=verdict.correctness,
                rationales=verdict.rationales,
                latency_ms=verdict.latency_ms,
                prompt_eval_count=verdict.prompt_eval_count,
            )
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop an unattended run
            if isinstance(exc, JudgeParseError):
                logger.warning("judge parse error on %s", answer["answer_id"])
            elif isinstance(exc, OllamaError):
                logger.error("judge call failed on %s: %s", answer["answer_id"], exc)
            else:
                logger.exception("judge failed on %s", answer["answer_id"])
            record.update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                raw_output=getattr(exc, "raw_output", ""),
            )
        record["timestamp"] = datetime.now(UTC).isoformat()
        _append(judgments_path, record)
        records.append(record)
        if n % 10 == 0:
            logger.info("judged %d/%d", n, len(answers))
    return records
