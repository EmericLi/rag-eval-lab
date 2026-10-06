"""Human annotation session used to calibrate the judge."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from rag_eval_lab.generation_experiment import load_answers

logger = logging.getLogger(__name__)

ANSWERS = {"o": True, "n": False, "?": None}
VIEW_KEY = "v"
PASSAGE_DISPLAY_LIMIT = 1200


def sample_for_annotation(answers: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over contexts so no context dominates the calibration set."""
    rng = random.Random(seed)
    by_context: dict[str, list[dict]] = {}
    for answer in answers:
        by_context.setdefault(answer["context"], []).append(answer)
    for records in by_context.values():
        records.sort(key=lambda r: r["answer_id"])
        rng.shuffle(records)
    selected: list[dict] = []
    position = 0
    while len(selected) < n:
        available = [c for c in sorted(by_context) if len(by_context[c]) > position]
        if not available:
            break
        for context in available:
            selected.append(by_context[context][position])
            if len(selected) == n:
                break
        position += 1
    return selected


def load_annotations(out_dir: Path) -> list[dict]:
    """Last record per answer_id; unreadable lines are skipped with a warning.

    Re-labelling replaces the earlier label. A Ctrl-C during a write can leave a torn
    line, which must not break resuming or the report (same policy as load_answers).
    """
    path = Path(out_dir) / "annotations.jsonl"
    if not path.exists():
        return []
    latest: dict[str, dict] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            latest[record["answer_id"]] = record
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.warning("skipping unreadable line %d of %s", number, path)
    return list(latest.values())


def append_annotation(out_dir: Path, record: dict) -> None:
    path = Path(out_dir) / "annotations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _passage_view(text: str, full: bool) -> str:
    """The passage as shown to the annotator, with a visible marker when it is cut."""
    if full or len(text) <= PASSAGE_DISPLAY_LIMIT:
        return text
    return (
        f"{text[:PASSAGE_DISPLAY_LIMIT]}… [tronqué, {len(text)} caractères au total"
        f" — tape {VIEW_KEY} pour tout afficher]"
    )


def _show_item(
    write: Callable[[str], None], answer: dict, index: int, total: int, full: bool
) -> None:
    write(f"\n--- {index}/{total} (contexte : {answer['context']})")
    write(f"Question : {answer['question']}")
    for position, passage in enumerate(answer["passages"], start=1):
        write(f"[{position}] {_passage_view(passage['text'], full)}")
    if not answer["passages"]:
        write("(aucun passage fourni)")
    write(f"Réponse générée : {answer['answer']}")
    write(f"Réponse de référence : {answer.get('reference', '(absente)')}")


def _ask(
    read: Callable[[str], str],
    write: Callable[[str], None],
    question: str,
    on_view: Callable[[], None],
) -> bool | None | str:
    while True:
        reply = (
            read(f"{question} [o/n/? = je ne sais pas, {VIEW_KEY} = tout afficher, q = quitter] ")
            .strip()
            .lower()
        )
        if reply == "q":
            return "quit"
        if reply == VIEW_KEY:
            on_view()
            continue
        if reply in ANSWERS:
            return ANSWERS[reply]
        write(f"Réponse attendue : o, n, ?, {VIEW_KEY} ou q.")


def run_annotation_session(
    out_dir: Path,
    n: int,
    seed: int,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    """Label sampled answers. The judge's verdict is never shown, to avoid anchoring."""
    out_dir = Path(out_dir)
    answers = load_answers(out_dir)
    if not answers:
        raise ValueError(f"no answers to annotate in {out_dir / 'answers.jsonl'}")
    already = {record["answer_id"] for record in load_annotations(out_dir)}
    todo = [a for a in sample_for_annotation(answers, n, seed) if a["answer_id"] not in already]

    added = 0
    for index, answer in enumerate(todo, start=1):
        started = time.perf_counter()
        total = len(todo)
        _show_item(write, answer, index, total, full=False)

        def show_full(answer=answer, index=index, total=total) -> None:
            _show_item(write, answer, index, total, full=True)

        faithfulness: bool | None = None
        if answer["passages"]:
            reply = _ask(read, write, "Fidèle aux passages ?", show_full)
            if reply == "quit":
                break
            faithfulness = reply
        reply = _ask(read, write, "Juste par rapport à la référence ?", show_full)
        if reply == "quit":
            break
        append_annotation(
            out_dir,
            {
                "answer_id": answer["answer_id"],
                "faithfulness": faithfulness,
                "correctness": reply,
                "seconds": round(time.perf_counter() - started, 1),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        added += 1
    return added
