"""Human annotation session used to calibrate the judge."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from rag_eval_lab.generation_experiment import load_answers

ANSWERS = {"o": True, "n": False, "?": None}


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
    """Last record per answer_id, so re-labelling replaces the earlier label."""
    path = Path(out_dir) / "annotations.jsonl"
    if not path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["answer_id"]] = record
    return list(latest.values())


def append_annotation(out_dir: Path, record: dict) -> None:
    path = Path(out_dir) / "annotations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ask(
    read: Callable[[str], str], write: Callable[[str], None], question: str
) -> bool | None | str:
    while True:
        reply = read(f"{question} [o/n/? = je ne sais pas, q = quitter] ").strip().lower()
        if reply == "q":
            return "quit"
        if reply in ANSWERS:
            return ANSWERS[reply]
        write("Réponse attendue : o, n, ? ou q.")


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
        write(f"\n--- {index}/{len(todo)} (contexte : {answer['context']})")
        write(f"Question : {answer['question']}")
        for position, passage in enumerate(answer["passages"], start=1):
            write(f"[{position}] {passage['text'][:600]}")
        if not answer["passages"]:
            write("(aucun passage fourni)")
        write(f"Réponse générée : {answer['answer']}")
        write(f"Réponse de référence : {answer.get('reference', '(absente)')}")

        faithfulness: bool | None = None
        if answer["passages"]:
            reply = _ask(read, write, "Fidèle aux passages ?")
            if reply == "quit":
                break
            faithfulness = reply
        reply = _ask(read, write, "Juste par rapport à la référence ?")
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
