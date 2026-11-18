"""Append-only JSON-lines files that survive a crash in the middle of a write."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_records(path: Path) -> list[dict]:
    """Every JSON object of the file, in order; unreadable lines are skipped with a warning.

    A killed run can leave a truncated last line. It must not make the file unreadable,
    or a multi-hour run could never be resumed or reported.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        if isinstance(record, dict):
            records.append(record)
        else:
            logger.warning("skipping unreadable line %d of %s", number, path)
    return records


def latest_by(records: list[dict], key: str) -> list[dict]:
    """The last record for each value of `key`; records without the key are dropped."""
    latest = {record[key]: record for record in records if key in record}
    return list(latest.values())


def append_record(path: Path, record: dict) -> None:
    """Append one record on its own line, never merging it into a truncated fragment."""
    path = Path(path)
    prefix = ""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                prefix = "\n"  # a crash left an unterminated line
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + json.dumps(record, ensure_ascii=False) + "\n")
