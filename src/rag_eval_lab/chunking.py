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
