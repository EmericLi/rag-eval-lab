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
        "Deuxième paragraphe un peu plus long que le premier. Il a deux phrases.\n\n" + "mot " * 60
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
