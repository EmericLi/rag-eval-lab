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
