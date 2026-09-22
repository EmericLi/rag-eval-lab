import rag_eval_lab
from rag_eval_lab.types import Chunk, Hit


def test_version():
    assert rag_eval_lab.__version__ == "0.1.0"


def test_hit_is_hashable_and_comparable():
    chunk = Chunk(id="d1#0", doc_id="d1", text="bonjour")
    assert Hit(chunk, 1.0) == Hit(chunk, 1.0)
    assert len({Hit(chunk, 1.0), Hit(chunk, 1.0)}) == 1


def test_mini_dataset_fixture(mini_dataset):
    assert len(mini_dataset.documents) == 20
    assert mini_dataset.qrels["q0"] == {"d0": 1}
