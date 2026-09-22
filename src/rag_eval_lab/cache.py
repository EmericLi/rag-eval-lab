"""On-disk cache for embedding matrices."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np


def make_key(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


class EmbeddingCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def get_or_compute(self, key: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
        path = self.directory / f"{key}.npy"
        if path.exists():
            return np.load(path)
        array = compute()
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / f"{key}.tmp.npy"
        np.save(tmp, array)
        tmp.replace(path)
        return array
