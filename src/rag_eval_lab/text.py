"""Text normalization shared by lexical components."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
