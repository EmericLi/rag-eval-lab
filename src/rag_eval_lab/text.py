"""Text normalization shared by lexical components."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# French function words (articles, pronouns, prepositions, conjunctions, common forms of
# être and avoir), accent-folded. Deliberately free of content or question words, so the
# list is not tuned to the benchmark's queries.
_FRENCH_STOPWORDS = frozenset(
    """
    a ai aie aient aies ait as au aura aurai auraient aurais aurait auras aurez auriez aurions
    aurons auront aux avaient avais avait avec avez aviez avions avons ayant ayez ayons c ce
    ces cet cette d dans de des du elle elles en es est et etaient etais etait etant ete etes
    etiez etions eu eue eues eurent eus eusse eut eux furent fus fut il ils j je l la le les
    leur leurs lui m ma mais me meme mes moi mon n ne nos notre nous on ont ou par pas pour qu
    que qui s sa se sera serai seraient serais serait seras serez seriez serions serons seront
    ses soient sois soit sommes son sont soyez soyons suis sur t ta te tes toi ton tu un une
    vos votre vous y
    """.split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _fold(token: str) -> str:
    """Strip accents: 'été' and 'ete' must index the same term."""
    decomposed = unicodedata.normalize("NFD", token)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def analyze_fr(text: str) -> list[str]:
    """French analysis for lexical retrieval: fold accents, drop function words, strip plurals.

    shortcut: plural stripping is a bare trailing s/x rule, not a stemmer; swap in a Snowball
    stemmer if verb inflection starts to matter.
    """
    tokens = []
    for token in tokenize(text):
        token = _fold(token)
        if token in _FRENCH_STOPWORDS:
            continue
        if len(token) > 3 and token[-1] in "sx":
            token = token[:-1]
        tokens.append(token)
    return tokens
