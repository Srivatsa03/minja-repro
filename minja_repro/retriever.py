"""Similarity for the memory bank.

MINJA only works if the poisoned record is actually retrieved by an innocent
victim query, so the retriever is load bearing: a reproduction that uses a
different similarity function is not measuring quite the same attack. The paper
uses embedding similarity over a vector store.

Two backends are provided and every reported number has to name which one ran.
The lexical default needs no network, no model and no API key, which keeps the
harness testable and keeps a failed run cheap. It is weaker than embeddings, and
a weaker retriever can make the attack look easier or harder for reasons that
have nothing to do with the attack, so it is a stated substitution rather than a
silent one.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, Sequence

_WORD = re.compile(r"[a-z0-9]+")

# Words this common carry no retrieval signal and let unrelated records match.
_STOP = frozenset(
    """a an the and or but if then than that this these those is are was were be been am
    of to in on at by for with from as it its they them their you your we our not no
    do does did have has had can will would should may might must which who whom what
    when where why how all any both each few more most other some such only own same so
    too very s t just don now""".split()
)


def _terms(text: str) -> Counter[str]:
    return Counter(w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1)


class Retriever(Protocol):
    def score(self, query: str, document: str) -> float: ...

    @property
    def name(self) -> str: ...


class LexicalRetriever:
    """Cosine over stopword-filtered unigram counts.

    Deterministic, offline, and good enough to exercise retrieval-time behaviour.
    Not a stand-in for embedding similarity when reporting an attack rate.
    """

    name = "lexical-cosine"

    def score(self, query: str, document: str) -> float:
        a, b = _terms(query), _terms(document)
        if not a or not b:
            return 0.0
        shared = set(a) & set(b)
        if not shared:
            return 0.0
        dot = sum(a[t] * b[t] for t in shared)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb)


class EmbeddingRetriever:
    """Cosine over vectors from a user-supplied embedding callable.

    Kept deliberately thin: the caller owns the model and the credentials, so the
    package has no network dependency and no key handling of its own.
    """

    def __init__(self, embed: "callable[[Sequence[str]], Sequence[Sequence[float]]]", name: str) -> None:
        self._embed = embed
        self._name = name
        self._cache: dict[str, Sequence[float]] = {}

    @property
    def name(self) -> str:
        return self._name

    def _vector(self, text: str) -> Sequence[float]:
        if text not in self._cache:
            self._cache[text] = self._embed([text])[0]
        return self._cache[text]

    def score(self, query: str, document: str) -> float:
        u, v = self._vector(query), self._vector(document)
        dot = sum(x * y for x, y in zip(u, v))
        nu = math.sqrt(sum(x * x for x in u))
        nv = math.sqrt(sum(y * y for y in v))
        return dot / (nu * nv) if nu and nv else 0.0
