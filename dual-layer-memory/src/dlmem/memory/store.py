"""External memory with BM25 retrieval (the fast, editable hippocampal layer)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from ..data.schema import Fact

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class ExternalMemory:
    """Append-only (by design) store with lexical BM25 retrieval.

    Write admission is the router's job; this class never filters on its own.
    """

    def __init__(self, top_k: int = 3, k1: float = 1.5, b: float = 0.75) -> None:
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.facts: list[Fact] = []
        self._ids: set[str] = set()
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._doc_len: list[int] = []
        self._total_len = 0

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.facts)

    def add(self, fact: Fact) -> bool:
        if fact.fact_id in self._ids:
            return False
        doc_id = len(self.facts)
        self.facts.append(fact)
        self._ids.add(fact.fact_id)
        toks = tokenize(fact.text)
        self._doc_len.append(len(toks))
        self._total_len += len(toks)
        for term, tf in Counter(toks).items():
            self._postings[term][doc_id] = tf
        return True

    def add_many(self, facts: Iterable[Fact]) -> int:
        return sum(1 for f in facts if self.add(f))

    def contains(self, fact_id: str) -> bool:
        return fact_id in self._ids

    def clear(self) -> None:
        self.__init__(self.top_k, self.k1, self.b)

    # ------------------------------------------------------------------ #
    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, {}))
        n = len(self.facts)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[int, float]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.facts:
            return {}
        n = len(self.facts)
        avgdl = self._total_len / max(1, n)
        acc: dict[int, float] = defaultdict(float)
        for term in set(q_tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_id, tf in postings.items():
                dl = self._doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, avgdl))
                acc[doc_id] += idf * tf * (self.k1 + 1) / denom
        return acc

    def retrieve(self, query: str, k: int | None = None) -> list[Fact]:
        k = k or self.top_k
        if k <= 0 or not self.facts:
            return []
        scores = self.scores(query)
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [self.facts[i] for i, _ in top]

    def texts(self, k: int | None = None) -> list[str]:
        facts = self.facts[: k or len(self.facts)]
        return [f.text for f in facts]
