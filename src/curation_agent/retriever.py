"""Stage 0 (optional): retrieve evidence from a knowledge corpus.

When retrieval is enabled, a `Retriever` fetches the most relevant snippets from
the corpus for every task and stores them in the task's `relevant_knowledge`
(its given `reference_context` is left untouched). If a task had no given
evidence, the retrieved snippets become its sole evidence — turning the pipeline
from "evidence is given" into "evidence is retrieved" (RAG); if it already had
evidence, the snippets serve as supplementary supporting context.

`KeywordRetriever` ranks corpus snippets by IDF-weighted term overlap with the
question (no dependencies, fully offline). Crucially it applies a **relevance
threshold**: if the best match doesn't clear it, retrieval returns nothing rather
than handing back an irrelevant snippet — so "the corpus has no useful knowledge
for this question" degrades to an honest miss instead of a misleading verdict.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from .models import RetrievedSnippet

_STOPWORDS = {
    "the", "a", "an", "in", "of", "and", "or", "to", "with", "was", "were", "is",
    "are", "for", "on", "by", "at", "as", "that", "this", "it", "its", "be", "what",
    "which", "how", "does", "do", "did", "from", "into", "than", "then", "they",
}


def _terms(text: str) -> set[str]:
    """Content terms: lowercased alphanumerics, length > 2, minus stopwords."""
    words = re.findall(r"[a-z0-9-]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 2) -> list[RetrievedSnippet]:
        ...


class KeywordRetriever:
    """IDF-weighted term-overlap retriever over a fixed corpus, with a relevance
    threshold so off-topic queries return nothing instead of noise."""

    def __init__(self, corpus: list[str], threshold: float = 0.25) -> None:
        self._docs = corpus
        self._doc_terms = [_terms(d) for d in corpus]
        self._threshold = threshold

        # Inverse document frequency: rarer terms across the corpus weigh more,
        # so matching a specific term (e.g. "osimertinib") counts far more than a
        # common one. A query term absent from the corpus gets the max idf.
        n = max(len(corpus), 1)
        df: dict[str, int] = {}
        for terms in self._doc_terms:
            for t in terms:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1 + n / c) for t, c in df.items()}
        self._default_idf = math.log(1 + n / 1)  # treat unseen terms as rare

    def _score(self, query_terms: set[str], doc_terms: set[str]) -> float:
        """Fraction of the query's total IDF weight covered by the document → [0,1]."""
        total = sum(self._idf.get(t, self._default_idf) for t in query_terms)
        if total == 0:
            return 0.0
        covered = sum(self._idf.get(t, self._default_idf) for t in query_terms if t in doc_terms)
        return covered / total

    def retrieve(self, query: str, k: int = 2) -> list[RetrievedSnippet]:
        query_terms = _terms(query)
        scored = sorted(
            ((self._score(query_terms, dt), i) for i, dt in enumerate(self._doc_terms)),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            RetrievedSnippet(text=self._docs[i], score=round(score, 4), source=f"corpus#{i}")
            for score, i in scored[:k] if score >= self._threshold
        ]
