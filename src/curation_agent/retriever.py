"""Stage 0 (optional): retrieve evidence for a task before verification.

When retrieval is enabled, a `Retriever` fetches the most relevant snippets for
each task and stores them in the task's `relevant_knowledge` (its given
`reference_context` is left untouched). If a task had no given evidence the
retrieved snippets become its sole evidence (RAG); otherwise they serve as
supplementary supporting context.

Three implementations, all behind the `Retriever` protocol:
- `KeywordRetriever` — IDF-weighted term overlap over a local corpus; no
  dependencies, fully offline.
- `PubMedRetriever` — live retrieval of abstracts from NCBI E-utilities (stdlib
  HTTP); network-dependent, degrades to nothing on any error.
- `CompositeRetriever` — fans out to several retrievers and merges their results,
  so a local corpus and PubMed can be used together.

All apply a **relevance threshold**: if the best match doesn't clear it, retrieval
returns nothing rather than handing back an irrelevant snippet — so "no useful
knowledge for this question" degrades to an honest miss, not a misleading verdict.
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


def _overlap_score(query_terms: set[str], text: str) -> float:
    """Fraction of the query's content terms that appear in `text` → [0, 1].

    A corpus-independent relevance score, used for sources like PubMed where
    there's no corpus to compute IDF over.
    """
    if not query_terms:
        return 0.0
    return len(query_terms & _terms(text)) / len(query_terms)


class Retriever(Protocol):
    """Fetches the most relevant snippets for a query.

    A Protocol: any class with a matching `retrieve` method satisfies it
    structurally — `KeywordRetriever`, `PubMedRetriever`, `CompositeRetriever`,
    or a test double.
    """

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


class PubMedRetriever:
    """Live retrieval of abstracts from NCBI E-utilities (esearch -> efetch).

    Uses only the standard library (urllib + xml.etree). Network-dependent and
    deliberately resilient: any error (no network, rate limit, malformed
    response) degrades to an empty result rather than raising, so an offline run
    or a flaky API never breaks curation. Retrieved abstracts are scored by query
    term overlap and gated by the same relevance threshold as the corpus retriever.

    NCBI asks unauthenticated callers to identify themselves (tool + email) and
    stay under ~3 requests/second; an optional api_key raises that limit.
    """

    _ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    _EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(
        self,
        threshold: float = 0.25,
        *,
        email: str = "curation-agent@example.com",
        tool: str = "curation_agent",
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._threshold = threshold
        self._email = email
        self._tool = tool
        self._api_key = api_key
        self._timeout = timeout

    def retrieve(self, query: str, k: int = 2) -> list[RetrievedSnippet]:
        try:
            pmids = self._esearch(query, k)
            if not pmids:
                return []
            articles = self._efetch(pmids)  # {pmid: abstract_text}
        except Exception:
            # Network/parse/rate-limit failure: an honest empty result, never a crash.
            return []

        query_terms = _terms(query)
        scored = []
        for pmid in pmids:  # preserve PubMed's relevance order as the tiebreak
            text = articles.get(pmid, "").strip()
            if not text:
                continue
            score = _overlap_score(query_terms, text)
            if score >= self._threshold:
                scored.append(RetrievedSnippet(text=text, score=round(score, 4), source=f"pubmed:{pmid}"))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    def _params(self, extra: dict[str, str]) -> str:
        from urllib.parse import urlencode

        params = {"db": "pubmed", "tool": self._tool, "email": self._email, **extra}
        if self._api_key:
            params["api_key"] = self._api_key
        return urlencode(params)

    def _get(self, url: str, params: str) -> bytes:
        from urllib.request import urlopen

        with urlopen(f"{url}?{params}", timeout=self._timeout) as resp:
            return resp.read()

    def _esearch(self, query: str, k: int) -> list[str]:
        import json

        raw = self._get(self._ESEARCH, self._params(
            {"term": query, "retmax": str(k), "retmode": "json", "sort": "relevance"}
        ))
        data = json.loads(raw)
        return list(data.get("esearchresult", {}).get("idlist", []))

    def _efetch(self, pmids: list[str]) -> dict[str, str]:
        import xml.etree.ElementTree as ET

        raw = self._get(self._EFETCH, self._params(
            {"id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
        ))
        root = ET.fromstring(raw)
        out: dict[str, str] = {}
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            if pmid_el is None or not pmid_el.text:
                continue
            title = (article.findtext(".//ArticleTitle") or "").strip()
            # An abstract can be split into labeled <AbstractText> sections.
            parts = [(el.text or "").strip() for el in article.findall(".//AbstractText")]
            abstract = " ".join(p for p in parts if p)
            out[pmid_el.text.strip()] = f"{title} {abstract}".strip()
        return out


class CompositeRetriever:
    """Fans out a query to several retrievers and merges their snippets.

    Lets a local corpus and PubMed (or any future sources) be used together: each
    member retrieves independently, the results are pooled, de-duplicated by text,
    sorted by score, and the top-k returned. A member that returns nothing (or
    fails gracefully) simply contributes nothing — the others still work.
    """

    def __init__(self, retrievers: list[Retriever]) -> None:
        if not retrievers:
            raise ValueError("CompositeRetriever needs at least one retriever")
        self._retrievers = retrievers

    def retrieve(self, query: str, k: int = 2) -> list[RetrievedSnippet]:
        merged: dict[str, RetrievedSnippet] = {}
        for retriever in self._retrievers:
            for snip in retriever.retrieve(query, k=k):
                # De-dup by snippet text; keep the higher-scoring source.
                existing = merged.get(snip.text)
                if existing is None or snip.score > existing.score:
                    merged[snip.text] = snip
        ranked = sorted(merged.values(), key=lambda s: s.score, reverse=True)
        return ranked[:k]
