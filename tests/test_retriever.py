"""Tests for the retrievers and their relevance threshold — no API key."""

from curation_agent.models import RetrievedSnippet
from curation_agent.retriever import KeywordRetriever

CORPUS = [
    "Activating EGFR mutations predict sensitivity to gefitinib in non-small cell lung cancer.",
    "The KRAS G12C variant is inhibited by sotorasib in lung adenocarcinoma.",
    "Statins inhibit HMG-CoA reductase and lower LDL cholesterol.",
    "Helicobacter pylori is the primary cause of peptic ulcer disease.",
]


def test_retrieves_the_relevant_snippet():
    r = KeywordRetriever(CORPUS, threshold=0.2)
    got = r.retrieve("Which mutation responds to gefitinib in lung cancer?")
    assert got, "expected at least one snippet"
    assert "EGFR" in got[0].text          # the EGFR snippet ranks first
    assert got[0].source == "corpus#0"


def test_ranks_best_match_first():
    r = KeywordRetriever(CORPUS, threshold=0.0)
    got = r.retrieve("How do statins affect LDL cholesterol?", k=4)
    assert "Statins" in got[0].text       # statin snippet is the top hit


def test_top_k_limits_results():
    r = KeywordRetriever(CORPUS, threshold=0.0)
    assert len(r.retrieve("cancer mutation lung", k=2)) == 2


def test_off_topic_query_returns_nothing():
    # the corpus has no useful knowledge for this -> threshold rejects the noise
    r = KeywordRetriever(CORPUS, threshold=0.25)
    assert r.retrieve("What is the boiling point of water at sea level?") == []


def test_threshold_gates_weak_matches():
    r_strict = KeywordRetriever(CORPUS, threshold=0.9)
    r_loose = KeywordRetriever(CORPUS, threshold=0.0)
    q = "Tell me something about pylori bacteria"
    # a loose threshold returns a (weak) match; a strict one rejects it
    assert r_loose.retrieve(q)
    assert r_strict.retrieve(q) == []


def test_scores_are_between_zero_and_one():
    r = KeywordRetriever(CORPUS, threshold=0.0)
    for snip in r.retrieve("EGFR gefitinib lung cancer", k=4):
        assert 0.0 <= snip.score <= 1.0


def test_empty_corpus_returns_nothing():
    r = KeywordRetriever([], threshold=0.0)
    assert r.retrieve("anything at all") == []


# --- pipeline integration: retrieval fills empty evidence before verifying ---

import dataclasses

from curation_agent.config import Config
from curation_agent.models import Task
from curation_agent.pipeline import Pipeline


class _AllGoodLLM:
    """Always 'supported'; covers verify/reflect/regenerate tool calls."""

    def complete_structured(self, *, system, prompt, schema, tool_name):
        if tool_name == "emit_reflection":
            return {"diagnosis": "d", "what_to_change": "c"}
        if tool_name == "emit_answer":
            return {"answer": "fixed"}
        return {
            "verdict": "supported", "confidence": 0.95, "question_quality_score": 0.9,
            "rubric": {"faithfulness": 1.0, "completeness": 1.0, "specificity": 1.0},
            "detected_issues": [], "rationale": "ok",
        }


def _evidenceless_task() -> Task:
    return Task(
        task_id="ret_x",
        question="Which mutation responds to gefitinib in lung cancer?",
        reference_context="",                       # no evidence — must be retrieved
        candidate_answer="EGFR mutations respond to gefitinib.",
    )


def _retrieve_config(tmp_path, corpus_lines, threshold=0.2) -> Config:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(corpus_lines), encoding="utf-8")
    return dataclasses.replace(
        Config(), retrieve=True, corpus_path=str(corpus), retrieval_threshold=threshold
    )


def test_pipeline_fills_empty_evidence_from_corpus(tmp_path):
    config = _retrieve_config(tmp_path, [
        "Activating EGFR mutations predict sensitivity to gefitinib in lung cancer.",
        "Statins inhibit HMG-CoA reductase and lower LDL cholesterol.",
    ])
    result = Pipeline.build(_AllGoodLLM(), config).run([_evidenceless_task()])[0]

    assert result.retrieved_evidence                                      # something was fetched
    assert any("EGFR" in s.text for s in result.retrieved_evidence)       # the right snippet
    assert "evidence_retrieved" in result.detected_issues
    assert "EGFR" in result.evidence[0]                                   # retrieved is the sole evidence entry


def test_pipeline_flags_retrieval_failed_when_corpus_irrelevant(tmp_path):
    config = _retrieve_config(tmp_path, [
        "Statins inhibit HMG-CoA reductase and lower LDL cholesterol.",
    ], threshold=0.25)
    result = Pipeline.build(_AllGoodLLM(), config).run([_evidenceless_task()])[0]

    assert result.retrieved_evidence == []
    assert "retrieval_failed" in result.detected_issues


def test_retrieval_left_off_by_default(tmp_path):
    # no --retrieve: a task with evidence is untouched, no retrieval tags
    task = Task(
        task_id="t", question="Q?", reference_context="Given evidence.",
        candidate_answer="A.",
    )
    result = Pipeline.build(_AllGoodLLM(), Config()).run([task])[0]
    assert result.retrieved_evidence == []
    assert "evidence_retrieved" not in result.detected_issues
    assert "retrieval_failed" not in result.detected_issues


def test_retrieval_keeps_given_and_adds_retrieved_separately(tmp_path):
    # retrieval runs for EVERY task; given evidence stays evidence[0] (unchanged),
    # retrieved knowledge is a SEPARATE entry — not concatenated into the given text
    config = _retrieve_config(tmp_path, [
        "Activating EGFR mutations predict sensitivity to gefitinib in lung cancer.",
    ])
    task = Task(
        task_id="t",
        question="Which mutation responds to gefitinib in lung cancer?",
        reference_context="Original given evidence about EGFR.",
        candidate_answer="EGFR mutations respond to gefitinib.",
    )
    result = Pipeline.build(_AllGoodLLM(), config).run([task])[0]

    assert result.retrieved_evidence                                  # retrieval ran despite given evidence
    assert result.evidence[0] == "Original given evidence about EGFR."  # given kept, unchanged, first
    assert "Activating EGFR" in result.evidence[1]                    # retrieved as a separate entry
    assert "evidence_retrieved" in result.detected_issues


def test_evidence_block_labels_primary_and_supplementary():
    # the EVIDENCE text the LLM receives labels given as PRIMARY, retrieved as SUPPLEMENTARY
    t = Task(task_id="t", question="q", reference_context="Given.", candidate_answer="a",
             relevant_knowledge="Retrieved.")
    block = t.evidence_block()
    assert "PRIMARY EVIDENCE" in block and "SUPPLEMENTARY RETRIEVED CONTEXT" in block
    assert block.index("Given.") < block.index("Retrieved.")        # primary first


def test_evidence_block_single_source_has_one_plain_section():
    # one source -> a single EVIDENCE section, no PRIMARY/SUPPLEMENTARY split
    given_only = Task(task_id="t", question="q", reference_context="Given.", candidate_answer="a")
    block = given_only.evidence_block()
    assert "Given." in block
    assert "PRIMARY EVIDENCE" not in block and "SUPPLEMENTARY" not in block

    retrieved_only = Task(task_id="t", question="q", reference_context="", candidate_answer="a",
                          relevant_knowledge="Retrieved.")
    block = retrieved_only.evidence_block()
    assert "Retrieved." in block
    assert "PRIMARY EVIDENCE" not in block and "SUPPLEMENTARY" not in block


def test_irrelevant_corpus_with_given_evidence_is_not_flagged_failed(tmp_path):
    # a task that HAS evidence isn't tagged retrieval_failed just because the
    # corpus had nothing to add
    config = _retrieve_config(tmp_path, [
        "Statins inhibit HMG-CoA reductase and lower LDL cholesterol.",
    ], threshold=0.25)
    task = Task(
        task_id="t",
        question="Which mutation responds to gefitinib in lung cancer?",
        reference_context="Given EGFR evidence.",
        candidate_answer="A.",
    )
    result = Pipeline.build(_AllGoodLLM(), config).run([task])[0]
    assert result.retrieved_evidence == []
    assert "retrieval_failed" not in result.detected_issues
    assert result.evidence[0] == "Given EGFR evidence."              # untouched


# --- PubMedRetriever (mocked HTTP — no real network in CI) ---

import json as _json
from curation_agent.retriever import CompositeRetriever, PubMedRetriever, _overlap_score

_ESEARCH_JSON = _json.dumps({"esearchresult": {"idlist": ["111", "222"]}}).encode()
_EFETCH_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle><MedlineCitation>
    <PMID>111</PMID>
    <Article><ArticleTitle>EGFR mutations and gefitinib response</ArticleTitle>
      <Abstract><AbstractText>EGFR tyrosine kinase mutations predict gefitinib sensitivity in lung cancer.</AbstractText></Abstract>
    </Article>
  </MedlineCitation></PubmedArticle>
  <PubmedArticle><MedlineCitation>
    <PMID>222</PMID>
    <Article><ArticleTitle>Unrelated cardiology study</ArticleTitle>
      <Abstract><AbstractText>Beta blockers reduce mortality in heart failure.</AbstractText></Abstract>
    </Article>
  </MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def _patch_pubmed(monkeypatch, esearch=_ESEARCH_JSON, efetch=_EFETCH_XML, fail=False):
    """Stub PubMedRetriever._get so no real HTTP happens."""
    def fake_get(self, url, params):
        if fail:
            raise OSError("network down")
        return esearch if "esearch" in url else efetch
    monkeypatch.setattr(PubMedRetriever, "_get", fake_get)


def test_pubmed_returns_scored_abstracts(monkeypatch):
    _patch_pubmed(monkeypatch)
    got = PubMedRetriever(threshold=0.1).retrieve(
        "EGFR tyrosine kinase mutations gefitinib lung cancer", k=2
    )
    assert got, "expected PubMed snippets"
    assert got[0].source.startswith("pubmed:")
    assert "EGFR" in got[0].text                       # relevant abstract ranks first
    assert all(0.0 <= s.score <= 1.0 for s in got)


def test_pubmed_threshold_filters_irrelevant(monkeypatch):
    _patch_pubmed(monkeypatch)
    # high threshold + a query that only matches one abstract -> the off-topic one is dropped
    got = PubMedRetriever(threshold=0.6).retrieve("EGFR gefitinib lung cancer mutations", k=5)
    assert all("heart failure" not in s.text for s in got)


def test_pubmed_degrades_to_empty_on_network_error(monkeypatch):
    _patch_pubmed(monkeypatch, fail=True)
    assert PubMedRetriever().retrieve("anything") == []   # never raises


def test_pubmed_empty_idlist_returns_nothing(monkeypatch):
    _patch_pubmed(monkeypatch, esearch=_json.dumps({"esearchresult": {"idlist": []}}).encode())
    assert PubMedRetriever().retrieve("nothing matches") == []


# --- CompositeRetriever ---

class _Stub:
    def __init__(self, snippets):
        self._snippets = snippets
    def retrieve(self, query, k=2):
        return list(self._snippets)


def _snip(text, score, source):
    return RetrievedSnippet(text=text, score=score, source=source)


def test_composite_merges_and_ranks_by_score():
    a = _Stub([_snip("corpus hit", 0.5, "corpus#0")])
    b = _Stub([_snip("pubmed hit", 0.8, "pubmed:1")])
    got = CompositeRetriever([a, b]).retrieve("q", k=2)
    assert [s.text for s in got] == ["pubmed hit", "corpus hit"]   # higher score first


def test_composite_dedupes_identical_text_keeping_higher_score():
    a = _Stub([_snip("same fact", 0.4, "corpus#0")])
    b = _Stub([_snip("same fact", 0.9, "pubmed:1")])
    got = CompositeRetriever([a, b]).retrieve("q", k=5)
    assert len(got) == 1
    assert got[0].score == 0.9 and got[0].source == "pubmed:1"


def test_composite_top_k_caps_merged_results():
    a = _Stub([_snip("a", 0.5, "c#0"), _snip("b", 0.4, "c#1")])
    b = _Stub([_snip("c", 0.9, "p:1")])
    assert len(CompositeRetriever([a, b]).retrieve("q", k=2)) == 2


def test_composite_tolerates_an_empty_member():
    got = CompositeRetriever([_Stub([]), _Stub([_snip("only", 0.7, "p:1")])]).retrieve("q")
    assert [s.text for s in got] == ["only"]


def test_overlap_score_bounds():
    assert _overlap_score(set(), "anything") == 0.0
    assert _overlap_score({"egfr", "gefitinib"}, "egfr drives gefitinib response") == 1.0
    assert _overlap_score({"egfr", "statin"}, "egfr only here") == 0.5
