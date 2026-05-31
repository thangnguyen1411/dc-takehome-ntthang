from curation_agent.checks import DeterministicChecker
from curation_agent.models import Task


def _task(**kw) -> Task:
    base = dict(
        task_id="t",
        question="Which gene drives resistance?",
        reference_context="Gene X drives resistance.",
        candidate_answer="Gene X.",
    )
    base.update(kw)
    return Task(**base)


def test_normalized_question_strips_case_and_punctuation():
    assert DeterministicChecker.normalized_question("Which  Gene, drives?") == "which gene drives"


def test_schema_issues_flags_empty_evidence():
    issues = DeterministicChecker.schema_issues(_task(reference_context=""))
    assert "missing_evidence" in issues


def test_retrieved_knowledge_counts_as_evidence():
    # a task with no given evidence but with retrieved relevant_knowledge is NOT
    # "missing_evidence" — retrieval supplied something to judge against
    t = _task(reference_context="", relevant_knowledge="Retrieved fact.")
    assert "missing_evidence" not in DeterministicChecker.schema_issues(t)


def test_missing_evidence_when_both_sources_empty():
    t = _task(reference_context="", relevant_knowledge="")
    assert "missing_evidence" in DeterministicChecker.schema_issues(t)


def test_schema_issues_clean_task_has_none():
    assert DeterministicChecker.schema_issues(_task()) == []


def test_clean_task_has_no_duplicate_issue():
    checker = DeterministicChecker([_task(task_id="solo", question="Unique question?")])
    assert "duplicate_task" not in checker.issues_for(_task(task_id="solo", question="Unique question?"))


def test_deterministic_issues_reports_duplicate_task():
    tasks = [
        _task(task_id="a", question="Repeated?"),
        _task(task_id="b", question="Repeated?"),
    ]
    checker = DeterministicChecker(tasks)
    assert "duplicate_task" in checker.issues_for(tasks[0])


def test_citation_absent_from_evidence_is_flagged():
    t = _task(
        reference_context="Gene X drives resistance.",
        candidate_answer="Gene X drives resistance [12].",
    )
    assert DeterministicChecker.citation_issues(t) == ["hallucinated_citation"]


def test_author_year_citation_absent_from_evidence_is_flagged():
    t = _task(
        reference_context="The pathway was upregulated.",
        candidate_answer="The pathway was upregulated (Smith et al., 2020).",
    )
    assert "hallucinated_citation" in DeterministicChecker.citation_issues(t)


def test_citation_present_in_evidence_is_not_flagged():
    t = _task(
        reference_context="As shown in [12], Gene X drives resistance.",
        candidate_answer="Gene X drives resistance [12].",
    )
    assert DeterministicChecker.citation_issues(t) == []


def test_no_citation_is_not_flagged():
    t = _task(
        reference_context="Gene X drives resistance.",
        candidate_answer="Gene X drives resistance.",
    )
    assert DeterministicChecker.citation_issues(t) == []


# --- one case per citation pattern (all absent from evidence -> flagged) ---

def test_numeric_range_citation_flagged():
    t = _task(reference_context="No refs here.", candidate_answer="Shown previously [5-7].")
    assert DeterministicChecker.citation_issues(t) == ["hallucinated_citation"]


def test_numeric_list_citation_flagged():
    t = _task(reference_context="No refs here.", candidate_answer="Reported in [3, 4].")
    assert DeterministicChecker.citation_issues(t) == ["hallucinated_citation"]


def test_single_author_year_citation_flagged():
    t = _task(reference_context="No refs.", candidate_answer="It was elevated (Lee 2019).")
    assert DeterministicChecker.citation_issues(t) == ["hallucinated_citation"]


def test_two_author_year_citation_flagged():
    t = _task(reference_context="No refs.", candidate_answer="Confirmed (Smith and Jones, 2021).")
    assert DeterministicChecker.citation_issues(t) == ["hallucinated_citation"]


def test_bare_et_al_citation_flagged():
    t = _task(reference_context="No refs.", candidate_answer="As Park et al. demonstrated.")
    assert DeterministicChecker.citation_issues(t) == ["hallucinated_citation"]


def test_doi_citation_flagged():
    t = _task(reference_context="No refs.", candidate_answer="See 10.1038/s41586-020-12345.")
    assert DeterministicChecker.citation_issues(t) == ["hallucinated_citation"]


def test_pmid_citation_flagged():
    t = _task(reference_context="No refs.", candidate_answer="Indexed as PMID: 31234567.")
    assert DeterministicChecker.citation_issues(t) == ["hallucinated_citation"]


# present-in-evidence variants are NOT flagged ---

def test_doi_present_in_evidence_is_not_flagged():
    t = _task(
        reference_context="Methods follow 10.1038/s41586-020-12345.",
        candidate_answer="We used the protocol 10.1038/s41586-020-12345.",
    )
    assert DeterministicChecker.citation_issues(t) == []


def test_author_year_present_in_evidence_is_not_flagged():
    t = _task(
        reference_context="The original report (Lee 2019) described the assay.",
        candidate_answer="Per (Lee 2019), the assay is validated.",
    )
    assert DeterministicChecker.citation_issues(t) == []


def test_citation_match_is_case_insensitive():
    # answer cites 'PMID: 31234567'; evidence has it lowercased — still a match
    t = _task(
        reference_context="see pmid: 31234567 for details.",
        candidate_answer="Indexed as PMID: 31234567.",
    )
    assert DeterministicChecker.citation_issues(t) == []


# --- mixed / edge cases ---

def test_one_present_one_absent_is_flagged():
    t = _task(
        reference_context="Established in [1].",
        candidate_answer="Established in [1], extended in [2].",
    )
    assert DeterministicChecker.citation_issues(t) == ["hallucinated_citation"]


def test_all_present_citations_not_flagged():
    t = _task(
        reference_context="Shown in [1] and [2].",
        candidate_answer="Shown in [1] and [2].",
    )
    assert DeterministicChecker.citation_issues(t) == []


def test_plain_numbers_are_not_treated_as_citations():
    # bare numbers / ranges without brackets must not trip the bracket pattern
    t = _task(
        reference_context="Levels rose 5-7 fold in 2020.",
        candidate_answer="Levels rose 5-7 fold in 2020.",
    )
    assert DeterministicChecker.citation_issues(t) == []


def test_citation_issue_flows_through_issues_for():
    t = _task(candidate_answer="Gene X drives resistance [99].")
    issues = DeterministicChecker([t]).issues_for(t)
    assert "hallucinated_citation" in issues
