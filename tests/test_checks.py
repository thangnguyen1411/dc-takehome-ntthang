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
