"""Calibration-display tests: an empty confidence bucket must render as n/a,
not as a misleading 0.00."""

from curation_agent.evaluate import EvalReport, Evaluator


def _render(report: EvalReport) -> str:
    # render_markdown only reads self._results for the rubric section; an empty
    # list is fine for exercising the calibration lines.
    evaluator = Evaluator.__new__(Evaluator)
    evaluator._results = []
    return evaluator.render_markdown(report)


def test_fmt_conf_empty_bucket_is_na():
    assert Evaluator._fmt_conf(0.0, 0) == "n/a (no such cases)"


def test_fmt_conf_with_samples_shows_value_and_count():
    assert Evaluator._fmt_conf(0.89, 3) == "0.89 (n=3)"


def test_all_correct_renders_incorrect_as_na():
    # 2/2 correct -> no incorrect cases -> the incorrect line must be n/a, and
    # the correct line must show the real mean, not a bare 0.00.
    report = EvalReport(
        total=2, scored=2, correct=2, binary_correct=2,
        mean_conf_correct=0.89, mean_conf_incorrect=0.0,
    )
    md = _render(report)
    assert "- Mean confidence when correct: **0.89 (n=2)**" in md
    assert "- Mean confidence when incorrect: **n/a (no such cases)**" in md


def test_mixed_results_show_both_means():
    report = EvalReport(
        total=3, scored=3, correct=2, binary_correct=2,
        mean_conf_correct=0.90, mean_conf_incorrect=0.70,
    )
    md = _render(report)
    assert "- Mean confidence when correct: **0.90 (n=2)**" in md
    assert "- Mean confidence when incorrect: **0.70 (n=1)**" in md
