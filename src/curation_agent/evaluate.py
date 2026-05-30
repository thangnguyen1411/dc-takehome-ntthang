"""Stage 5: score the agent against held-out ground-truth labels.

The agent never sees `ground_truth_signal`; this harness uses it to compute
verdict accuracy, a per-label confusion table, and a confidence-calibration
summary (is the agent more confident when it is right?).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import CurationResult, Task


@dataclass
class EvalReport:
    total: int = 0
    scored: int = 0
    # Exact 4-way category matches: predicted verdict == gold verdict.
    correct: int = 0
    # binary view: did the agent agree on supported-vs-flagged, ignoring which
    # specific issue category? Robust to fuzzy category boundaries.
    binary_correct: int = 0
    # Confusion matrix as confusion[gold][predicted] = count — shows not just
    # whether the agent erred but what it confused for what. default_factory
    # gives each instance its own dict (never share a mutable default).
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    # Mean confidence the agent reported on tasks it got RIGHT.
    mean_conf_correct: float = 0.0
    # Mean confidence on tasks it got WRONG. Pair with the line above for
    # calibration: a good agent is more confident when correct than when wrong.
    mean_conf_incorrect: float = 0.0
    # How many tasks actually triggered the refine loop (refinement_applied).
    refinements: int = 0

    @property
    def accuracy(self) -> float:
        # Exact-match accuracy rate; guard against divide-by-zero when nothing
        # was scored (e.g. an unlabeled CSV) by returning 0.0 instead of crashing.
        return self.correct / self.scored if self.scored else 0.0

    @property
    def binary_accuracy(self) -> float:
        # Supported-vs-flagged accuracy rate, same zero-guard as `accuracy`.
        return self.binary_correct / self.scored if self.scored else 0.0


class Evaluator:
    """Scores a batch of curation results against held-out ground-truth labels.

    Owns the results + the gold-label lookup, and exposes two behaviors:
    `evaluate()` (compute the metrics) and `render_markdown()` (format them).
    These used to be free functions; bundling them here gives the scoring logic
    a single, testable home and keeps the gold-label map encapsulated.
    """

    def __init__(self, results: list[CurationResult], tasks: list[Task]) -> None:
        self._results = results
        # task_id -> gold label. Built once; the agent never sees this.
        self._truth = {t.task_id: t.ground_truth_signal for t in tasks}

    @staticmethod
    def _initial_verdict(result: CurationResult) -> str:
        """The verdict on the original candidate answer (iteration 0)."""
        return result.iterations[0].verdict.verdict if result.iterations else result.final_verdict

    def evaluate(self) -> EvalReport:
        """Score every result and return a filled `EvalReport`."""
        report = EvalReport(total=len(self._results))
        conf_correct: list[float] = []
        conf_incorrect: list[float] = []

        for result in self._results:
            if result.refinement_applied:
                report.refinements += 1

            gold = self._truth.get(result.task_id)
            if not gold:
                continue
            report.scored += 1

            # Compare the agent's verdict on the ORIGINAL answer to the gold
            # label, since refinement deliberately changes the final verdict.
            predicted = self._initial_verdict(result)
            result.ground_truth_signal = gold
            matched = predicted == gold
            result.verdict_matches_ground_truth = matched

            report.confusion.setdefault(gold, {}).setdefault(predicted, 0)
            report.confusion[gold][predicted] += 1

            if (predicted == "supported") == (gold == "supported"):
                report.binary_correct += 1

            initial_conf = (
                result.iterations[0].verdict.confidence
                if result.iterations
                else result.answer_confidence
            )
            if matched:
                report.correct += 1
                conf_correct.append(initial_conf)
            else:
                conf_incorrect.append(initial_conf)

        report.mean_conf_correct = sum(conf_correct) / len(conf_correct) if conf_correct else 0.0
        report.mean_conf_incorrect = sum(conf_incorrect) / len(conf_incorrect) if conf_incorrect else 0.0
        return report

    def render_markdown(self, report: EvalReport) -> str:
        """Format an `EvalReport` (plus this run's results) as a markdown report."""
        lines: list[str] = ["# Curation Eval Report", ""]
        lines.append(f"- Tasks processed: **{report.total}**")
        lines.append(f"- Tasks scored against ground truth: **{report.scored}**")
        lines.append(f"- Verdict accuracy, exact 4-way category (on original answers): **{report.accuracy:.0%}**")
        lines.append(f"- Verdict accuracy, binary supported-vs-flagged: **{report.binary_accuracy:.0%}**")
        lines.append(f"- Refinements applied: **{report.refinements}**")
        lines.append(f"- Mean confidence when correct: **{report.mean_conf_correct:.2f}**")
        lines.append(f"- Mean confidence when incorrect: **{report.mean_conf_incorrect:.2f}**")
        lines.append("")

        lines.append("## Confusion (gold -> predicted)")
        lines.append("")
        lines.append("| gold label | predicted | count |")
        lines.append("|---|---|---|")
        for gold in sorted(report.confusion):
            for predicted, count in sorted(report.confusion[gold].items()):
                mark = "" if gold == predicted else "  ⚠"
                lines.append(f"| {gold} | {predicted}{mark} | {count} |")
        lines.append("")

        lines.append("## Per-task")
        lines.append("")
        lines.append("| task | gold | predicted | match | conf | refined |")
        lines.append("|---|---|---|---|---|---|")
        for r in self._results:
            pred = self._initial_verdict(r)
            match = "✓" if r.verdict_matches_ground_truth else ("✗" if r.ground_truth_signal else "—")
            lines.append(
                f"| {r.task_id} | {r.ground_truth_signal or '—'} | {pred} | {match} "
                f"| {r.answer_confidence:.2f} | {'yes' if r.refinement_applied else 'no'} |"
            )
        lines.append("")
        return "\n".join(lines)
