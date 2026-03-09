"""Tests for the HTML report generator."""

import tempfile
from pathlib import Path

from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.reporting.html_report import generate_html_report


def _make_results() -> dict[str, StepResult]:
    parent = ProteinCandidate(sequence="MACVGKLSDE", name="parent")
    parent.scores["e1_fitness"] = -0.5

    mut = Mutation(position=3, wt="C", mut="S", source_step="cysteine_scan")
    variant = parent.apply_mutation(mut)
    variant.scores["e1_fitness"] = -0.3
    variant.scores["composite_score"] = 0.7

    return {
        "cysteine_scan": StepResult(
            step_name="cysteine_scan",
            candidates=[parent, variant],
            warnings=["Test warning"],
        ),
    }


class TestHTMLReport:
    def test_generates_html(self):
        results = _make_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            out = generate_html_report(results, path)
            assert out.exists()
            content = out.read_text()
            assert "<!DOCTYPE html>" in content
            assert "parent" in content
            assert "C3S" in content

    def test_empty_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            out = generate_html_report({}, path)
            assert out.exists()
