"""Tests for scoring and filtering modules."""

import pytest

from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.scoring.aggregator import aggregate_scores, rank_candidates
from protein_optimizer.scoring.filters import (
    FilterConfig,
    apply_diversity_filter,
    filter_candidates,
)


def _make_scored_result() -> StepResult:
    """Create a StepResult with scored variants for testing."""
    parent = ProteinCandidate(sequence="MACVGKLSDE" * 5, name="parent")
    parent.scores["e1_fitness"] = -0.5

    variants = []
    for i in range(5):
        mut = Mutation(position=i + 1, wt=parent.sequence[i], mut="G", source_step="test")
        try:
            v = parent.apply_mutation(mut)
        except ValueError:
            v = ProteinCandidate(
                sequence=parent.sequence[:i] + "G" + parent.sequence[i+1:],
                name=f"v{i}",
                parent_id=parent.candidate_id,
                mutations=[mut],
            )
        v.scores["e1_fitness"] = -0.3 + i * 0.1
        v.scores["esm1v_delta"] = -0.1 + i * 0.05
        v.scores["consensus_score"] = 0.5 + i * 0.1
        variants.append(v)

    return StepResult(
        step_name="test",
        candidates=[parent] + variants,
    )


class TestAggregateScores:
    def test_adds_composite_score(self):
        result = _make_scored_result()
        aggregate_scores(result)
        for c in result.candidates:
            assert "composite_score" in c.scores

    def test_custom_weights(self):
        result = _make_scored_result()
        aggregate_scores(result, weights={"e1_fitness": 1.0})
        assert "composite_score" in result.candidates[0].scores

    def test_no_normalize(self):
        result = _make_scored_result()
        aggregate_scores(result, normalize=False)
        assert "composite_score" in result.candidates[0].scores

    def test_empty_result(self):
        result = StepResult(step_name="test", candidates=[])
        aggregate_scores(result)
        # Should not raise


class TestRankCandidates:
    def test_rank_by_composite(self):
        result = _make_scored_result()
        aggregate_scores(result)
        ranked = rank_candidates(result, "composite_score")
        # Should be sorted descending
        for i in range(len(ranked) - 1):
            if "composite_score" in ranked[i].scores and "composite_score" in ranked[i+1].scores:
                assert ranked[i].scores["composite_score"] >= ranked[i+1].scores["composite_score"]

    def test_rank_ascending(self):
        result = _make_scored_result()
        aggregate_scores(result)
        ranked = rank_candidates(result, "composite_score", ascending=True)
        for i in range(len(ranked) - 1):
            if "composite_score" in ranked[i].scores and "composite_score" in ranked[i+1].scores:
                assert ranked[i].scores["composite_score"] <= ranked[i+1].scores["composite_score"]


class TestFilterCandidates:
    def test_basic_filter(self):
        result = _make_scored_result()
        passed, failed = filter_candidates(result)
        # Parent should always pass
        assert any(c.parent_id is None for c in passed)

    def test_max_mutations_filter(self):
        parent = ProteinCandidate(sequence="MACVGKLSDE", name="p")
        variant = ProteinCandidate(
            sequence="GGGGGGGGGD",
            name="v",
            parent_id=parent.candidate_id,
            mutations=[Mutation(position=i+1, wt="X", mut="G", source_step="t") for i in range(9)],
        )
        result = StepResult(step_name="test", candidates=[parent, variant])
        fc = FilterConfig(max_mutations=5)
        passed, failed = filter_candidates(result, fc)
        assert len(failed) == 1
        assert "too many mutations" in failed[0].metadata.get("filter_fail_reasons", [])[0]

    def test_no_new_cysteines(self):
        parent = ProteinCandidate(sequence="MACGG", name="p")
        variant = ProteinCandidate(
            sequence="MCCGG",
            name="v",
            parent_id=parent.candidate_id,
            mutations=[Mutation(position=2, wt="A", mut="C", source_step="t")],
        )
        result = StepResult(step_name="test", candidates=[parent, variant])
        passed, failed = filter_candidates(result, FilterConfig(forbid_new_cysteines=True))
        assert len(failed) == 1

    def test_custom_config_dict(self):
        result = _make_scored_result()
        passed, failed = filter_candidates(result, {"max_mutations": 100})
        assert len(passed) == len(result.candidates)


class TestDiversityFilter:
    def test_removes_duplicates(self):
        candidates = [
            ProteinCandidate(sequence="MACGG", name=f"v{i}")
            for i in range(5)
        ]
        selected = apply_diversity_filter(candidates, min_sequence_distance=1)
        assert len(selected) == 1  # all identical

    def test_keeps_diverse(self):
        candidates = [
            ProteinCandidate(sequence="MACGG", name="v1"),
            ProteinCandidate(sequence="MASGG", name="v2"),
            ProteinCandidate(sequence="MALGG", name="v3"),
        ]
        selected = apply_diversity_filter(candidates, min_sequence_distance=1)
        assert len(selected) == 3

    def test_max_candidates(self):
        candidates = [
            ProteinCandidate(
                sequence=f"M{'ACDEFGHIKL'[i % 10]}CGG", name=f"v{i}"
            )
            for i in range(20)
        ]
        selected = apply_diversity_filter(candidates, min_sequence_distance=1, max_candidates=5)
        assert len(selected) <= 5
