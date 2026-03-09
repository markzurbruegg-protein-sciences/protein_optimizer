"""Tests for core data models: Mutation, ProteinCandidate, StepResult."""

import json
import tempfile
from pathlib import Path

import pytest

from protein_optimizer.models import Mutation, ProteinCandidate, StepResult


# ── Mutation ─────────────────────────────────────────────────────────────


class TestMutation:
    def test_label(self):
        m = Mutation(position=45, wt="C", mut="S", source_step="test")
        assert m.label == "C45S"

    def test_roundtrip_dict(self):
        m = Mutation(position=10, wt="A", mut="V", source_step="test", score=0.5)
        d = m.to_dict()
        m2 = Mutation.from_dict(d)
        assert m2.position == 10
        assert m2.wt == "A"
        assert m2.mut == "V"
        assert m2.score == 0.5


# ── ProteinCandidate ─────────────────────────────────────────────────────


GFP_SEQ = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN"
    "RIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHY"
    "QQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


class TestProteinCandidate:
    def test_creation(self):
        pc = ProteinCandidate(sequence=GFP_SEQ, name="GFP")
        assert pc.name == "GFP"
        assert len(pc.sequence) == len(GFP_SEQ)
        assert pc.parent_id is None
        assert pc.num_mutations == 0

    def test_apply_mutation(self):
        pc = ProteinCandidate(sequence=GFP_SEQ, name="GFP")
        mut = Mutation(position=48, wt="C", mut="S", source_step="test")
        variant = pc.apply_mutation(mut)
        assert variant.sequence[47] == "S"  # 0-indexed
        assert variant.parent_id == pc.candidate_id
        assert len(variant.mutations) == 1

    def test_apply_mutation_wrong_wt_raises(self):
        pc = ProteinCandidate(sequence="MACGG", name="test")
        mut = Mutation(position=2, wt="X", mut="V", source_step="test")
        with pytest.raises(ValueError, match="Expected X"):
            pc.apply_mutation(mut)

    def test_apply_mutation_out_of_range(self):
        pc = ProteinCandidate(sequence="MACGG", name="test")
        mut = Mutation(position=100, wt="A", mut="V", source_step="test")
        with pytest.raises(ValueError, match="out of range"):
            pc.apply_mutation(mut)

    def test_apply_mutations_multiple(self):
        pc = ProteinCandidate(sequence="MACGG", name="test")
        m1 = Mutation(position=1, wt="M", mut="L", source_step="test")
        m2 = Mutation(position=3, wt="C", mut="S", source_step="test")
        variant = pc.apply_mutations([m1, m2])
        assert variant.sequence == "LASGG"
        assert len(variant.mutations) == 2

    def test_unique_candidate_ids(self):
        pc1 = ProteinCandidate(sequence="AAA", name="a")
        pc2 = ProteinCandidate(sequence="AAA", name="b")
        assert pc1.candidate_id != pc2.candidate_id

    def test_roundtrip_dict(self):
        pc = ProteinCandidate(
            sequence="MACGG", name="test",
            scores={"e1": 0.5, "ddg": -1.2},
        )
        d = pc.to_dict()
        pc2 = ProteinCandidate.from_dict(d)
        assert pc2.sequence == "MACGG"
        assert pc2.scores["e1"] == 0.5


# ── StepResult ───────────────────────────────────────────────────────────


class TestStepResult:
    def test_creation(self):
        pc = ProteinCandidate(sequence="MACGG", name="test")
        sr = StepResult(step_name="test_step", candidates=[pc])
        assert sr.step_name == "test_step"
        assert len(sr.candidates) == 1

    def test_wild_type_property(self):
        pc = ProteinCandidate(sequence="MACGG", name="wt")
        sr = StepResult(step_name="test", candidates=[pc])
        assert sr.wild_type is pc

    def test_wild_type_empty(self):
        sr = StepResult(step_name="test", candidates=[])
        assert sr.wild_type is None

    def test_top_candidates(self):
        candidates = []
        for i in range(5):
            pc = ProteinCandidate(sequence="MACGG", name=f"v{i}")
            pc.scores["test_score"] = float(i)
            candidates.append(pc)
        sr = StepResult(step_name="test", candidates=candidates)
        top3 = sr.top_candidates("test_score", n=3, ascending=True)
        assert len(top3) == 3
        assert top3[0].scores["test_score"] == 0.0

    def test_save_and_load(self):
        pc = ProteinCandidate(
            sequence="MACGG", name="test",
            scores={"score1": 1.5},
        )
        mut = Mutation(position=3, wt="C", mut="S", source_step="test")
        variant = pc.apply_mutation(mut)

        sr = StepResult(
            step_name="test_step",
            candidates=[pc, variant],
            warnings=["test warning"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            sr.save(path)

            assert path.exists()

            sr2 = StepResult.load(path)
            assert sr2.step_name == "test_step"
            assert len(sr2.candidates) == 2
            assert sr2.candidates[1].mutations[0].label == "C3S"
            assert sr2.warnings == ["test warning"]

    def test_get_candidate(self):
        pc1 = ProteinCandidate(sequence="AAA", name="a")
        pc2 = ProteinCandidate(sequence="BBB", name="b")
        sr = StepResult(step_name="test", candidates=[pc1, pc2])
        assert sr.get_candidate(pc1.candidate_id) is pc1
        assert sr.get_candidate("nonexistent") is None
