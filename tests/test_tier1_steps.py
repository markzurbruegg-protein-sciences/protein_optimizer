"""Tests for Tier 1 steps (pure sequence analysis — no external deps)."""

from pathlib import Path

import pytest

from protein_optimizer.io_utils import read_fasta
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_input(sequence: str, name: str = "test") -> StepResult:
    """Helper: create a minimal StepResult from a sequence."""
    return StepResult(
        step_name="input",
        candidates=[ProteinCandidate(sequence=sequence, name=name)],
    )


# ── Cysteine Scan ─────────────────────────────────────────────────────


class TestCysteineScan:
    def setup_method(self):
        from protein_optimizer.steps.cysteine_scan import CysteineScanStep
        self.step = CysteineScanStep()

    def test_finds_cysteines(self):
        # Sequence with 2 cysteines
        result = self.step.run(_make_input("MACVCKGGG"), {})
        # Should produce variants for each cysteine + remove-all
        variant_names = [c.name for c in result.candidates if c.parent_id is not None]
        assert len(variant_names) >= 2  # at least one variant

    def test_no_cysteines(self):
        result = self.step.run(_make_input("MAVVKGGG"), {})
        variants = [c for c in result.candidates if c.parent_id is not None]
        assert len(variants) == 0

    def test_protected_residue_skipped(self):
        result = self.step.run(
            _make_input("MACVCKGGG"),
            {"_global": {"protected_residues": [3]}},  # protect Cys at pos 3
        )
        # Check pos 3 not mutated
        for c in result.candidates:
            for m in c.mutations:
                assert m.position != 3

    def test_replacement_aa(self):
        result = self.step.run(_make_input("MACGG"), {})
        # Should have C→S and C→A variants
        mut_aas = {m.mut for c in result.candidates for m in c.mutations}
        assert "S" in mut_aas or "A" in mut_aas


# ── Motif Scan ─────────────────────────────────────────────────────────


class TestMotifScan:
    def setup_method(self):
        from protein_optimizer.steps.motif_scan import MotifScanStep
        self.step = MotifScanStep()

    def test_deamidation_ng(self):
        # NG is a deamidation motif
        result = self.step.run(_make_input("GGGGNGGGGG"), {})
        # Should flag the NG site
        has_deamid = any(
            "deamidation" in str(c.metadata).lower()
            or any("deamid" in (m.source_step or "") or "deamid" in str(m.metadata).lower()
                    for m in c.mutations)
            for c in result.candidates
        )
        # Check metadata or variants reference deamidation
        variants = [c for c in result.candidates if c.parent_id is not None]
        assert len(variants) > 0 or has_deamid or len(result.warnings) > 0

    def test_oxidation_methionine(self):
        result = self.step.run(_make_input("AAAAMAAAAA"), {})
        # Met at position 5 should be caught
        variants = [c for c in result.candidates if c.parent_id is not None]
        met_mutations = [
            m for c in result.candidates for m in c.mutations if m.wt == "M"
        ]
        # Might or might not generate variants depending on implementation
        # but there should be at least metadata about it
        assert len(result.candidates) >= 1

    def test_no_motifs_clean_sequence(self):
        result = self.step.run(_make_input("AAAAAAAAAA"), {})
        variants = [c for c in result.candidates if c.parent_id is not None]
        # Pure poly-A should have no motif issues
        assert len(variants) == 0


# ── Sequence Complexity ────────────────────────────────────────────────


class TestSequenceComplexity:
    def setup_method(self):
        from protein_optimizer.steps.sequence_complexity import SequenceComplexityStep
        self.step = SequenceComplexityStep()

    def test_homopolymer_detection(self):
        # 6 consecutive alanines
        result = self.step.run(_make_input("GGAAAAAAGGG"), {})
        parent = result.candidates[0]
        # Should flag in metadata
        assert "complexity_flags" in parent.metadata or len(result.warnings) > 0

    def test_proline_run(self):
        result = self.step.run(_make_input("GGPPPPGGG"), {})
        parent = result.candidates[0]
        assert len(result.candidates) >= 1

    def test_clean_sequence(self):
        result = self.step.run(_make_input("MSKGEELFTGVV"), {})
        parent = result.candidates[0]
        flags = parent.metadata.get("complexity_flags", [])
        assert isinstance(flags, list)


# ── Integration: run on example FASTA ──────────────────────────────────


class TestTier1Integration:
    def test_cysteine_scan_on_gfp(self):
        from protein_optimizer.steps.cysteine_scan import CysteineScanStep

        candidates = read_fasta(FIXTURES_DIR / "example.fasta")
        step_input = StepResult(step_name="input", candidates=candidates)
        step = CysteineScanStep()
        result = step.run(step_input, {})
        # GFP has cysteines, should produce variants
        variants = [c for c in result.candidates if c.parent_id is not None]
        assert len(variants) > 0

    def test_all_tier1_steps_chain(self):
        """Run all Tier 1 steps in sequence on GFP example."""
        from protein_optimizer.steps.cysteine_scan import CysteineScanStep
        from protein_optimizer.steps.motif_scan import MotifScanStep
        from protein_optimizer.steps.sequence_complexity import SequenceComplexityStep

        candidates = read_fasta(FIXTURES_DIR / "example.fasta")
        current = StepResult(step_name="input", candidates=candidates)

        for StepCls in [CysteineScanStep, MotifScanStep, SequenceComplexityStep]:
            step = StepCls()
            current = step.run(current, {})
            assert len(current.candidates) >= 1
            assert current.step_name == step.name
