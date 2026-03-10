"""Tests for the protein_characterization step."""

import pytest

from protein_optimizer.models import ProteinCandidate, StepResult


def _make_input(sequence: str, name: str = "test") -> StepResult:
    """Helper: create a minimal StepResult from a sequence."""
    return StepResult(
        step_name="input",
        candidates=[ProteinCandidate(sequence=sequence, name=name)],
    )


# A realistic test sequence (insulin B-chain-like)
_SHORT_SEQ = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"

# A longer test sequence with known properties
_LONG_SEQ = (
    "MLRLPNPMVGFGVPNEPAIVTNRTLPEAAVGPPYFYYENVALAPRGVWQTISRFLYDVEP"
    "EFVDSKYFCAAARKRGYVHNLPIQNRYPLLPLPPYTIHEALPLTKKWWPSWDTRTKLNCL"
    "QTCVGSAKLTDRIRKALEDYEGDPPLTVQKFVLDECRKWNLVWVGRNKVAPLEPDEVEM"
    "LLGFPRNHTRGGGLSRTDRYKSLGASFQVDTVAYHLSVLKDMFPGGINVLSLFSGIGG"
    "AEVALHRLGIRLKNVVSVEISEVNRNIMRCWWEQTNQSGTLIDIVDVQHLNADRLEQLM"
    "NCFGGFDLVVGGSPCANLAGSNRHHRDGLEGKESSLFFDYCRILDLVKCIMTRT"
)


class TestProteinCharacterizationStep:
    def setup_method(self):
        from protein_optimizer.steps.protein_characterization import (
            ProteinCharacterizationStep,
        )
        self.step = ProteinCharacterizationStep()

    def test_basic_run(self):
        """Step runs without errors and produces characterization dict."""
        result = self.step.run(_make_input(_SHORT_SEQ), {})
        parent = result.candidates[0]
        assert "characterization" in parent.metadata
        char = parent.metadata["characterization"]
        assert char["length"] == len(_SHORT_SEQ)
        assert char["molecular_weight"] > 0
        assert 0 < char["isoelectric_point"] < 14

    def test_step_metadata(self):
        result = self.step.run(_make_input(_SHORT_SEQ), {})
        assert result.step_name == "protein_characterization"

    def test_empty_sequence(self):
        result = self.step.run(_make_input(""), {})
        assert len(result.warnings) > 0

    def test_no_candidates(self):
        result = self.step.run(
            StepResult(step_name="input", candidates=[]),
            {},
        )
        assert len(result.warnings) > 0


class TestMolecularWeight:
    def test_basic(self):
        from protein_optimizer.steps.protein_characterization import _molecular_weight
        mw = _molecular_weight("A")
        assert mw == pytest.approx(89.09, abs=1.0)

    def test_dipeptide(self):
        from protein_optimizer.steps.protein_characterization import _molecular_weight
        mw = _molecular_weight("AG")
        expected = 89.09 + 75.03 - 18.015
        assert mw == pytest.approx(expected, abs=0.1)


class TestIsoelectricPoint:
    def test_basic_range(self):
        from protein_optimizer.steps.protein_characterization import _isoelectric_point
        pI = _isoelectric_point("ACDEFGHIKLMNPQRSTVWY")
        assert 0 < pI < 14

    def test_acidic_protein(self):
        from protein_optimizer.steps.protein_characterization import _isoelectric_point
        pI = _isoelectric_point("DDDDEEEE")
        assert pI < 5.0

    def test_basic_protein(self):
        from protein_optimizer.steps.protein_characterization import _isoelectric_point
        pI = _isoelectric_point("KKKKRRRR")
        assert pI > 10.0


class TestExtinctionCoefficient:
    def test_no_chromophores(self):
        from protein_optimizer.steps.protein_characterization import _extinction_coefficient
        ec = _extinction_coefficient("AAAAA")
        assert ec["reduced"] == 0
        assert ec["oxidized"] == 0

    def test_tryptophan(self):
        from protein_optimizer.steps.protein_characterization import _extinction_coefficient
        ec = _extinction_coefficient("WAW")
        assert ec["reduced"] > 0


class TestCysteineAnalysis:
    def test_no_cys(self):
        from protein_optimizer.steps.protein_characterization import _cysteine_analysis
        result = _cysteine_analysis("AAAAA")
        assert result["count"] == 0
        assert result["positions"] == []

    def test_with_cys(self):
        from protein_optimizer.steps.protein_characterization import _cysteine_analysis
        result = _cysteine_analysis("ACGCAAA")
        assert result["count"] == 2
        assert result["positions"] == [2, 4]

    def test_disulfide_potential(self):
        from protein_optimizer.steps.protein_characterization import _cysteine_analysis
        # Two cysteines 20 residues apart — should be potential disulfide pair
        seq = "C" + "A" * 20 + "C"
        result = _cysteine_analysis(seq)
        assert result["disulfide_potential"]["possible_pairs"] >= 1


class TestRareCodonAnalysis:
    def test_no_rare(self):
        from protein_optimizer.steps.protein_characterization import _rare_codon_analysis
        result = _rare_codon_analysis("AAAAAAA")
        assert result["count"] == 0

    def test_tryptophan_is_rare(self):
        from protein_optimizer.steps.protein_characterization import _rare_codon_analysis
        result = _rare_codon_analysis("AWWWAA")
        assert result["count"] >= 3  # W is rare
        assert result["wcm_count"] >= 3


class TestSignalPeptide:
    def test_no_signal(self):
        from protein_optimizer.steps.protein_characterization import _signal_peptide
        result = _signal_peptide("KKKKDDDDEEEE", mode="off")
        # Short charged sequence → no signal
        assert result["detected"] is False

    def test_heuristic_mode(self):
        from protein_optimizer.steps.protein_characterization import _signal_peptide
        # Synthetic signal-peptide-like: positive N + hydrophobic H + AXA cleavage
        seq = "MKRLLLLLLLLLLLLLLAGA" + "A" * 50
        result = _signal_peptide(seq, mode="off")
        assert result["method"] == "heuristic"
        # May or may not detect — just ensure it runs without error


class TestDisorderPrediction:
    def test_basic(self):
        from protein_optimizer.steps.protein_characterization import _disorder_prediction
        result = _disorder_prediction("AAAAAAAAAAAAAAAAAAAAAA", mode="off")
        assert "scores" in result
        assert "regions" in result
        assert "fraction" in result
        assert len(result["scores"]) == 22

    def test_glycine_rich_is_disordered(self):
        from protein_optimizer.steps.protein_characterization import _disorder_prediction
        # Very glycine/proline-rich sequence → should be disordered
        seq = "GGGPGPGPGPGGGPGPGPGPGGGPGPGPGP"
        result = _disorder_prediction(seq, mode="off")
        assert result["fraction"] > 0  # at least some disorder detected


class TestDomainArchitecture:
    def test_short_protein(self):
        from protein_optimizer.steps.protein_characterization import _domain_architecture
        result = _domain_architecture("AAAA", mode="off")
        assert result["count"] == 1  # single domain for tiny protein

    def test_long_protein(self):
        from protein_optimizer.steps.protein_characterization import _domain_architecture
        result = _domain_architecture(_LONG_SEQ, mode="off")
        assert result["count"] >= 1
        assert result["method"] == "heuristic"


class TestOligomericState:
    def test_likely_monomer(self):
        from protein_optimizer.steps.protein_characterization import _oligomeric_state
        result = _oligomeric_state("AGVTAGVTAGVT")
        assert "monomer" in result["state"].lower()


class TestCofactorMotifs:
    def test_no_motifs(self):
        from protein_optimizer.steps.protein_characterization import _cofactor_motifs
        result = _cofactor_motifs("AAAAAAA")
        assert result["summary"] == "None detected"

    def test_hexxh_metalloprotease(self):
        from protein_optimizer.steps.protein_characterization import _cofactor_motifs
        result = _cofactor_motifs("AAAAHEAAHAAAA")
        assert len(result["motifs"]) >= 1
        assert any("HExxH" in m["name"] for m in result["motifs"])


class TestThermalStability:
    def test_basic(self):
        from protein_optimizer.steps.protein_characterization import _thermal_stability
        result = _thermal_stability(_LONG_SEQ, 35.0, 80.0)
        assert 20 <= result["tm"] <= 100
        assert result["confidence"] in ("low", "moderate")


class TestPhStability:
    def test_curve_generation(self):
        from protein_optimizer.steps.protein_characterization import _ph_stability
        result = _ph_stability("ACDEFGHIKLMNPQRSTVWY")
        assert len(result["curve"]) > 0
        assert len(result["stable_range"]) == 2
        assert result["stable_range"][0] < result["stable_range"][1]


class TestAggregationRegions:
    def test_hydrophilic(self):
        from protein_optimizer.steps.protein_characterization import _aggregation_regions
        result = _aggregation_regions("DDDDEEEEKKKK")
        assert result["count"] == 0

    def test_hydrophobic_stretch(self):
        from protein_optimizer.steps.protein_characterization import _aggregation_regions
        result = _aggregation_regions("VVVVIIIILLLLFFFF")
        # Should detect at least one APR
        assert result["score"] > 0


class TestColloidalStability:
    def test_basic(self):
        from protein_optimizer.steps.protein_characterization import _colloidal_stability
        result = _colloidal_stability("KKDDEERR")
        assert "charge_symmetry" in result
        assert "charged_fraction" in result
        assert result["charged_fraction"] == pytest.approx(1.0, abs=0.01)

    def test_empty(self):
        from protein_optimizer.steps.protein_characterization import _colloidal_stability
        result = _colloidal_stability("")
        assert result["charged_fraction"] == 0


class TestSolubilityPrediction:
    def test_basic(self):
        from protein_optimizer.steps.protein_characterization import _solubility_prediction
        result = _solubility_prediction(_LONG_SEQ)
        assert "score" in result
        assert result["label"] in ("Soluble", "Borderline", "Insoluble")

    def test_empty(self):
        from protein_optimizer.steps.protein_characterization import _solubility_prediction
        result = _solubility_prediction("")
        assert result["label"] == "Unknown"


class TestFullCharacterization:
    """Integration test — run the full step on the jcDRM-like sequence."""

    def test_full_run(self):
        from protein_optimizer.steps.protein_characterization import (
            ProteinCharacterizationStep,
        )
        step = ProteinCharacterizationStep()
        result = step.run(_make_input(_LONG_SEQ, "jcDRM"), {})

        parent = result.candidates[0]
        char = parent.metadata["characterization"]

        # Check all property groups are present
        assert "sequence" in char
        assert "molecular_weight" in char
        assert "isoelectric_point" in char
        assert "extinction_coefficient" in char
        assert "aa_composition" in char
        assert "cysteine_count" in char
        assert "disulfide_potential" in char
        assert "rare_codon_count" in char
        assert "signal_peptide" in char
        assert "disorder_scores" in char
        assert "disorder_regions" in char
        assert "domains" in char
        assert "oligomeric_state" in char
        assert "cofactor_motifs" in char
        assert "estimated_tm" in char
        assert "ph_curve" in char
        assert "ph_stable_range" in char
        assert "aggregation_regions" in char
        assert "charge_symmetry" in char
        assert "solubility_score" in char
        assert "solubility_class" in char

        # Sanity checks on values
        assert char["length"] == len(_LONG_SEQ)
        assert char["molecular_weight"] > 30000
        assert 0 < char["isoelectric_point"] < 14
        assert len(char["disorder_scores"]) == len(_LONG_SEQ)
        assert len(char["ph_curve"]) > 10
        assert char["cysteine_count"] == _LONG_SEQ.count("C")
