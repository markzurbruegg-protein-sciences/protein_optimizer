"""Tests for IO utilities."""

import tempfile
from pathlib import Path

import pytest

from protein_optimizer.io_utils import (
    load_input,
    parse_protected_residues,
    read_fasta,
    write_fasta,
)
from protein_optimizer.models import ProteinCandidate, StepResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestReadFasta:
    def test_read_example(self):
        candidates = read_fasta(FIXTURES_DIR / "example.fasta")
        assert len(candidates) == 1
        assert candidates[0].name == "GFP_example"
        assert candidates[0].sequence.startswith("MSKGEELFTG")

    def test_read_nonexistent_raises(self):
        with pytest.raises(Exception):
            read_fasta("/nonexistent/path.fasta")


class TestWriteFasta:
    def test_roundtrip(self):
        candidates = [
            ProteinCandidate(sequence="MACGG", name="test1"),
            ProteinCandidate(sequence="MALGG", name="test2"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.fasta"
            write_fasta(candidates, path)

            assert path.exists()
            reloaded = read_fasta(path)
            assert len(reloaded) == 2
            assert reloaded[0].sequence == "MACGG"
            assert reloaded[1].sequence == "MALGG"


class TestLoadInput:
    def test_load_fasta(self):
        result = load_input(FIXTURES_DIR / "example.fasta")
        assert isinstance(result, StepResult)
        assert result.step_name == "input"
        assert len(result.candidates) == 1

    def test_load_json(self):
        sr = StepResult(
            step_name="test",
            candidates=[ProteinCandidate(sequence="MACGG", name="test")],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            sr.save(path)
            loaded = load_input(path)
            assert loaded.step_name == "test"


class TestParseProtectedResidues:
    def test_none(self):
        assert parse_protected_residues(None) == set()

    def test_list(self):
        assert parse_protected_residues([10, 25, 30]) == {10, 25, 30}

    def test_string(self):
        assert parse_protected_residues("10,25,30") == {10, 25, 30}

    def test_string_with_spaces(self):
        assert parse_protected_residues("10, 25, 30") == {10, 25, 30}

    def test_empty_string(self):
        assert parse_protected_residues("") == set()
