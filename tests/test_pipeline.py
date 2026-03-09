"""Tests for config loading and pipeline orchestration."""

import tempfile
from pathlib import Path

import pytest
import yaml

from protein_optimizer.config import load_config
from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.pipeline import Pipeline, _import_all_steps, _get_step_config
from protein_optimizer.steps.base import get_step, list_steps


class TestConfig:
    def test_default_config(self):
        config = load_config(None)
        assert "pipeline" in config
        assert "global" in config
        assert "steps" in config

    def test_custom_config_merge(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "global": {"expression_host": "pichia"},
                "pipeline": {"steps": ["motif_scan"]},
            }, f)
            f.flush()

            config = load_config(f.name)
            assert config["global"]["expression_host"] == "pichia"
            assert config["pipeline"]["steps"] == ["motif_scan"]
            # Defaults preserved
            assert "steps" in config

    def test_get_step_config(self):
        config = load_config(None)
        sc = _get_step_config(config, "cysteine_scan")
        assert isinstance(sc, dict)

    def test_get_step_config_missing(self):
        config = load_config(None)
        sc = _get_step_config(config, "nonexistent_step")
        assert sc == {}


class TestStepRegistry:
    def test_import_all_steps(self):
        _import_all_steps()
        steps = list_steps()
        assert len(steps) >= 10  # should have many steps

    def test_get_known_step(self):
        _import_all_steps()
        step = get_step("cysteine_scan")
        assert step.name == "cysteine_scan"
        assert step.tier == 1

    def test_get_unknown_step_raises(self):
        with pytest.raises(KeyError, match="Unknown step"):
            get_step("nonexistent_step_xyz")

    def test_all_steps_have_metadata(self):
        _import_all_steps()
        steps = list_steps()
        for name, cls in steps.items():
            assert cls.name, f"{name} missing name"
            assert cls.tier > 0, f"{name} missing tier"
            assert cls.title, f"{name} missing title"
            assert cls.description, f"{name} missing description"


class TestPipeline:
    def test_tier1_pipeline(self):
        """Run a minimal Tier 1 pipeline end-to-end."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        fasta_path = fixtures_dir / "example.fasta"

        config = load_config(None)
        config["pipeline"]["steps"] = [
            "cysteine_scan",
            "motif_scan",
            "sequence_complexity",
        ]

        pipeline = Pipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.run(fasta_path, tmpdir)

            assert len(result.candidates) >= 1
            assert result.step_name == "sequence_complexity"

            # Check intermediate files were saved
            assert (Path(tmpdir) / "cysteine_scan.json").exists()
            assert (Path(tmpdir) / "motif_scan.json").exists()
            assert (Path(tmpdir) / "sequence_complexity.json").exists()

    def test_pipeline_with_config_file(self):
        """Test pipeline with a YAML config file."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        fasta_path = fixtures_dir / "example.fasta"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "pipeline": {"steps": ["cysteine_scan"]},
                "global": {"protected_residues": [48]},
            }, f)
            f.flush()

            config = load_config(f.name)
            pipeline = Pipeline(config)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = pipeline.run(fasta_path, tmpdir)
                # Position 48 (Cys in GFP) should be protected
                for c in result.candidates:
                    for m in c.mutations:
                        assert m.position != 48
