"""YAML-based configuration system for the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "pipeline": {
        "name": "default",
        "description": "Default protein optimization pipeline",
        "steps": [
            "cysteine_scan",
            "motif_scan",
            "sequence_complexity",
        ],
    },
    "global": {
        "protected_residues": [],
        "expression_host": "ecoli",
        "output_dir": "./results",
    },
    "steps": {
        "cysteine_scan": {
            "replacement": "S",  # S (serine) or A (alanine)
            "generate_remove_all": True,
        },
        "motif_scan": {
            "check_deamidation": True,
            "check_oxidation": True,
            "check_proteolysis": True,
            "check_aggregation": True,
            "hydrophobic_run_length": 5,
        },
        "sequence_complexity": {
            "homopolymer_threshold": 5,
            "proline_run_threshold": 3,
        },
        "find_homologs": {
            "database": "UniRef90",
            "max_hits": 500,
            "min_identity": 0.3,
            "evalue": 1e-5,
        },
        "consensus_design": {
            "min_conservation": 0.5,
            "max_mutations": 20,
        },
        "pssm_analysis": {
            "min_log_odds": 2.0,
            "max_mutations": 30,
        },
        "predict_structure": {
            "method": "boltz2",  # boltz2 or esmfold
            "use_msa_server": True,
        },
        "stability_ddg": {
            "method": "rosetta",  # rosetta or foldx
            "ddg_threshold": -1.0,
            "saturation_mutagenesis": False,
            "positions": [],  # empty = use flagged positions from earlier steps
        },
        "disulfide_design": {
            "cb_distance_min": 3.5,
            "cb_distance_max": 4.5,
            "max_candidates": 10,
        },
        "cavity_fill": {
            "min_volume": 20.0,  # Å³
            "max_mutations": 10,
        },
        "surface_patch": {
            "sap_threshold": 0.5,
            "max_mutations": 10,
        },
        "e1_score": {
            "model": "Profluent-Bio/E1-600m",
            "retrieval_augmented": True,
            "top_k_mutations": 50,
        },
        "esm1v_score": {
            "num_models": 5,  # ensemble of 5
            "top_k_mutations": 50,
        },
        "proteinmpnn_design": {
            "use_soluble_model": True,
            "omit_aas": "C",
            "sampling_temp": 0.1,
            "num_sequences": 100,
            "batch_size": 1,
        },
        "esmif1_score": {
            "temperature": 1.0,
        },
        "combine_variants": {
            "max_mutations_per_candidate": 8,
            "epistasis_check": True,
            "min_individual_score_percentile": 75,
        },
        "solubility_ssm": {
            "conda_env": "protopt",
            "protsolm_model": "protsolm",
            "max_positions": 100,
            "batch_size": 32,
        },
        "mutation_optimizer": {
            "e1_conda_env": "e1",
            "e1_model": "Profluent-Bio/E1-600m",
            "retrieval_augmented": True,
            "max_homologs_context": 8,
            "max_pool_size": 20,
            "max_combination_order": 5,
            "max_library_size": 25000,
            "max_charge_shift": 5,
        },
        "rfdiffusion_diversify": {
            "partial_T": 15,
            "T": 50,
            "noise_scale_ca": 0.5,
            "noise_scale_frame": 0.5,
            "num_designs": 10,
        },
        "design_validate": {
            "plddt_threshold": 80.0,
            "rmsd_threshold": 2.0,
            "structure_method": "boltz2",
        },
        "motif_scaffold": {
            "motif_residues": [],
            "contig_length_range": [50, 150],
            "num_designs": 10,
        },
    },
    "scoring": {
        "weights": {
            "e1_score": 1.0,
            "esm1v_score": 1.0,
            "proteinmpnn_global_score": 1.0,
            "esmif1_score": 0.5,
            "ddg": 1.0,
            "consensus_conservation": 0.5,
        },
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load pipeline configuration from YAML file.

    Merges user config on top of defaults so every key has a value.
    """
    config = _deep_copy(DEFAULT_CONFIG)
    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                user_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, user_config)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _deep_copy(d: dict) -> dict:
    """Simple deep copy for nested dicts/lists."""
    import copy
    return copy.deepcopy(d)
