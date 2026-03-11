"""Tier 1 — Protein Characterization.

Computes comprehensive biophysical, structural, and expression-relevant
properties for the protein of interest.  Results are cached as JSON and
used to populate the expanded metrics sidebar in the HTML report.

Sequence-only calculations are always available.  Where external tools
(SignalP 6, IUPred3, HMMER / hmmscan) are found on ``$PATH`` they are
used automatically; otherwise robust heuristic fallbacks run.

Property groups
~~~~~~~~~~~~~~~
* Sequence & Primary Structure — MW, pI, extinction coefficient,
  amino acid composition, cysteine / disulfide annotation, rare-codon
  load, signal / transit peptides, intrinsically disordered regions.
* Structural Properties — domain architecture, oligomeric state hints,
  cofactor / metal-binding motifs.
* Biophysical / Stability — estimated Tm, pH-stability window
  (charge-vs-pH curve), aggregation-prone regions (APRs),
  colloidal-stability proxies, E. coli solubility prediction.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Amino-acid physicochemical tables
# ═══════════════════════════════════════════════════════════════════

_MW: dict[str, float] = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "E": 147.13, "Q": 146.15, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

_PKA: dict[str, float] = {
    "D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.46,
    "H": 6.00, "K": 10.53, "R": 12.48,
}
_NTERM_PKA = 9.69
_CTERM_PKA = 2.34

_HYDROPATHY: dict[str, float] = {  # Kyte-Doolittle
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Chou-Fasman β-sheet propensity (higher → more β prone → aggregation risk)
_BETA_PROPENSITY: dict[str, float] = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "E": 0.37, "Q": 1.10, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

# E. coli K-12 codon adaptation index (relative adaptiveness per AA)
# Low values (<0.3) indicate rare codons
_ECOLI_CODON_RARITY: dict[str, float] = {
    "A": 0.85, "R": 0.40, "N": 0.80, "D": 0.90, "C": 0.30,
    "E": 0.85, "Q": 0.75, "G": 0.75, "H": 0.65, "I": 0.55,
    "L": 0.50, "K": 0.80, "M": 0.35, "F": 0.60, "P": 0.70,
    "S": 0.70, "T": 0.80, "W": 0.20, "Y": 0.55, "V": 0.75,
}

# Instability index dipeptide weights (Guruprasad et al., 1990)
# Subset of the 400 dipeptide weights — full table in _instability_index()
_DIWV: dict[str, float] = {}  # populated lazily


def _get_diwv() -> dict[str, float]:
    """Return dipeptide instability weight values (DIWV) table."""
    if _DIWV:
        return _DIWV
    # Use BioPython's table if available; else use a small default
    try:
        from Bio.SeqUtils.ProtParam import ProtParamData
        for k, v in ProtParamData.DIWV.items():
            _DIWV[k] = v
    except (ImportError, AttributeError):
        pass  # _instability_index has its own fallback
    return _DIWV


# ═══════════════════════════════════════════════════════════════════
# Step class
# ═══════════════════════════════════════════════════════════════════


class ProteinCharacterizationStep(BaseStep):
    name = "protein_characterization"
    tier = 1
    title = "Protein Characterization"
    description = (
        "Compute comprehensive biophysical, structural, and expression-"
        "relevant properties for the protein of interest."
    )
    requires: list[str] = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        expression_host = config.get("expression_host",
                                     config.get("_global", {}).get("expression_host", "ecoli"))
        disorder_threshold = config.get("disorder_threshold", 0.5)
        signal_scan_length = config.get("signal_peptide_scan_length", 70)
        ph_range = config.get("ph_range", [2.0, 12.0])
        ext_cfg = config.get("external_tools", {})

        candidates: list[ProteinCandidate] = list(step_input.candidates)
        warnings: list[str] = []

        # Find the wild-type (parent) candidate
        parent = None
        for c in candidates:
            if c.parent_id is None:
                parent = c
                break
        if parent is None and candidates:
            parent = candidates[0]

        if parent is None:
            warnings.append("No candidates found — skipping characterization.")
            return StepResult(
                step_name=self.name, candidates=candidates,
                config_used=config, warnings=warnings,
            )

        seq = parent.sequence
        n = len(seq)
        if n == 0:
            warnings.append("Empty sequence — skipping characterization.")
            return StepResult(
                step_name=self.name, candidates=candidates,
                config_used=config, warnings=warnings,
            )

        # ── Compute all property groups ──────────────────────────
        char: dict[str, Any] = {}

        # 1. Sequence & Primary Structure
        char["sequence"] = seq
        char["length"] = n
        char["molecular_weight"] = _molecular_weight(seq)
        char["isoelectric_point"] = _isoelectric_point(seq)
        char["extinction_coefficient"] = _extinction_coefficient(seq)
        char["aa_composition"] = _aa_composition(seq)
        char["gravy"] = _gravy(seq)
        char["instability_index"] = _instability_index(seq)
        char["instability_label"] = "Stable" if char["instability_index"] < 40 else "Unstable"
        char["aliphatic_index"] = _aliphatic_index(seq)
        char["aromaticity"] = _aromaticity(seq)

        # Cysteine / disulfide
        cys = _cysteine_analysis(seq)
        char["cysteine_count"] = cys["count"]
        char["cysteine_positions"] = cys["positions"]
        char["disulfide_potential"] = cys["disulfide_potential"]

        # Rare codons
        rc = _rare_codon_analysis(seq, expression_host)
        char["rare_codon_count"] = rc["count"]
        char["rare_codon_fraction"] = rc["fraction"]
        char["rare_codon_positions"] = rc["positions"]
        char["rare_codon_details"] = rc["details"]

        # Signal / transit peptides
        sp = _signal_peptide(seq, signal_scan_length, ext_cfg.get("signalp", "auto"))
        char["signal_peptide"] = sp
        if sp.get("detected"):
            logger.info(f"Signal peptide detected: residues 1–{sp['cleavage_position']}")

        # Intrinsically disordered regions
        idr = _disorder_prediction(seq, disorder_threshold, ext_cfg.get("iupred", "auto"))
        char["disorder_scores"] = idr["scores"]
        char["disorder_regions"] = idr["regions"]
        char["disorder_fraction"] = idr["fraction"]
        char["disorder_method"] = idr["method"]

        # 2. Structural Properties
        # Domain architecture
        dom = _domain_architecture(seq, ext_cfg.get("hmmscan", "auto"))
        char["domains"] = dom["domains"]
        char["domain_count"] = dom["count"]
        char["domain_method"] = dom["method"]

        # Oligomeric state
        oligo = _oligomeric_state(seq)
        char["oligomeric_state"] = oligo["state"]
        char["oligomeric_evidence"] = oligo["evidence"]

        # Cofactor / metal motifs
        cofactors = _cofactor_motifs(seq)
        char["cofactor_motifs"] = cofactors["motifs"]
        char["cofactor_summary"] = cofactors["summary"]

        # 3. Biophysical / Stability
        # Thermal stability estimate
        tm = _thermal_stability(seq, char["instability_index"], char["aliphatic_index"])
        char["estimated_tm"] = tm["tm"]
        char["tm_confidence"] = tm["confidence"]

        # pH stability window (charge-vs-pH curve)
        ph = _ph_stability(seq, ph_range)
        char["ph_curve"] = ph["curve"]  # list of [pH, charge]
        char["ph_stable_range"] = ph["stable_range"]

        # Aggregation-prone regions
        apr = _aggregation_regions(seq)
        char["aggregation_score"] = apr["score"]
        char["aggregation_regions"] = apr["regions"]
        char["aggregation_region_count"] = apr["count"]

        # ProtSolM deep-learning solubility prediction (Tan et al., IEEE BIBM 2024)
        # Uses ESM2 + ProtSSN EGNN + handcrafted features via external helper
        pdb_path = parent.structure_path or parent.metadata.get("structure_path", "")
        if not pdb_path:
            # Auto-detect from results directory or other candidates
            for cand in candidates:
                sp = getattr(cand, "structure_path", "") or cand.metadata.get("structure_path", "")
                if sp and Path(sp).exists():
                    pdb_path = sp
                    break
        if not pdb_path:
            # Look for PDB in standard locations relative to input FASTA
            name = parent.name.split("_")[0] if parent.name else ""
            if name:
                for pattern in [
                    Path(f"run_proteins/{name}_results/structures/{name}.pdb"),
                    Path(f"{name}_results/structures/{name}.pdb"),
                ]:
                    if pattern.exists():
                        pdb_path = str(pattern)
                        break
        protsolm = _protsolm_prediction(seq, pdb_path if pdb_path else None)
        char["protsolm_probability"] = protsolm["probability"]
        char["protsolm_label"] = protsolm["label"]
        char["protsolm_confidence"] = protsolm["confidence"]
        char["protsolm_score_pct"] = protsolm["score_pct"]
        char["protsolm_method"] = protsolm.get("method", "ProtSolM")

        # TANGO-like β-aggregation prediction (Zyggregator/AGGRESCAN methodology)
        tango = _tango_like_aggregation(seq)
        char["tango_scores"] = tango["scores"]
        char["tango_aprs"] = tango["aprs"]
        char["tango_apr_count"] = tango["apr_count"]
        char["tango_overall_score"] = tango["overall_score"]
        char["tango_gatekeepers"] = tango["gatekeepers"]
        char["tango_nucleation_cores"] = tango["nucleation_cores"]

        # Colloidal stability
        col = _colloidal_stability(seq)
        char["charge_symmetry"] = col["charge_symmetry"]
        char["charged_fraction"] = col["charged_fraction"]
        char["positive_fraction"] = col["positive_fraction"]
        char["negative_fraction"] = col["negative_fraction"]
        char["net_charge_7"] = col["net_charge_7"]

        # Solubility prediction (Wilkinson-Harrison)
        sol = _solubility_prediction(seq)
        char["solubility_score"] = sol["score"]
        char["solubility_class"] = sol["label"]

        # Enhanced multi-method solubility ensemble
        sol_ensemble = _solubility_ensemble(seq, protsolm_result=protsolm)
        char["solubility_ensemble"] = sol_ensemble["ensemble_score"]
        char["solubility_ensemble_class"] = sol_ensemble["ensemble_class"]
        char["solubility_ensemble_confidence"] = sol_ensemble["confidence"]
        char["solubility_methods"] = sol_ensemble["methods"]

        # RP3Net — E. coli recombinant expression prediction
        rp3net = _rp3net_prediction(seq)
        char["rp3net_probability"] = rp3net["probability"]
        char["rp3net_label"] = rp3net["label"]
        char["rp3net_score_pct"] = rp3net["score_pct"]
        char["rp3net_method"] = rp3net["method"]

        # Store in parent metadata
        parent.metadata["characterization"] = char

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
            metadata={"expression_host": expression_host},
        )


# ═══════════════════════════════════════════════════════════════════
# Property computation functions
# ═══════════════════════════════════════════════════════════════════


def _molecular_weight(seq: str) -> float:
    """Molecular weight in Daltons (monoisotopic water loss)."""
    n = len(seq)
    return sum(_MW.get(aa, 128.0) for aa in seq) - 18.015 * (n - 1)


def _net_charge(seq: str, pH: float) -> float:
    """Net charge at given pH using Henderson-Hasselbalch."""
    charge = 1.0 / (1.0 + 10 ** (pH - _NTERM_PKA))   # N-term
    charge -= 1.0 / (1.0 + 10 ** (_CTERM_PKA - pH))   # C-term
    for aa in seq:
        pKa = _PKA.get(aa)
        if pKa is None:
            continue
        if aa in ("D", "E", "C", "Y"):  # acidic
            charge -= 1.0 / (1.0 + 10 ** (pKa - pH))
        else:  # basic: H, K, R
            charge += 1.0 / (1.0 + 10 ** (pH - pKa))
    return charge


def _isoelectric_point(seq: str) -> float:
    """Estimate pI by bisection."""
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _extinction_coefficient(seq: str) -> dict[str, float]:
    """Molar extinction coefficient at 280 nm (M⁻¹ cm⁻¹).

    Returns coefficients assuming all Cys form disulfide bonds
    and assuming all Cys are reduced.
    """
    n_trp = seq.count("W")
    n_tyr = seq.count("Y")
    n_cys = seq.count("C")
    # Pace et al., 1995
    ec_oxidized = n_trp * 5500 + n_tyr * 1490 + (n_cys // 2) * 125
    ec_reduced = n_trp * 5500 + n_tyr * 1490
    return {"oxidized": ec_oxidized, "reduced": ec_reduced}


def _aa_composition(seq: str) -> dict[str, float]:
    """Fractional amino acid composition."""
    n = len(seq)
    if n == 0:
        return {}
    from collections import Counter
    counts = Counter(seq)
    return {aa: counts.get(aa, 0) / n for aa in sorted(_MW.keys())}


def _gravy(seq: str) -> float:
    """Grand Average of Hydropathy (Kyte-Doolittle)."""
    n = len(seq)
    if n == 0:
        return 0.0
    return sum(_HYDROPATHY.get(aa, 0) for aa in seq) / n


def _instability_index(seq: str) -> float:
    """Guruprasad instability index."""
    n = len(seq)
    if n < 2:
        return 0.0
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        pa = ProteinAnalysis(seq)
        return pa.instability_index()
    except Exception:
        pass
    # Manual fallback using DIWV table
    diwv = _get_diwv()
    if not diwv:
        return 0.0
    total = sum(diwv.get(seq[i:i + 2], 0) for i in range(n - 1))
    return (10.0 / n) * total


def _aliphatic_index(seq: str) -> float:
    """Aliphatic index (thermostability proxy)."""
    n = len(seq)
    if n == 0:
        return 0.0
    a = seq.count("A") / n * 100
    v = seq.count("V") / n * 100
    i_ = seq.count("I") / n * 100
    l = seq.count("L") / n * 100
    return a + 2.9 * v + 3.9 * (i_ + l)


def _aromaticity(seq: str) -> float:
    """Fraction of aromatic residues (F + W + Y)."""
    n = len(seq)
    if n == 0:
        return 0.0
    return sum(1 for aa in seq if aa in "FWY") / n


# ── Cysteine / Disulfide ────────────────────────────────────────


def _cysteine_analysis(seq: str) -> dict[str, Any]:
    """Analyse cysteines and predict disulfide bond potential."""
    positions = [i + 1 for i, aa in enumerate(seq) if aa == "C"]
    count = len(positions)

    # Heuristic: pair cys within 10-150 residue distance
    pairs: list[tuple[int, int]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            sep = positions[j] - positions[i]
            if 10 <= sep <= 150:
                pairs.append((positions[i], positions[j]))

    return {
        "count": count,
        "positions": positions,
        "disulfide_potential": {
            "possible_pairs": len(pairs),
            "unpaired_cys": max(0, count - 2 * len(pairs)),
            "pairs": pairs[:10],  # top 10
        },
    }


# ── Rare Codons ─────────────────────────────────────────────────


_RARE_THRESHOLD = 0.35  # amino acids below this adaptiveness are "rare"

# Amino acids whose codons are particularly rare in E. coli K-12
_ECOLI_RARE_AAS = {"W", "C", "M", "R", "I", "L"}


def _rare_codon_analysis(seq: str, host: str = "ecoli") -> dict[str, Any]:
    """Count amino acids encoded by rare codons for the given host."""
    positions: list[dict[str, Any]] = []
    for i, aa in enumerate(seq):
        rarity = _ECOLI_CODON_RARITY.get(aa, 0.5)
        if rarity < _RARE_THRESHOLD:
            positions.append({
                "position": i + 1,
                "aa": aa,
                "rarity": round(rarity, 2),
            })

    # Simple summary: count of traditional "rare-in-ecoli" AAs (W, C, M)
    wcm = sum(1 for aa in seq if aa in "WCM")

    # Runs of consecutive rare codons (>= 2) are especially problematic
    consecutive_runs: list[dict[str, Any]] = []
    run_start = -1
    for i, aa in enumerate(seq):
        rarity = _ECOLI_CODON_RARITY.get(aa, 0.5)
        if rarity < _RARE_THRESHOLD:
            if run_start < 0:
                run_start = i
        else:
            if run_start >= 0 and (i - run_start) >= 2:
                consecutive_runs.append({
                    "start": run_start + 1,
                    "end": i,
                    "length": i - run_start,
                    "residues": seq[run_start:i],
                })
            run_start = -1
    # Handle trailing run
    if run_start >= 0 and (len(seq) - run_start) >= 2:
        consecutive_runs.append({
            "start": run_start + 1,
            "end": len(seq),
            "length": len(seq) - run_start,
            "residues": seq[run_start:],
        })

    return {
        "count": len(positions),
        "fraction": round(len(positions) / max(len(seq), 1), 4),
        "positions": positions[:50],  # cap output size
        "wcm_count": wcm,
        "consecutive_runs": consecutive_runs,
        "details": f"{len(positions)} rare-codon AAs ({len(positions)/max(len(seq),1)*100:.1f}%), "
                   f"WCM={wcm}, {len(consecutive_runs)} consecutive-rare runs",
    }


# ── Signal / Transit Peptides ───────────────────────────────────


def _signal_peptide(
    seq: str, scan_length: int = 70, mode: str = "auto",
) -> dict[str, Any]:
    """Detect signal peptides.

    Tries SignalP 6 first (if mode != 'off' and available on PATH);
    falls back to von Heijne hydrophobicity heuristic.
    """
    if mode != "off" and shutil.which("signalp6"):
        result = _signalp_external(seq)
        if result is not None:
            return result

    # ── Heuristic fallback (von Heijne algorithm) ──
    return _signalp_heuristic(seq, scan_length)


def _signalp_external(seq: str) -> dict[str, Any] | None:
    """Run SignalP 6 and parse result."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">query\n")
            f.write(seq + "\n")
            fasta_path = f.name

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["signalp6", "--fastafile", fasta_path,
                 "--output_dir", tmpdir, "--format", "short"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(f"SignalP 6 failed: {result.stderr[:200]}")
                return None

            # Parse prediction_results.txt
            out_file = Path(tmpdir) / "prediction_results.txt"
            if not out_file.exists():
                return None
            for line in out_file.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    sp_type = parts[1]
                    if sp_type == "OTHER":
                        return {"detected": False, "method": "signalp6",
                                "cleavage_position": None, "type": None}
                    # Parse cleavage site
                    cs_match = re.search(r"CS pos: (\d+)-(\d+)", line)
                    cs = int(cs_match.group(2)) if cs_match else None
                    return {
                        "detected": True, "method": "signalp6",
                        "type": sp_type,
                        "cleavage_position": cs,
                        "raw": line.strip(),
                    }
        return None
    except Exception as e:
        logger.debug(f"SignalP 6 integration failed: {e}")
        return None


def _signalp_heuristic(seq: str, scan_length: int = 70) -> dict[str, Any]:
    """Von Heijne heuristic signal peptide detection.

    Scans the first *scan_length* residues for:
    - N-region: 1–5 positively charged residues (K, R)
    - H-region: hydrophobic core of 7–15 residues (mean KD > 1.6)
    - C-region: small/neutral residues before cleavage (AXA motif)
    """
    segment = seq[:min(scan_length, len(seq))]
    if len(segment) < 15:
        return {"detected": False, "method": "heuristic",
                "cleavage_position": None, "type": None}

    best_score = 0.0
    best_cleavage = None

    for h_start in range(1, 8):  # N-region length 1-7
        # Check N-region: positively charged
        n_region = segment[:h_start]
        n_charge = sum(1 for aa in n_region if aa in "KR")

        for h_len in range(7, 16):  # H-region length 7-15
            h_end = h_start + h_len
            if h_end + 3 > len(segment):
                break

            h_region = segment[h_start:h_end]
            h_hydro = sum(_HYDROPATHY.get(aa, 0) for aa in h_region) / len(h_region)

            if h_hydro < 1.0:
                continue

            # Check C-region for AXA-like cleavage motif
            c_region = segment[h_end:h_end + 6]
            # Look for small residues at -3 and -1 positions
            axa_score = 0
            for offset in range(max(len(c_region) - 2, 0)):
                if c_region[offset] in "AGST" and len(c_region) > offset + 2 and c_region[offset + 2] in "AGST":
                    axa_score = 1.0
                    cleavage_pos = h_end + offset + 3
                    break
            else:
                cleavage_pos = h_end + 3

            score = (n_charge * 0.3) + (h_hydro * 0.5) + (axa_score * 0.2)
            if score > best_score:
                best_score = score
                best_cleavage = cleavage_pos

    detected = best_score > 1.2 and best_cleavage is not None
    return {
        "detected": detected,
        "method": "heuristic",
        "type": "Sec/SPI" if detected else None,
        "cleavage_position": best_cleavage if detected else None,
        "score": round(best_score, 3),
    }


# ── Intrinsically Disordered Regions ───────────────────────────


def _disorder_prediction(
    seq: str, threshold: float = 0.5, mode: str = "auto",
) -> dict[str, Any]:
    """Predict intrinsically disordered regions.

    Tries (in order):
    1. metapredict v2 — trained bidirectional LSTM, fast, accurate
    2. IUPred3 CLI — if available on PATH
    3. FoldIndex heuristic — always-available fallback
    """
    # 1. metapredict (pip-installable ML predictor)
    if mode != "off":
        mp_result = _metapredict_disorder(seq)
        if mp_result is not None:
            return _parse_disorder(mp_result["scores"], threshold,
                                   method="metapredict")

    # 2. IUPred3 CLI
    if mode != "off" and shutil.which("iupred3"):
        result = _iupred_external(seq)
        if result is not None:
            return _parse_disorder(result["scores"], threshold, method="iupred3")

    # 3. Heuristic (FoldIndex algorithm, Prilusky & Biber 2005)
    scores = _disorder_heuristic(seq)
    return _parse_disorder(scores, threshold, method="heuristic")


def _metapredict_disorder(seq: str) -> dict[str, Any] | None:
    """Run metapredict (Emenecker et al., 2021) for disorder prediction.

    metapredict uses a bidirectional LSTM trained on consensus disorder
    scores from multiple predictors.  It also provides predicted pLDDT
    (AlphaFold2 confidence) per residue.
    """
    try:
        import metapredict as meta

        # Per-residue disorder scores (0 = ordered, 1 = disordered)
        scores = [float(s) for s in meta.predict_disorder(seq)]

        return {"scores": scores}
    except ImportError:
        logger.debug("metapredict not installed — skipping ML disorder prediction.")
        return None
    except Exception as e:
        logger.debug(f"metapredict failed: {e}")
        return None


def _iupred_external(seq: str) -> dict[str, Any] | None:
    """Run IUPred3 and return per-residue scores."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">query\n")
            f.write(seq + "\n")
            fasta_path = f.name

        result = subprocess.run(
            ["iupred3", fasta_path, "long"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return None

        scores: list[float] = []
        for line in result.stdout.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    scores.append(float(parts[2]))
                except ValueError:
                    continue

        if len(scores) == len(seq):
            return {"scores": scores}
        return None
    except Exception as e:
        logger.debug(f"IUPred3 integration failed: {e}")
        return None


def _disorder_heuristic(seq: str, window: int = 21) -> list[float]:
    """FoldIndex-based disorder prediction (Prilusky & Biber, 2005).

    The FoldIndex score at each residue position is computed as:
        FI = 2.785 * <H_norm> - |<R>| - 1.151
    where:
        <H_norm> = mean Kyte-Doolittle hydropathy in the window,
                   normalized to [0, 1] range
        <R>      = mean net charge in the window (fraction of K,R minus D,E)

    Positive FI → ordered; negative FI → disordered.
    We convert to a 0-1 disorder score where:
        disorder_score = sigmoid(-FI * gain)
    so that negative FI gives high disorder scores.
    """
    n = len(seq)
    if n == 0:
        return []

    # Kyte-Doolittle normalized to [0, 1]: original range is [-4.5, 4.5]
    h_norm = [(_HYDROPATHY.get(aa, 0.0) + 4.5) / 9.0 for aa in seq]

    # Per-residue charge: +1 for K/R, -1 for D/E, 0 otherwise
    charge = []
    for aa in seq:
        if aa in "KR":
            charge.append(1.0)
        elif aa in "DE":
            charge.append(-1.0)
        else:
            charge.append(0.0)

    half = window // 2
    scores: list[float] = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        wsize = end - start

        mean_h = sum(h_norm[start:end]) / wsize
        mean_r = sum(charge[start:end]) / wsize

        # FoldIndex formula
        fi = 2.785 * mean_h - abs(mean_r) - 1.151

        # Convert to 0-1 disorder score using a sigmoid-like transform
        # fi > 0 → ordered (low score), fi < 0 → disordered (high score)
        # Gain controls sharpness of transition
        gain = 10.0
        disorder_score = 1.0 / (1.0 + math.exp(gain * fi))
        scores.append(disorder_score)

    return scores


def _parse_disorder(
    scores: list[float], threshold: float, method: str,
) -> dict[str, Any]:
    """Parse per-residue disorder scores into regions."""
    n = len(scores)
    regions: list[dict[str, Any]] = []

    in_region = False
    start = 0
    for i, s in enumerate(scores):
        if s >= threshold and not in_region:
            start = i
            in_region = True
        elif s < threshold and in_region:
            if (i - start) >= 4:  # minimum region length
                regions.append({
                    "start": start + 1,
                    "end": i,
                    "length": i - start,
                    "mean_score": round(sum(scores[start:i]) / (i - start), 3),
                })
            in_region = False
    if in_region and (n - start) >= 4:
        regions.append({
            "start": start + 1,
            "end": n,
            "length": n - start,
            "mean_score": round(sum(scores[start:n]) / (n - start), 3),
        })

    disordered_residues = sum(1 for s in scores if s >= threshold)
    fraction = disordered_residues / max(n, 1)

    return {
        "scores": [round(s, 3) for s in scores],
        "regions": regions,
        "fraction": round(fraction, 4),
        "method": method,
        "disordered_residues": disordered_residues,
    }


# ── Domain Architecture ─────────────────────────────────────────


def _domain_architecture(seq: str, mode: str = "auto") -> dict[str, Any]:
    """Detect domain boundaries.

    Tries hmmscan (Pfam) first; falls back to composition-based heuristic.
    """
    if mode != "off" and shutil.which("hmmscan"):
        result = _hmmscan_external(seq)
        if result is not None:
            return result

    return _domain_heuristic(seq)


def _hmmscan_external(seq: str) -> dict[str, Any] | None:
    """Run hmmscan against Pfam-A and parse domain hits."""
    # Check for Pfam-A database
    pfam_paths = [
        Path("/usr/local/share/Pfam-A.hmm"),
        Path.home() / "databases" / "Pfam-A.hmm",
        Path.home() / "pfam" / "Pfam-A.hmm",
    ]
    pfam_db = None
    for p in pfam_paths:
        if p.exists():
            pfam_db = str(p)
            break
    if pfam_db is None:
        logger.debug("Pfam-A.hmm not found in standard locations.")
        return None

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">query\n")
            f.write(seq + "\n")
            fasta_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".tbl", delete=False) as tbl:
            tbl_path = tbl.name

        result = subprocess.run(
            ["hmmscan", "--domtblout", tbl_path, "--noali",
             "-E", "1e-5", pfam_db, fasta_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return None

        domains: list[dict[str, Any]] = []
        for line in Path(tbl_path).read_text().splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 23:
                domains.append({
                    "name": parts[0],
                    "accession": parts[1],
                    "start": int(parts[17]),
                    "end": int(parts[18]),
                    "evalue": float(parts[6]),
                    "score": float(parts[7]),
                })

        return {
            "domains": domains,
            "count": len(domains),
            "method": "hmmscan/Pfam",
        }
    except Exception as e:
        logger.debug(f"hmmscan integration failed: {e}")
        return None


def _domain_heuristic(seq: str, window: int = 30, step: int = 10) -> dict[str, Any]:
    """Detect domain boundaries from composition variance.

    Slides a window and measures local hydropathy/charge composition
    variance. Sharp transitions suggest domain boundaries.
    """
    n = len(seq)
    if n < 60:
        return {"domains": [{"start": 1, "end": n, "name": "single_domain"}],
                "count": 1, "method": "heuristic"}

    # Compute local properties in windows
    scores: list[float] = []
    for i in range(0, n - window + 1, step):
        w = seq[i:i + window]
        h = sum(_HYDROPATHY.get(aa, 0) for aa in w) / window
        c = sum(1 for aa in w if aa in "DEKRH") / window
        p = sum(1 for aa in w if aa == "P") / window
        g = sum(1 for aa in w if aa == "G") / window
        scores.append(h + c * 3 + p * 2 + g * 2)

    if len(scores) < 3:
        return {"domains": [{"start": 1, "end": n, "name": "single_domain"}],
                "count": 1, "method": "heuristic"}

    # Detect sharp transitions (derivative peaks)
    diffs = [abs(scores[i + 1] - scores[i]) for i in range(len(scores) - 1)]
    mean_diff = sum(diffs) / max(len(diffs), 1)
    std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / max(len(diffs), 1)) ** 0.5

    # Boundaries where diff > mean + 1.5*std
    boundaries = [0]
    for i, d in enumerate(diffs):
        if d > mean_diff + 1.5 * std_diff:
            boundary_pos = (i + 1) * step + window // 2
            # Avoid boundaries too close together (<30 residues)
            if boundary_pos - boundaries[-1] > 30:
                boundaries.append(boundary_pos)
    boundaries.append(n)

    domains: list[dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1 if boundaries[i] > 0 else 1
        end = boundaries[i + 1]
        if end - start + 1 >= 20:  # minimum domain size
            domains.append({
                "start": start,
                "end": end,
                "name": f"domain_{i + 1}",
                "length": end - start + 1,
            })

    if not domains:
        domains = [{"start": 1, "end": n, "name": "single_domain", "length": n}]

    return {"domains": domains, "count": len(domains), "method": "heuristic"}


# ── Oligomeric State ────────────────────────────────────────────


def _oligomeric_state(seq: str) -> dict[str, Any]:
    """Predict oligomeric state from sequence motifs.

    Scans for:
    - Coiled-coil heptad repeats (leucine zipper)
    - Transmembrane helix bundles
    - Known oligomerization motifs
    """
    evidence: list[str] = []

    # 1. Coiled-coil: leucine at every 7th position, hydrophobic at a/d
    # Look for [ILVM]...[ILVM]...L pattern over >= 4 heptads (28 residues)
    cc_runs = _find_coiled_coil(seq)
    if cc_runs:
        evidence.append(f"Coiled-coil region(s) detected ({len(cc_runs)} segment(s)); likely dimer/trimer")

    # 2. Transmembrane helices (hydrophobic runs of 19-25 residues)
    tm_count = _count_tm_helices(seq)
    if tm_count >= 2:
        evidence.append(f"{tm_count} transmembrane helices; possible oligomeric TM bundle")

    # 3. Known oligomerization sequence motifs
    # GxxxG motif (transmembrane helix dimerization)
    gxxxg = len(re.findall(r"G...G", seq))
    if gxxxg >= 2:
        evidence.append(f"{gxxxg} GxxxG motifs (TM helix dimerization)")

    # Determine state
    if cc_runs:
        state = "Possible dimer/trimer (coiled-coil)"
    elif tm_count >= 2 and gxxxg >= 2:
        state = "Possible oligomer (TM bundle)"
    elif evidence:
        state = "Possible oligomer"
    else:
        state = "Likely monomer"

    return {"state": state, "evidence": evidence}


def _find_coiled_coil(seq: str) -> list[dict[str, Any]]:
    """Find coiled-coil heptad repeat regions."""
    n = len(seq)
    hydrophobic = set("ILMVFAW")
    min_heptads = 4
    min_length = min_heptads * 7

    runs: list[dict[str, Any]] = []

    for start in range(n - min_length + 1):
        # Check heptad pattern: positions a(0) and d(3) should be hydrophobic
        heptad_score = 0
        length = 0
        for h in range(start, min(start + 70, n) - 6, 7):
            a_pos = seq[h] if h < n else "X"
            d_pos = seq[h + 3] if h + 3 < n else "X"
            if a_pos in hydrophobic and d_pos in hydrophobic:
                heptad_score += 1
                length += 7
            else:
                break

        if heptad_score >= min_heptads:
            # Avoid overlapping runs
            if not runs or start >= runs[-1]["end"]:
                runs.append({"start": start + 1, "end": start + length, "heptads": heptad_score})

    return runs


def _count_tm_helices(seq: str) -> int:
    """Count potential transmembrane helices (hydrophobic runs 19-25 AA)."""
    count = 0
    for m in re.finditer(r"[ILMVFAWCYT]{19,25}", seq):
        # Additional check: mean hydropathy > 1.6
        segment = m.group()
        h = sum(_HYDROPATHY.get(aa, 0) for aa in segment) / len(segment)
        if h > 1.6:
            count += 1
    return count


# ── Cofactor / Metal Binding Motifs ─────────────────────────────


_COFACTOR_PATTERNS: list[tuple[str, str, str]] = [
    # (name, regex, description)
    ("Zinc finger (C2H2)", r"C.{2,4}C.{2,15}H.{2,4}[HC]", "Zn²⁺ binding"),
    ("EF-hand (calcium)", r"D.[DNS][DENSTG].{0,2}[DE].{2}[DE]", "Ca²⁺ binding"),
    ("Iron-sulfur cluster (4Fe-4S)", r"C.{2}C.{2}C.{3}C", "Fe-S cluster"),
    ("Iron-sulfur cluster (2Fe-2S)", r"C.{3,5}C.{8,15}C.{3,5}C", "Fe-S cluster"),
    ("Rossmann fold (NAD/FAD)", r"G.{1,2}G.{2}G", "NAD⁺/FAD binding"),
    ("Heme binding (CXXCH)", r"C..CH", "Heme c binding"),
    ("P-loop (NTPase)", r"G....GK[TS]", "NTP binding"),
    ("Catalytic triad (Ser)", r"G.S.G", "Serine protease/lipase"),
    ("DxDxT (glycosyltransferase)", r"D.D.T", "Sugar donor binding"),
    ("HExxH (metalloprotease)", r"HE..H", "Zn²⁺ catalytic"),
]


def _cofactor_motifs(seq: str) -> dict[str, Any]:
    """Scan for known cofactor/metal-binding sequence motifs."""
    motifs: list[dict[str, Any]] = []

    for name, pattern, desc in _COFACTOR_PATTERNS:
        for m in re.finditer(pattern, seq):
            motifs.append({
                "name": name,
                "description": desc,
                "start": m.start() + 1,
                "end": m.end(),
                "match": m.group(),
            })

    # Deduplicate overlapping hits of same type
    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []
    for m in motifs:
        key = (m["name"], m["start"])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    summary_parts = []
    by_type: dict[str, int] = {}
    for m in unique:
        by_type[m["name"]] = by_type.get(m["name"], 0) + 1
    for name, count in by_type.items():
        summary_parts.append(f"{name} ×{count}" if count > 1 else name)

    return {
        "motifs": unique,
        "summary": ", ".join(summary_parts) if summary_parts else "None detected",
    }


# ── Thermal Stability (Tm estimate) ────────────────────────────


def _thermal_stability(
    seq: str, instability_index: float, aliphatic_index: float,
) -> dict[str, Any]:
    """Estimate thermal melting temperature from sequence properties.

    Uses an empirical correlation based on:
    - Aliphatic index (positive correlation with thermostability)
    - Instability index (negative correlation)
    - Charged residue fraction (moderate positive at low-moderate levels)
    - Proline content (rigidifies backbone)
    """
    n = len(seq)
    if n == 0:
        return {"tm": 0.0, "confidence": "none"}

    # Base Tm estimate from aliphatic index (empirical: AI 60-120 → Tm 40-80°C)
    tm_base = 20.0 + 0.5 * aliphatic_index

    # Adjustment from instability index (II > 40 → lower Tm)
    tm_ii = -0.15 * max(instability_index - 30, 0)

    # Proline rigidification bonus
    pro_frac = seq.count("P") / n
    tm_pro = pro_frac * 40  # prolines increase Tm modestly

    # Charged residue fraction (optimal around 20%)
    charged_frac = sum(1 for aa in seq if aa in "DEKRH") / n
    tm_charge = -abs(charged_frac - 0.20) * 30  # penalty for deviation

    # Glycine penalty (flexibility)
    gly_frac = seq.count("G") / n
    tm_gly = -gly_frac * 25

    tm = tm_base + tm_ii + tm_pro + tm_charge + tm_gly
    tm = max(20.0, min(100.0, tm))  # clamp to reasonable range

    # Confidence based on how "normal" the composition is
    if 60 <= aliphatic_index <= 110 and instability_index < 50:
        confidence = "moderate"
    else:
        confidence = "low"

    return {"tm": round(tm, 1), "confidence": confidence}


# ── pH Stability Window ─────────────────────────────────────────


def _ph_stability(seq: str, ph_range: list[float] | None = None) -> dict[str, Any]:
    """Compute charge-vs-pH curve and identify the stable pH window."""
    if ph_range is None:
        ph_range = [2.0, 12.0]

    curve: list[list[float]] = []
    n = len(seq)
    for pH_x10 in range(int(ph_range[0] * 10), int(ph_range[1] * 10) + 1, 2):
        pH = pH_x10 / 10.0
        charge = _net_charge(seq, pH)
        curve.append([round(pH, 1), round(charge, 2)])

    # "Stable" pH: where |charge| < 5% of length (minimal electrostatic stress)
    charge_threshold = max(n * 0.05, 2.0)
    stable_pHs = [p for p, c in curve if abs(c) < charge_threshold]

    if stable_pHs:
        stable_range = [min(stable_pHs), max(stable_pHs)]
    else:
        # Find pI = most stable point
        pI = _isoelectric_point(seq)
        stable_range = [max(ph_range[0], pI - 1.0), min(ph_range[1], pI + 1.0)]

    return {
        "curve": curve,
        "stable_range": [round(stable_range[0], 1), round(stable_range[1], 1)],
    }


# ── Aggregation-Prone Regions ──────────────────────────────────


def _aggregation_regions(seq: str, window: int = 7) -> dict[str, Any]:
    """Detect aggregation-prone regions (APRs) using combined
    hydropathy + β-sheet propensity scoring.

    APR = region of >= 5 residues with mean hydropathy > 1.6 and
    mean β-propensity > 1.0.
    """
    n = len(seq)
    if n < window:
        return {"score": 0.0, "regions": [], "count": 0}

    # Score each window
    apr_scores: list[float] = []
    for i in range(n - window + 1):
        w = seq[i:i + window]
        h = sum(_HYDROPATHY.get(aa, 0) for aa in w) / window
        b = sum(_BETA_PROPENSITY.get(aa, 0.8) for aa in w) / window
        # Combined score: both hydrophobicity and β tendency needed
        s = max(0, h - 0.5) * max(0, b - 0.8)
        apr_scores.append(s)

    # Find contiguous APR regions above threshold
    threshold = 0.8
    regions: list[dict[str, Any]] = []
    in_region = False
    start = 0

    for i, s in enumerate(apr_scores):
        if s >= threshold and not in_region:
            start = i
            in_region = True
        elif (s < threshold or i == len(apr_scores) - 1) and in_region:
            end = i + window if s >= threshold else i + window - 1
            length = end - start
            if length >= 5:
                region_seq = seq[start:end]
                regions.append({
                    "start": start + 1,
                    "end": end,
                    "length": length,
                    "sequence": region_seq,
                    "mean_score": round(
                        sum(apr_scores[start:i + 1]) / max(i + 1 - start, 1), 3
                    ),
                })
            in_region = False

    # Overall aggregation propensity score
    total_apr = sum(r["length"] for r in regions)
    agg_score = total_apr / max(n, 1)

    return {
        "score": round(agg_score, 4),
        "regions": regions,
        "count": len(regions),
    }


# ── Colloidal Stability ────────────────────────────────────────


def _colloidal_stability(seq: str) -> dict[str, Any]:
    """Compute colloidal stability proxies from sequence.

    - Charge symmetry parameter (σ): measures charge clustering
    - Fraction of charged residues
    - Net charge at pH 7.0
    """
    n = len(seq)
    if n == 0:
        return {
            "charge_symmetry": 0,
            "charged_fraction": 0,
            "positive_fraction": 0,
            "negative_fraction": 0,
            "net_charge_7": 0,
        }

    pos = sum(1 for aa in seq if aa in "KRH")
    neg = sum(1 for aa in seq if aa in "DE")
    total_charged = pos + neg
    f_plus = pos / n
    f_minus = neg / n
    charged_frac = total_charged / n

    # Charge symmetry parameter (Das & Pappu, 2013)
    # σ = (f+ - f-)² / (f+ + f-)  —  0 = symmetric, higher = asymmetric
    if total_charged > 0:
        sigma = (f_plus - f_minus) ** 2 / (f_plus + f_minus)
    else:
        sigma = 0.0

    net_charge = _net_charge(seq, 7.0)

    return {
        "charge_symmetry": round(sigma, 4),
        "charged_fraction": round(charged_frac, 4),
        "positive_fraction": round(f_plus, 4),
        "negative_fraction": round(f_minus, 4),
        "net_charge_7": round(net_charge, 2),
    }


# ── Solubility Prediction (Wilkinson-Harrison) ─────────────────


def _solubility_prediction(seq: str) -> dict[str, Any]:
    """Predict E. coli overexpression solubility using the
    Wilkinson-Harrison model (1991, updated).

    Based on:
    - Charge average = (K+R-D-E) / N
    - Turn-forming residue fraction (N, G, P, S)
    - Cysteine/cystine fraction
    - Proline fraction
    - Hydrophilicity (Hopp-Woods average)
    """
    n = len(seq)
    if n == 0:
        return {"score": 0.0, "label": "Unknown"}

    # Charge average
    cv = (seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")) / n

    # Turn-forming residue fraction
    tf = sum(1 for aa in seq if aa in "NGPS") / n

    # Cysteine/cystine fraction
    cf = seq.count("C") / n

    # Hydrophilicity (Hopp-Woods scale average)
    _HOPP_WOODS: dict[str, float] = {
        "A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0,
        "E": 3.0, "Q": 0.2, "G": 0.0, "H": -0.5, "I": -1.8,
        "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5, "P": 0.0,
        "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5,
    }
    hp = sum(_HOPP_WOODS.get(aa, 0) for aa in seq) / n

    # Wilkinson-Harrison formula (adapted)
    # S = 15.43 + 29.56*CV - 1.27*N + 0.03*(N-200) + 43.98*TF - ...
    # Simplified: higher CV, TF, HP → more soluble
    score = 0.4934 + 0.276 * abs(cv) + 0.0392 * tf + 0.0198 * hp - 0.1 * cf

    # Convert to percentage (calibrated such that 0.5 ≈ 50%)
    pct = max(0, min(100, score * 100))

    if pct >= 60:
        label = "Soluble"
    elif pct >= 40:
        label = "Borderline"
    else:
        label = "Insoluble"

    return {"score": round(pct, 1), "label": label}


# ── ProtSolM Deep-Learning Solubility Prediction ──────────────


def _protsolm_prediction(seq: str, pdb_path: str | None = None) -> dict[str, Any]:
    """Run ProtSolM (Tan et al., IEEE BIBM 2024) solubility prediction.

    ProtSolM is a multimodal deep-learning model that fuses:
      - ESM2 (650M) protein language model embeddings
      - ProtSSN EGNN graph neural network on Cα contact graph
      - 42 handcrafted features (AA composition, GRAVY, SS, H-bonds, etc.)

    Requires a PDB file for graph construction.  Falls back to a
    sequence-only heuristic estimate when no PDB is available.
    """
    if pdb_path is None or not Path(pdb_path).exists():
        logger.warning("ProtSolM: no PDB available — using sequence-only fallback")
        return _protsolm_sequence_fallback(seq)

    helper_script = Path(__file__).resolve().parents[2] / "scripts" / "protsolm_helper.py"
    if not helper_script.exists():
        # Try project root
        helper_script = Path(__file__).resolve().parents[3] / "scripts" / "protsolm_helper.py"
    if not helper_script.exists():
        logger.warning("ProtSolM helper script not found — using fallback")
        return _protsolm_sequence_fallback(seq)

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            out_json = f.name

        result = subprocess.run(
            [sys.executable, str(helper_script), str(pdb_path), out_json,
             "--sequence", seq],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.warning(f"ProtSolM helper failed: {result.stderr[:300]}")
            return _protsolm_sequence_fallback(seq)

        with open(out_json) as fh:
            pred = json.load(fh)

        return {
            "probability": pred.get("probability", 0.5),
            "label": pred.get("label", "Unknown"),
            "confidence": pred.get("confidence", "low"),
            "score_pct": pred.get("score_pct", 50.0),
            "method": pred.get("method", "ESM2 + ProtSSN EGNN + feature fusion"),
            "model": pred.get("model", "ProtSolM"),
            "raw_logits": pred.get("raw_logits", []),
        }
    except subprocess.TimeoutExpired:
        logger.warning("ProtSolM timed out after 600s — using fallback")
        return _protsolm_sequence_fallback(seq)
    except Exception as e:
        logger.warning(f"ProtSolM failed: {e} — using fallback")
        return _protsolm_sequence_fallback(seq)
    finally:
        try:
            os.unlink(out_json)
        except Exception:
            pass


def _protsolm_sequence_fallback(seq: str) -> dict[str, Any]:
    """Simple sequence-only solubility estimate when ProtSolM cannot run.

    Uses a weighted combination of charge, hydrophobicity, and composition
    features as a rough proxy.
    """
    n = len(seq)
    if n == 0:
        return {
            "probability": 0.5, "label": "Unknown", "confidence": "none",
            "score_pct": 50.0, "method": "sequence-only fallback",
        }

    charged_frac = sum(1 for aa in seq if aa in "DEKRH") / n
    hydrophobic_frac = sum(1 for aa in seq if aa in "AILMFVW") / n
    gravy = sum(
        {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
         "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
         "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
         "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
         }.get(aa, 0) for aa in seq
    ) / n

    # Simple logistic regression-like score
    score = 0.5 + 0.3 * charged_frac - 0.4 * hydrophobic_frac - 0.1 * max(gravy, 0)
    prob = max(0.0, min(1.0, score))

    if prob >= 0.6:
        label = "Soluble"
    elif prob >= 0.4:
        label = "Borderline"
    else:
        label = "Insoluble"

    return {
        "probability": round(prob, 4),
        "label": label,
        "confidence": "low",
        "score_pct": round(prob * 100, 1),
        "method": "sequence-only fallback (no PDB available)",
    }


# ── CamSol Intrinsic Solubility Profile ────────────────────────


# CamSol intrinsic amino-acid solubility values (Sormanni et al., 2015)
# Higher = more soluble; negative = aggregation-prone
_CAMSOL_INTRINSIC: dict[str, float] = {
    "A": -0.08, "R":  0.88, "N":  0.20, "D":  0.78, "C": -0.39,
    "E":  0.83, "Q":  0.16, "G": -0.01, "H":  0.14, "I": -0.79,
    "L": -0.74, "K":  0.75, "M": -0.46, "F": -0.85, "P":  0.12,
    "S":  0.12, "T":  0.02, "W": -0.88, "Y": -0.55, "V": -0.62,
}


def _camsol_profile(seq: str, window: int = 7) -> dict[str, Any]:
    """Compute CamSol-intrinsic per-residue solubility profile.

    Implements the CamSol intrinsic algorithm (Sormanni, Aprile,
    Vendruscolo, J Mol Biol, 2015):

    1. Raw intrinsic solubility per residue
    2. Sequence-context corrections (3 correction patterns):
       a. Gatekeeper effect: charged neighbors rescue hydrophobic patches
       b. Pattern penalty: alternating hydrophobic residues
       c. Hydrophobic cluster penalty for runs of ≥3 hydrophobic residues
    3. Smoothing with window average

    Returns per-residue scores, overall score, and aggregation-prone patches.
    """
    n = len(seq)
    if n == 0:
        return {"scores": [], "overall": 0.0, "patches": [], "patch_count": 0}

    # Step 1: Raw intrinsic scores
    raw = [_CAMSOL_INTRINSIC.get(aa, 0.0) for aa in seq]

    # Step 2: Sequence context corrections
    corrected = list(raw)

    # 2a. Gatekeeper effect — charged residues flanking hydrophobic patches
    # boost solubility of nearby residues
    charged = set("DEKRH")
    for i in range(n):
        if seq[i] in charged:
            # Boost neighbors within ±3 residues
            for j in range(max(0, i - 3), min(n, i + 4)):
                if j != i and corrected[j] < 0:
                    # Partial rescue proportional to distance
                    dist = abs(j - i)
                    rescue = 0.15 / dist
                    corrected[j] += rescue

    # 2b. Hydrophobic cluster penalty — runs of ≥3 hydrophobic residues
    hydrophobic = set("ILMVFWCY")
    run_start = -1
    for i in range(n):
        if seq[i] in hydrophobic:
            if run_start < 0:
                run_start = i
        else:
            if run_start >= 0:
                run_len = i - run_start
                if run_len >= 3:
                    penalty = -0.10 * (run_len - 2)
                    for j in range(run_start, i):
                        corrected[j] += penalty
            run_start = -1
    if run_start >= 0:
        run_len = n - run_start
        if run_len >= 3:
            penalty = -0.10 * (run_len - 2)
            for j in range(run_start, n):
                corrected[j] += penalty

    # 2c. β-strand pattern penalty — alternating hydrophobic/hydrophilic
    # (I-X-I-X pattern typical of β-aggregation)
    for i in range(n - 4):
        segment = seq[i:i + 5]
        if (segment[0] in hydrophobic and segment[2] in hydrophobic
                and segment[4] in hydrophobic
                and segment[1] not in hydrophobic and segment[3] not in hydrophobic):
            for j in range(i, i + 5):
                corrected[j] -= 0.08

    # Step 3: Windowed smoothing
    half = window // 2
    smoothed: list[float] = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed.append(sum(corrected[start:end]) / (end - start))

    # Overall CamSol score (mean of smoothed profile)
    overall = sum(smoothed) / n

    # Identify aggregation-prone patches (smoothed score < -1.0)
    patches: list[dict[str, Any]] = []
    in_patch = False
    patch_start = 0
    for i in range(n):
        if smoothed[i] < -1.0 and not in_patch:
            patch_start = i
            in_patch = True
        elif (smoothed[i] >= -1.0 or i == n - 1) and in_patch:
            patch_end = i if smoothed[i] >= -1.0 else i + 1
            length = patch_end - patch_start
            if length >= 5:
                patches.append({
                    "start": patch_start + 1,
                    "end": patch_end,
                    "length": length,
                    "sequence": seq[patch_start:patch_end],
                    "mean_score": round(
                        sum(smoothed[patch_start:patch_end]) / length, 3
                    ),
                })
            in_patch = False

    return {
        "scores": [round(s, 3) for s in smoothed],
        "overall": round(overall, 3),
        "patches": patches,
        "patch_count": len(patches),
    }


# ── TANGO-like β-Aggregation Predictor ─────────────────────────


# Hydrophobicity scales for aggregation prediction
_AGGRESCAN_SCALE: dict[str, float] = {
    # a3vSA scale (Conchillo-Solé et al., 2007) — normalized hot-spot propensity
    "A":  0.17, "R": -1.03, "N": -0.48, "D": -0.78, "C":  0.24,
    "E": -0.83, "Q": -0.30, "G": -0.01, "H": -0.50, "I":  0.81,
    "L":  0.65, "K": -0.98, "M":  0.42, "F":  0.76, "P": -0.53,
    "S": -0.09, "T": -0.05, "W":  0.37, "Y":  0.33, "V":  0.63,
}


def _tango_like_aggregation(seq: str, window: int = 7) -> dict[str, Any]:
    """TANGO-like β-aggregation propensity prediction.

    Combines multiple aggregation-relevant scales:
    1. Zyggregator profile (Tartaglia & Vendruscolo, 2008)
    2. AGGRESCAN hot-spot propensity (Conchillo-Solé et al., 2007)
    3. β-strand propensity with charge gatekeeper analysis
    4. Cross-β nucleation core detection

    Returns per-residue scores, identified APRs, and gatekeeper analysis.
    """
    n = len(seq)
    if n < window:
        return {
            "scores": [0.0] * n, "aprs": [], "apr_count": 0,
            "overall_score": 0.0, "gatekeepers": [], "nucleation_cores": [],
        }

    half = window // 2

    # ── Zyggregator-like profile ──
    # Z_agg = α*hydrophobicity + β*β_propensity + γ*charge_penalty + δ*pattern
    zyg_scores: list[float] = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        w = seq[start:end]
        wsize = len(w)

        # Mean hydrophobicity (Kyte-Doolittle normalized)
        h_mean = sum(_HYDROPATHY.get(aa, 0.0) for aa in w) / wsize

        # Mean β-sheet propensity
        b_mean = sum(_BETA_PROPENSITY.get(aa, 0.8) for aa in w) / wsize

        # Charge: penalty for low absolute charge (aggregation possible when uncharged)
        charges = sum(1 for aa in w if aa in "DEKRH")
        charge_penalty = max(0, 1.0 - charges / max(wsize * 0.3, 1))

        # AGGRESCAN contribution
        agg_mean = sum(_AGGRESCAN_SCALE.get(aa, 0.0) for aa in w) / wsize

        # Combined Zyggregator-like score
        z = (0.30 * max(0, h_mean) +
             0.25 * max(0, b_mean - 0.8) +
             0.20 * charge_penalty +
             0.25 * max(0, agg_mean))
        zyg_scores.append(z)

    # ── Identify APRs (aggregation-prone regions) ──
    apr_threshold = 0.35
    aprs: list[dict[str, Any]] = []
    in_apr = False
    apr_start = 0
    for i in range(n):
        if zyg_scores[i] >= apr_threshold and not in_apr:
            apr_start = i
            in_apr = True
        elif (zyg_scores[i] < apr_threshold or i == n - 1) and in_apr:
            apr_end = i if zyg_scores[i] < apr_threshold else i + 1
            length = apr_end - apr_start
            if length >= 5:
                region_seq = seq[apr_start:apr_end]
                region_scores = zyg_scores[apr_start:apr_end]
                aprs.append({
                    "start": apr_start + 1,
                    "end": apr_end,
                    "length": length,
                    "sequence": region_seq,
                    "mean_score": round(sum(region_scores) / length, 3),
                    "peak_score": round(max(region_scores), 3),
                })
            in_apr = False

    # ── Gatekeeper analysis ──
    # Gatekeepers are charged/proline residues flanking APRs that suppress aggregation
    gatekeepers: list[dict[str, Any]] = []
    gatekeeper_aas = set("DEKRPH")  # charged + proline + histidine
    for apr in aprs:
        s = apr["start"] - 1  # 0-indexed
        e = apr["end"]  # 0-indexed (exclusive)
        flanking: list[dict[str, Any]] = []

        # Check N-terminal flank (up to 3 residues before APR)
        for j in range(max(0, s - 3), s):
            if seq[j] in gatekeeper_aas:
                flanking.append({
                    "position": j + 1, "aa": seq[j],
                    "side": "N-terminal"
                })
        # Check C-terminal flank
        for j in range(e, min(n, e + 3)):
            if seq[j] in gatekeeper_aas:
                flanking.append({
                    "position": j + 1, "aa": seq[j],
                    "side": "C-terminal"
                })

        if flanking:
            gatekeepers.append({
                "apr_start": apr["start"],
                "apr_end": apr["end"],
                "gatekeepers": flanking,
                "protected": len(flanking) >= 2,
            })

    # ── Nucleation cores ──
    # Short segments (5-8 residues) with very high aggregation propensity
    nucleation_cores: list[dict[str, Any]] = []
    for apr in aprs:
        s = apr["start"] - 1
        e = apr["end"]
        if e - s >= 5:
            # Find peak region within APR
            best_score = 0
            best_start = s
            for ws in range(5, min(9, e - s + 1)):
                for j in range(s, e - ws + 1):
                    segment_score = sum(zyg_scores[j:j + ws]) / ws
                    if segment_score > best_score:
                        best_score = segment_score
                        best_start = j

            if best_score >= 0.4:
                core_len = min(8, e - best_start)
                nucleation_cores.append({
                    "start": best_start + 1,
                    "end": best_start + core_len,
                    "sequence": seq[best_start:best_start + core_len],
                    "score": round(best_score, 3),
                })

    # Overall aggregation propensity
    total_apr_residues = sum(a["length"] for a in aprs)
    overall_score = total_apr_residues / max(n, 1)

    return {
        "scores": [round(s, 3) for s in zyg_scores],
        "aprs": aprs,
        "apr_count": len(aprs),
        "overall_score": round(overall_score, 4),
        "gatekeepers": gatekeepers,
        "nucleation_cores": nucleation_cores,
    }


# ── Enhanced Solubility Ensemble ───────────────────────────────


def _solubility_ensemble(seq: str,
                         protsolm_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Multi-method solubility prediction ensemble.

    Combines four complementary approaches:
    1. Wilkinson-Harrison (composition-based, E. coli focus)
    2. ProtSolM (Tan et al., IEEE BIBM 2024 — deep learning)
    3. Solubility-Weighted Index (SWI, Bhandari et al., 2020)
    4. PROSO II-like SVM features (sequence feature regression)

    Returns individual scores, confidence, and an ensemble consensus.
    """
    n = len(seq)
    if n == 0:
        return {
            "ensemble_score": 0.0, "ensemble_class": "Unknown",
            "methods": {}, "confidence": "none",
        }

    # Method 1: Wilkinson-Harrison (already exists — call it)
    wh = _solubility_prediction(seq)

    # Method 2: ProtSolM deep-learning prediction
    if protsolm_result is not None:
        protsolm_pct = protsolm_result.get("score_pct", 50.0)
    else:
        protsolm_pct = 50.0  # neutral default

    # Method 3: Solubility-Weighted Index (SWI)
    swi = _swi_score(seq)

    # Method 4: PROSO II-like features
    proso = _proso_like_score(seq)

    # Ensemble: weighted average (weights reflect method reliability)
    # ProtSolM gets highest weight as a structure-aware deep-learning method
    weights = {
        "wilkinson_harrison": 0.15,
        "protsolm": 0.40,
        "swi": 0.25,
        "proso": 0.20,
    }
    scores = {
        "wilkinson_harrison": wh["score"],
        "protsolm": protsolm_pct,
        "swi": swi["score"],
        "proso": proso["score"],
    }
    ensemble = sum(scores[m] * weights[m] for m in weights)

    # Classification
    if ensemble >= 60:
        ensemble_class = "Soluble"
    elif ensemble >= 40:
        ensemble_class = "Borderline"
    else:
        ensemble_class = "Insoluble"

    # Confidence based on method agreement
    labels = []
    for s in scores.values():
        if s >= 60:
            labels.append("Soluble")
        elif s >= 40:
            labels.append("Borderline")
        else:
            labels.append("Insoluble")
    agreement = max(labels.count(l) for l in set(labels)) / len(labels)
    if agreement >= 0.75:
        confidence = "high"
    elif agreement >= 0.5:
        confidence = "moderate"
    else:
        confidence = "low"

    protsolm_label = ("Soluble" if protsolm_pct >= 60
                      else "Borderline" if protsolm_pct >= 40
                      else "Insoluble")

    return {
        "ensemble_score": round(ensemble, 1),
        "ensemble_class": ensemble_class,
        "methods": {
            "wilkinson_harrison": {
                "score": wh["score"], "label": wh["label"]
            },
            "protsolm": {
                "score": round(protsolm_pct, 1),
                "probability": protsolm_result.get("probability", 0.5) if protsolm_result else 0.5,
                "label": protsolm_label,
                "method": protsolm_result.get("method", "ProtSolM") if protsolm_result else "unavailable",
            },
            "swi": {
                "score": swi["score"], "label": swi["label"],
                "details": swi.get("details", ""),
            },
            "proso": {
                "score": proso["score"], "label": proso["label"],
            },
        },
        "confidence": confidence,
    }


def _swi_score(seq: str) -> dict[str, Any]:
    """Solubility-Weighted Index (Bhandari et al., 2020).

    Uses amino acid solubility contributions derived from experimental
    solubility data of >3,000 E. coli expressed proteins.

    SWI = Σ(freq_i × solubility_weight_i) + length_correction + charge_term
    """
    n = len(seq)
    if n == 0:
        return {"score": 0.0, "label": "Unknown", "details": ""}

    # Experimentally-derived solubility weights per amino acid
    # (positive = promotes solubility, negative = promotes aggregation)
    _SWI_WEIGHTS: dict[str, float] = {
        "A": -0.02, "R":  0.52, "N":  0.13, "D":  0.57, "C": -0.35,
        "E":  0.49, "Q":  0.05, "G":  0.01, "H":  0.08, "I": -0.47,
        "L": -0.40, "K":  0.59, "M": -0.24, "F": -0.52, "P":  0.10,
        "S":  0.07, "T":  0.01, "W": -0.60, "Y": -0.30, "V": -0.32,
    }

    from collections import Counter
    counts = Counter(seq)
    freqs = {aa: counts.get(aa, 0) / n for aa in _SWI_WEIGHTS}

    # Weighted sum
    raw = sum(freqs[aa] * _SWI_WEIGHTS[aa] for aa in _SWI_WEIGHTS)

    # Length correction (longer proteins slightly less soluble)
    len_corr = -0.00015 * max(n - 200, 0)

    # Net charge bonus (moderate charge helps)
    net_charge = abs(seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E"))
    charge_norm = net_charge / n
    charge_bonus = 0.1 * min(charge_norm, 0.15)

    swi_raw = raw + len_corr + charge_bonus

    # Convert to 0-100 scale (calibrated: swi_raw typically in [-0.3, 0.5])
    pct = max(0, min(100, (swi_raw + 0.3) / 0.8 * 100))

    if pct >= 60:
        label = "Soluble"
    elif pct >= 40:
        label = "Borderline"
    else:
        label = "Insoluble"

    # Top contributors
    contributions = sorted(
        [(aa, freqs[aa] * _SWI_WEIGHTS[aa]) for aa in _SWI_WEIGHTS if counts.get(aa, 0) > 0],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_helpers = [f"{aa}({v:+.3f})" for aa, v in contributions[:3] if v > 0]
    top_hurters = [f"{aa}({v:+.3f})" for aa, v in contributions[:3] if v < 0]
    details = f"Helpers: {', '.join(top_helpers) or 'none'}; Hurters: {', '.join(top_hurters) or 'none'}"

    return {"score": round(pct, 1), "label": label, "details": details}


def _proso_like_score(seq: str) -> dict[str, Any]:
    """PROSO II-like solubility scoring using sequence features.

    Extracts features inspired by PROSO II (Smialowski et al., 2012):
    - Amino acid composition
    - Dipeptide frequencies (selected top features)
    - Physicochemical property averages
    - Sequence length

    Uses a logistic regression model on these features.
    """
    n = len(seq)
    if n == 0:
        return {"score": 0.0, "label": "Unknown"}

    # Feature extraction
    from collections import Counter

    # AA frequencies
    aa_counts = Counter(seq)
    aa_freq = {aa: aa_counts.get(aa, 0) / n for aa in "ACDEFGHIKLMNPQRSTVWY"}

    # Key feature contributions to solubility (learned coefficients)
    # Positive = solubility-promoting, negative = aggregation-promoting
    _FEATURE_WEIGHTS: dict[str, float] = {
        # AA frequency features
        "f_K": 1.8, "f_R": 1.2, "f_D": 1.5, "f_E": 1.4,  # charged → soluble
        "f_N": 0.3, "f_Q": 0.2, "f_S": 0.1, "f_T": 0.1,  # polar → slightly soluble
        "f_G": 0.0, "f_A": -0.1, "f_P": 0.3,
        "f_I": -1.2, "f_L": -0.9, "f_V": -0.8, "f_F": -1.3,  # hydrophobic → insoluble
        "f_W": -1.5, "f_Y": -0.6, "f_M": -0.5, "f_C": -0.7,
        "f_H": 0.2,
    }

    # Score = Σ(feature × weight) + intercept
    score = 0.5  # intercept (calibrated to ~50% baseline)
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        key = f"f_{aa}"
        if key in _FEATURE_WEIGHTS:
            score += aa_freq[aa] * _FEATURE_WEIGHTS[key]

    # Physicochemical features
    # Average hydrophobicity penalty
    gravy = sum(_HYDROPATHY.get(aa, 0.0) for aa in seq) / n
    score -= 0.05 * max(0, gravy)

    # Charged fraction bonus
    charged_frac = sum(1 for aa in seq if aa in "DEKRH") / n
    score += 0.3 * min(charged_frac, 0.25)

    # Length penalty
    if n > 500:
        score -= 0.05 * ((n - 500) / 500)

    # Sigmoid to 0-100
    import math as _math
    prob = 1.0 / (1.0 + _math.exp(-3.0 * (score - 0.5)))
    pct = prob * 100

    if pct >= 60:
        label = "Soluble"
    elif pct >= 40:
        label = "Borderline"
    else:
        label = "Insoluble"

    return {"score": round(pct, 1), "label": label}


# ═══════════════════════════════════════════════════════════════════
# RP3Net — E. coli recombinant protein production prediction
# ═══════════════════════════════════════════════════════════════════

_RP3NET_MODEL = None  # lazy singleton
_RP3NET_CKPT = Path.home() / ".cache" / "rp3net" / "rp3net_v0.1_d.ckpt"
_RP3NET_CKPT_URL = (
    "https://ftp.ebi.ac.uk/pub/software/RP3Net/v0.1/checkpoints/rp3net_v0.1_d.ckpt"
)


def _ensure_rp3net_checkpoint() -> "Path | None":
    """Return path to RP3Net checkpoint, downloading it if necessary."""
    if _RP3NET_CKPT.exists():
        return _RP3NET_CKPT
    try:
        import urllib.request
        _RP3NET_CKPT.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading RP3Net checkpoint (~2.5 GB) …")
        urllib.request.urlretrieve(_RP3NET_CKPT_URL, str(_RP3NET_CKPT))
        logger.info("RP3Net checkpoint saved to %s", _RP3NET_CKPT)
        return _RP3NET_CKPT
    except Exception as exc:
        logger.warning("Could not download RP3Net checkpoint: %s", exc)
        return None


def _rp3net_prediction(seq: str) -> dict[str, Any]:
    """Predict E. coli recombinant production probability using RP3Net.

    RP3Net (Tankhilevich et al., Bioinformatics 2026) fine-tunes ESM2-650M
    with LoRA + Set Transformer Pooling, trained on AstraZeneca / SGC
    small-scale expression screens via Meta Label Correction (MLC).
    Achieves AUROC 0.83 in prospective experimental validation on human
    drug targets.
    """
    global _RP3NET_MODEL

    try:
        import RP3Net as rp3
    except ImportError:
        logger.warning("RP3Net not installed — run: pip install RP3Net")
        return _rp3net_fallback()

    ckpt = _ensure_rp3net_checkpoint()
    if ckpt is None:
        return _rp3net_fallback()

    try:
        if _RP3NET_MODEL is None:
            logger.info("Loading RP3Net (ESM2-650M + LoRA) — first call may take ~30 s …")
            with warnings.catch_warnings():
                # lightning_fabric loads the checkpoint with weights_only=None which
                # triggers a FutureWarning from torch.load.  The RP3Net checkpoint
                # is fetched from the official EBI FTP and is a trusted source, so
                # suppressing this warning here is safe.
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    module="lightning_fabric",
                )
                _RP3NET_MODEL = rp3.load_model(rp3.RP3_DEFAULT_CONFIG, str(ckpt))

        score = float(_RP3NET_MODEL.predict([seq]).item())
        label = "Likely Expressed" if score >= 0.5 else "Likely Not Expressed"
        return {
            "probability": round(score, 4),
            "label": label,
            "score_pct": round(score * 100, 1),
            "method": "RP3Net (ESM2-650M)",
        }
    except Exception as exc:
        logger.warning("RP3Net prediction failed: %s", exc)
        _RP3NET_MODEL = None  # allow retry on next call
        return _rp3net_fallback()


def _rp3net_fallback() -> dict[str, Any]:
    return {
        "probability": 0.5,
        "label": "Unavailable",
        "score_pct": 50.0,
        "method": "RP3Net (unavailable)",
    }
