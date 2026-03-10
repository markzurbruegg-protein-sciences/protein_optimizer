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

import logging
import math
import re
import shutil
import subprocess
import tempfile
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

    Tries IUPred3 first; falls back to FoldIndex-based heuristic.
    """
    if mode != "off" and shutil.which("iupred3"):
        result = _iupred_external(seq)
        if result is not None:
            return _parse_disorder(result["scores"], threshold, method="iupred3")

    # ── Heuristic (FoldIndex algorithm, Prilusky & Biber 2005) ──
    scores = _disorder_heuristic(seq)
    return _parse_disorder(scores, threshold, method="heuristic")


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
