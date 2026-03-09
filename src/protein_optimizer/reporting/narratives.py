"""Programmatic narrative text generation for step results.

Reads StepResult objects and produces human-readable prose summaries
explaining what each step found, what it means, and what was proposed.
"""

from __future__ import annotations

import html
from collections import Counter
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult


def generate_step_narratives(results: dict[str, StepResult]) -> list[str]:
    """Generate HTML narrative blocks for each step result.

    Returns a list of HTML strings, one per step, with human-readable
    explanations of what was found and proposed.
    """
    narratives = []
    for step_name, result in results.items():
        fn = _NARRATORS.get(step_name, _generic_narrative)
        narratives.append(fn(step_name, result))
    return narratives


# ── Per-step narrators ──────────────────────────────────────────────


def _cysteine_scan_narrative(name: str, result: StepResult) -> str:
    """Narrative for cysteine_scan step."""
    candidates = result.candidates
    wt = [c for c in candidates if c.parent_id is None]
    variants = [c for c in candidates if c.parent_id is not None]

    # Extract unique cys mutations
    cys_muts = []
    combo_variant = None
    for v in variants:
        if len(v.mutations) > 1:
            combo_variant = v
        elif len(v.mutations) == 1:
            m = v.mutations[0]
            ctx = m.metadata.get("context", "") if m.metadata else ""
            cys_muts.append({
                "position": m.position,
                "wt": m.wt,
                "mut": m.mut,
                "risk": m.score or 0.0,
                "context": ctx,
            })

    cys_muts.sort(key=lambda x: x["position"])
    n_cys = len(cys_muts)

    if n_cys == 0:
        return _wrap_narrative(
            name,
            "Cysteine Scan",
            "<p>No cysteine residues were found in the input sequence. "
            "No modifications needed for cysteine-related expression issues.</p>",
        )

    seq_len = len(wt[0].sequence) if wt else "?"

    lines = []
    lines.append(
        f"<p>The input protein ({seq_len} residues) contains "
        f"<strong>{n_cys} cysteine{'s' if n_cys != 1 else ''}</strong>. "
        f"Free cysteines in <em>E.&nbsp;coli</em>'s reducing cytoplasm can form "
        f"unwanted intermolecular disulfide bonds, leading to aggregation and inclusion "
        f"body formation. Each cysteine was scored for risk based on surface exposure "
        f"(flanking charged residues), terminal proximity, and nearby cysteine pairs.</p>"
    )

    # Build table
    lines.append('<table style="margin:0.5rem 0">')
    lines.append(
        "<thead><tr>"
        "<th>Mutation</th><th>Position</th><th>Context</th>"
        "<th>Risk</th><th>Notes</th>"
        "</tr></thead><tbody>"
    )
    for cm in cys_muts:
        risk = cm["risk"]
        risk_class = "bad" if risk >= 0.7 else "warn" if risk >= 0.5 else "good"
        notes = []
        if risk >= 0.7:
            notes.append("high risk — prioritize replacement")
        elif risk >= 0.5:
            notes.append("moderate risk")
        else:
            notes.append("lower risk")

        ctx_html = html.escape(cm["context"]) if cm["context"] else "—"
        lines.append(
            f'<tr><td class="mutations">{cm["wt"]}{cm["position"]}{cm["mut"]}</td>'
            f'<td>{cm["position"]}</td>'
            f'<td class="mutations">{ctx_html}</td>'
            f'<td class="{risk_class}">{risk:.2f}</td>'
            f'<td>{"".join(notes)}</td></tr>'
        )
    lines.append("</tbody></table>")

    # Highest risk
    highest = max(cys_muts, key=lambda x: x["risk"])
    lines.append(
        f'<p>The highest-risk cysteine is <strong>{highest["wt"]}{highest["position"]}'
        f'{highest["mut"]}</strong> (risk={highest["risk"]:.2f}). '
    )

    if combo_variant:
        lines.append(
            f"An all-cysteines-removed variant was also generated with "
            f"{len(combo_variant.mutations)} simultaneous Cys→Ser mutations.</p>"
        )
    else:
        lines.append("</p>")

    lines.append(
        f"<p><strong>Output:</strong> {len(candidates)} candidates total "
        f"(1 wild-type + {len(cys_muts)} single Cys mutants"
        f'{" + 1 all-Cys-removed variant" if combo_variant else ""}).</p>'
    )

    return _wrap_narrative(name, "Cysteine Scan", "\n".join(lines))


def _motif_scan_narrative(name: str, result: StepResult) -> str:
    """Narrative for motif_scan step."""
    candidates = result.candidates
    variants = [c for c in candidates if c.parent_id is not None]
    parents = [c for c in candidates if c.parent_id is None]

    # Collect unique motif-sourced mutations
    motif_muts: dict[str, list[dict]] = {}  # category → list of mutations
    seen = set()
    for v in variants:
        for m in v.mutations:
            if m.source_step != "motif_scan":
                continue
            key = (m.position, m.wt, m.mut)
            if key in seen:
                continue
            seen.add(key)
            cat = (m.metadata or {}).get("category", "unknown")
            motif_muts.setdefault(cat, []).append({
                "position": m.position,
                "wt": m.wt,
                "mut": m.mut,
                "risk": m.score or 0.0,
                "rationale": (m.metadata or {}).get("rationale", ""),
            })

    total_unique = sum(len(v) for v in motif_muts.values())

    if total_unique == 0:
        return _wrap_narrative(
            name,
            "Motif Scan",
            "<p>No problematic sequence motifs were detected. The sequence "
            "appears clean with respect to deamidation, oxidation, proteolytic "
            "susceptibility, and aggregation-prone regions.</p>",
        )

    # Category descriptions
    cat_descriptions = {
        "deamidation": (
            "Deamidation / Isomerization",
            "Asparagine residues followed by small amino acids (NG, NS, NH) spontaneously "
            "deamidate, converting Asn→Asp and introducing charge heterogeneity. "
            "Asp-Gly motifs can isomerize via a succinimide intermediate. "
            "Conservative fix: <strong>Asn→Gln</strong> (retains amide) or "
            "<strong>Asp→Glu</strong> (retains charge).",
        ),
        "oxidation": (
            "Oxidation",
            "Surface-exposed methionine residues are susceptible to oxidation "
            "(Met→Met-sulfoxide) during fermentation, purification, or storage, "
            "potentially reducing activity or shelf life. "
            "Conservative fix: <strong>Met→Leu</strong> (similar hydrophobicity, "
            "no sulfur).",
        ),
        "proteolysis": (
            "Proteolytic Susceptibility",
            "Dibasic motifs (KR, RR, RK, KK) are recognition sites for host "
            "proteases including OmpT and furin-like enzymes. Cleavage causes "
            "truncated product and reduced yield. "
            "Conservative fix: <strong>Lys/Arg→Gln</strong> (maintains polarity, "
            "removes basicity).",
        ),
        "aggregation": (
            "Aggregation-Prone Regions",
            "Stretches of consecutive hydrophobic residues (≥5) can nucleate "
            "aggregation through β-sheet interactions during folding. "
            "Fix: introduce a charged residue to break the hydrophobic run.",
        ),
    }

    lines = []
    lines.append(
        f"<p>The motif scanner identified <strong>{total_unique} unique problematic "
        f"motifs</strong> across {len(cat_descriptions)} categories. Each motif "
        f"was addressed with a conservative amino acid substitution designed to "
        f"eliminate the liability while preserving protein function.</p>"
    )

    for cat, muts in motif_muts.items():
        muts.sort(key=lambda x: x["position"])
        cat_title, cat_desc = cat_descriptions.get(
            cat, (cat.replace("_", " ").title(), "")
        )
        lines.append(f"<h4>{cat_title} ({len(muts)} site{'s' if len(muts) != 1 else ''})</h4>")
        if cat_desc:
            lines.append(f"<p style='color:#8b949e;font-size:0.9rem'>{cat_desc}</p>")
        lines.append('<table style="margin:0.5rem 0"><thead><tr>'
                      "<th>Mutation</th><th>Position</th><th>Risk</th>"
                      "<th>Rationale</th></tr></thead><tbody>")
        for m in muts:
            risk_class = "bad" if m["risk"] >= 0.7 else "warn" if m["risk"] >= 0.4 else "good"
            lines.append(
                f'<tr><td class="mutations">{m["wt"]}{m["position"]}{m["mut"]}</td>'
                f'<td>{m["position"]}</td>'
                f'<td class="{risk_class}">{m["risk"]:.2f}</td>'
                f'<td>{html.escape(m["rationale"])}</td></tr>'
            )
        lines.append("</tbody></table>")

    # Combinatorial note
    n_inherited = sum(
        1
        for v in variants
        for m_ in v.mutations
        if m_.source_step != "motif_scan"
    )
    if n_inherited > 0:
        lines.append(
            "<p><em>Note:</em> Each motif fix was applied on top of each upstream "
            "variant (e.g., cysteine scan outputs), producing a combinatorial set.</p>"
        )

    lines.append(
        f"<p><strong>Output:</strong> {len(candidates)} candidates total "
        f"({len(parents)} inherited parent{'s' if len(parents) != 1 else ''} "
        f"+ {len(variants)} new variants).</p>"
    )

    return _wrap_narrative(name, "Motif Scan", "\n".join(lines))


def _sequence_complexity_narrative(name: str, result: StepResult) -> str:
    """Narrative for sequence_complexity step."""
    candidates = result.candidates

    # Gather all flags
    all_flags: list[dict] = []
    for c in candidates[:1]:  # flags are about the sequence, just check one representative
        all_flags = c.metadata.get("complexity_flags", [])
        break

    if not all_flags:
        return _wrap_narrative(
            name,
            "Sequence Complexity",
            "<p>No sequence complexity issues were detected. The protein has no "
            "homopolymeric runs, proline stretches, low-entropy regions, or "
            "extreme charge clusters that would impair <em>E.&nbsp;coli</em> expression.</p>"
            f"<p><strong>Output:</strong> All {len(candidates)} candidates passed "
            "complexity checks unchanged.</p>",
        )

    # Group by category
    by_cat: dict[str, list[dict]] = {}
    for f in all_flags:
        by_cat.setdefault(f["category"], []).append(f)

    cat_names = {
        "homopolymer": "Homopolymeric Runs",
        "proline_run": "Proline Stretches",
        "proline_stall": "Proline Stalling Motifs",
        "charge_cluster": "Charge Clusters",
        "low_complexity": "Low-Complexity Regions",
    }

    lines = []
    lines.append(
        f"<p>The sequence complexity checker flagged <strong>{len(all_flags)} "
        f"issue{'s' if len(all_flags) != 1 else ''}</strong> that may affect "
        f"<em>E.&nbsp;coli</em> expression or protein behavior:</p>"
    )

    for cat, flags in by_cat.items():
        cat_title = cat_names.get(cat, cat.replace("_", " ").title())
        lines.append(f"<h4>{cat_title} ({len(flags)} hit{'s' if len(flags) != 1 else ''})</h4>")
        lines.append('<table style="margin:0.5rem 0"><thead><tr>'
                      "<th>Position</th><th>Residues</th><th>Risk</th>"
                      "<th>Description</th></tr></thead><tbody>")
        for f in flags:
            risk_class = "bad" if f["risk"] >= 0.7 else "warn" if f["risk"] >= 0.3 else "good"
            lines.append(
                f'<tr><td>{f["position"]}</td>'
                f'<td class="mutations">{html.escape(f["residues"])}</td>'
                f'<td class="{risk_class}">{f["risk"]:.2f}</td>'
                f'<td>{html.escape(f["description"])}</td></tr>'
            )
        lines.append("</tbody></table>")

    lines.append(
        f"<p><em>Note:</em> This step is diagnostic — it annotates candidates with "
        f"complexity flags but does not propose mutations. These flags inform "
        f"downstream scoring and candidate prioritization.</p>"
        f"<p><strong>Output:</strong> {len(candidates)} candidates passed through, "
        f"annotated with complexity metadata.</p>"
    )

    return _wrap_narrative(name, "Sequence Complexity", "\n".join(lines))


def _find_homologs_narrative(name: str, result: StepResult) -> str:
    """Narrative for find_homologs step."""
    meta = result.metadata or {}
    n_hits = meta.get("n_homologs", "?")
    method = meta.get("search_method", "colabfold")
    db = meta.get("database", "colabfold_api")

    if method == "colabfold":
        method_desc = (
            "the <strong>ColabFold MMseqs2 API</strong> server, which searches "
            "pre-indexed UniRef30 and environmental sequence databases"
        )
    else:
        method_desc = f"local <strong>MMseqs2</strong> against <strong>{db}</strong>"

    return _wrap_narrative(
        name,
        "Homolog Search",
        f"<p>Searched for homologous sequences using {method_desc}. "
        f"Found <strong>{n_hits}</strong> unique homologs. "
        f"These were aligned together with the query sequence using MAFFT to build "
        f"a multiple sequence alignment (MSA) of {n_hits}+1 sequences.</p>"
        f"<p>The MSA serves as the evolutionary foundation for the next steps: "
        f"<strong>consensus design</strong> identifies positions where the query "
        f"deviates from the evolutionary consensus, and <strong>PSSM analysis</strong> "
        f"scores each proposed mutation against evolutionary substitution frequencies. "
        f"The raw homolog sequences are also stored for retrieval-augmented protein "
        f"language model scoring in Tier 4.</p>"
        f"<p><strong>Output:</strong> {len(result.candidates)} candidates with "
        f"MSA metadata attached.</p>",
    )


def _consensus_design_narrative(name: str, result: StepResult) -> str:
    """Narrative for consensus_design step."""
    candidates = result.candidates
    variants = [c for c in candidates if c.parent_id is not None]
    parents = [c for c in candidates if c.parent_id is None]

    # Collect unique consensus mutations with scores
    mut_list: list[dict] = []
    seen = set()
    combo_variant = None
    for v in variants:
        cons_muts = [m for m in v.mutations if m.source_step == "consensus_design"]
        if len(cons_muts) > 1:
            combo_variant = v
            continue
        for m in cons_muts:
            key = (m.position, m.wt, m.mut)
            if key in seen:
                continue
            seen.add(key)
            mut_list.append({
                "position": m.position,
                "wt": m.wt,
                "mut": m.mut,
                "conservation": m.score or 0.0,
            })

    mut_list.sort(key=lambda x: x["conservation"], reverse=True)

    if not mut_list:
        return _wrap_narrative(
            name,
            "Consensus Design",
            "<p>No consensus mutations were identified above the conservation "
            "threshold. The input sequence already matches the evolutionary "
            "consensus at all positions.</p>"
            f"<p><strong>Output:</strong> {len(candidates)} candidates.</p>",
        )

    lines = []
    lines.append(
        f"<p>Back-to-consensus analysis compared the input sequence to the "
        f"evolutionary consensus derived from the MSA. At each alignment column, "
        f"the most frequent amino acid was determined. Positions where the input "
        f"carries a non-consensus residue — and the consensus residue appears in "
        f"&gt;50% of homologs — are candidates for <strong>back-to-consensus "
        f"mutations</strong>.</p>"
    )
    lines.append(
        f"<p>These mutations are among the safest engineering changes because "
        f"they restore residues maintained by natural selection across hundreds of "
        f"homologs. Each typically contributes +0.5–2.0&nbsp;°C to thermal stability.</p>"
    )
    lines.append(
        f"<p>Found <strong>{len(mut_list)} consensus mutations</strong>:</p>"
    )

    # Table
    lines.append('<table style="margin:0.5rem 0"><thead><tr>'
                  '<th>Mutation</th><th>Position</th><th>Conservation</th>'
                  '<th>Confidence</th><th>Interpretation</th>'
                  '</tr></thead><tbody>')
    for m in mut_list:
        cons = m["conservation"]
        if cons >= 0.8:
            conf = "high"
            conf_class = "good"
            interp = "Very strong evolutionary preference — high confidence"
        elif cons >= 0.6:
            conf = "medium"
            conf_class = "warn"
            interp = "Moderate evolutionary preference — likely tolerated"
        else:
            conf = "low"
            conf_class = ""
            interp = "Weak consensus — test experimentally"

        lines.append(
            f'<tr><td class="mutations">{m["wt"]}{m["position"]}{m["mut"]}</td>'
            f'<td>{m["position"]}</td>'
            f'<td>{cons:.0%}</td>'
            f'<td class="{conf_class}">{conf}</td>'
            f'<td>{interp}</td></tr>'
        )
    lines.append('</tbody></table>')

    if combo_variant:
        n_combo = len([m for m in combo_variant.mutations if m.source_step == "consensus_design"])
        lines.append(
            f"<p>A combined variant with the <strong>top {n_combo} consensus "
            f"mutations</strong> was also generated for testing as a single construct.</p>"
        )

    lines.append(
        f"<p><strong>Output:</strong> {len(candidates)} candidates "
        f"({len(parents)} parent{'s' if len(parents) != 1 else ''} + "
        f"{len(variants)} variants including "
        f"{len(mut_list)} single mutants"
        f'{" + 1 combined variant" if combo_variant else ""}).</p>'
    )

    return _wrap_narrative(name, "Consensus Design", "\n".join(lines))


def _pssm_analysis_narrative(name: str, result: StepResult) -> str:
    """Narrative for pssm_analysis step."""
    candidates = result.candidates
    variants = [c for c in candidates if c.parent_id is not None]

    # Collect unique PSSM mutations
    mut_list: list[dict] = []
    seen = set()
    for v in variants:
        for m in v.mutations:
            if m.source_step != "pssm_analysis":
                continue
            key = (m.position, m.wt, m.mut)
            if key in seen:
                continue
            seen.add(key)
            mut_list.append({
                "position": m.position,
                "wt": m.wt,
                "mut": m.mut,
                "delta": m.score or 0.0,
            })

    mut_list.sort(key=lambda x: x["delta"], reverse=True)

    if not mut_list:
        return _wrap_narrative(
            name,
            "PSSM Analysis",
            "<p>No beneficial mutations were identified by PSSM analysis above "
            "the log-odds threshold.</p>"
            f"<p><strong>Output:</strong> {len(candidates)} candidates.</p>",
        )

    lines = []
    lines.append(
        f"<p>A Position-Specific Scoring Matrix (PSSM) was built from the MSA by "
        f"computing amino acid frequencies at each alignment column. For each "
        f"position, the log-odds ratio between each amino acid's observed frequency "
        f"and a uniform background (1/20) was calculated. Mutations where the "
        f"alternative log-odds score exceeds the wild-type score by &gt;2.0 are "
        f"considered <strong>evolutionarily favorable</strong>.</p>"
    )
    lines.append(
        f"<p>Unlike consensus design (which only proposes the single most common "
        f"residue), PSSM analysis reveals <em>all</em> viable substitutions at "
        f"each position. This captures cases where multiple amino acids are "
        f"tolerated, or where a rare but chemically similar residue may improve "
        f"stability.</p>"
    )
    lines.append(
        f"<p>Found <strong>{len(mut_list)} evolutionarily favorable mutations</strong>:</p>"
    )

    # Group by position to show multi-substitution positions
    by_pos: dict[int, list[dict]] = {}
    for m in mut_list:
        by_pos.setdefault(m["position"], []).append(m)

    lines.append('<table style="margin:0.5rem 0"><thead><tr>'
                  '<th>Mutation</th><th>Position</th>'
                  '<th>PSSM Δ</th><th>Strength</th><th>Notes</th>'
                  '</tr></thead><tbody>')

    for m in mut_list[:30]:  # Top 30
        delta = m["delta"]
        if delta >= 5.0:
            strength = "very strong"
            s_class = "good"
        elif delta >= 3.0:
            strength = "strong"
            s_class = "good"
        elif delta >= 2.0:
            strength = "moderate"
            s_class = "warn"
        else:
            strength = "weak"
            s_class = ""

        notes = []
        # Check if this position has multiple alternatives
        alts = by_pos.get(m["position"], [])
        if len(alts) > 1:
            other_aas = [a["mut"] for a in alts if a["mut"] != m["mut"]]
            if other_aas:
                notes.append(f"also: {', '.join(other_aas)}")
        # Check if Cys is involved
        if m["wt"] == "C":
            notes.append("Cys removal — agrees with Tier 1")
        if m["mut"] == "C":
            notes.append("⚠ introduces Cys")

        lines.append(
            f'<tr><td class="mutations">{m["wt"]}{m["position"]}{m["mut"]}</td>'
            f'<td>{m["position"]}</td>'
            f'<td class="{s_class}">{delta:.2f}</td>'
            f'<td class="{s_class}">{strength}</td>'
            f'<td>{", ".join(notes) if notes else "—"}</td></tr>'
        )
    lines.append('</tbody></table>')

    if len(mut_list) > 30:
        lines.append(
            f"<p><em>Showing top 30 of {len(mut_list)} mutations. "
            f"See full JSON output for the complete list.</em></p>"
        )

    # Cross-reference summary
    cys_muts = [m for m in mut_list if m["wt"] == "C"]
    if cys_muts:
        lines.append(
            f"<p><strong>Cross-reference:</strong> {len(cys_muts)} cysteine "
            f"replacement{'s' if len(cys_muts) != 1 else ''} also identified in "
            f"Tier 1 cysteine scan, providing evolutionary support for these changes: "
            + ", ".join(f'{m["wt"]}{m["position"]}{m["mut"]}' for m in cys_muts)
            + ".</p>"
        )

    lines.append(
        f"<p><strong>Output:</strong> {len(candidates)} candidates "
        f"({len(mut_list)} PSSM-scored variants).</p>"
    )

    return _wrap_narrative(name, "PSSM Analysis", "\n".join(lines))


def _predict_structure_narrative(name: str, result: StepResult) -> str:
    candidates = result.candidates
    with_structure = [c for c in candidates if c.structure_path]
    plddt_scores = [c.scores["plddt"] for c in candidates if "plddt" in c.scores]

    text = (
        f"<p>3D structures were predicted using <strong>Boltz-2</strong> for "
        f"the wild-type protein."
    )
    if plddt_scores:
        avg_plddt = sum(plddt_scores) / len(plddt_scores)
        text += (
            f" Average predicted pLDDT = <strong>{avg_plddt:.1f}</strong> "
            f"(>80 = high confidence, 60–80 = moderate, <60 = low)."
        )
    text += (
        f"</p><p>Structures are used for downstream analyses: disulfide engineering, "
        f"cavity filling, surface patch analysis, and inverse folding scoring.</p>"
        f"<p><strong>Output:</strong> {len(with_structure)} candidate(s) with "
        f"predicted structures.</p>"
    )
    if result.warnings:
        text += f'<p class="warn">⚠ {len(result.warnings)} warning(s).</p>'

    return _wrap_narrative(name, "Structure Prediction", text)


def _stability_ddg_narrative(name: str, result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]
    ddg_scores = [c.scores.get("ddg", 0) for c in variants if "ddg" in c.scores]

    text = (
        f"<p>Thermodynamic stability was estimated for mutations using "
        f"Rosetta/FoldX ΔΔG calculations. <strong>Negative ΔΔG = stabilizing</strong>; "
        f"positive = destabilizing.</p>"
    )
    if ddg_scores:
        stabilizing = [d for d in ddg_scores if d < 0]
        text += (
            f"<p>Found <strong>{len(stabilizing)} stabilizing</strong> mutations "
            f"out of {len(ddg_scores)} tested (ΔΔG range: {min(ddg_scores):.1f} to "
            f"{max(ddg_scores):.1f} kcal/mol).</p>"
        )
    text += f"<p><strong>Output:</strong> {len(result.candidates)} candidates.</p>"
    if result.warnings:
        text += f'<p class="warn">⚠ {len(result.warnings)} warning(s).</p>'

    return _wrap_narrative(name, "Stability ΔΔG", text)


def _disulfide_design_narrative(name: str, result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None
                and any(m.source_step == name for m in c.mutations)]
    pairs = []
    for v in variants:
        pair = v.metadata.get("disulfide_pair")
        dist = v.scores.get("disulfide_cb_distance")
        if pair:
            pairs.append((pair, dist))

    text = (
        "<p>Analyzed the predicted 3D structure for residue pairs that meet "
        "disulfide bond geometric criteria (Cβ–Cβ distance 3.5–4.5 Å, DbD 2.0 algorithm). "
        "Engineered disulfides can significantly improve thermostability (typically "
        "+5–15°C ΔTm).</p>"
    )
    if pairs:
        text += f"<p>Found <strong>{len(pairs)} candidate disulfide pairs</strong>:</p>"
        text += (
            '<table style="margin:0.5rem 0"><thead><tr>'
            '<th>Pair</th><th>Cβ Distance (Å)</th><th>Mutations</th>'
            '</tr></thead><tbody>'
        )
        for pair, dist in pairs:
            text += (
                f'<tr><td>{pair[0]}–{pair[1]}</td>'
                f'<td>{dist:.2f}</td>'
                f'<td>→ Cys</td></tr>'
            )
        text += '</tbody></table>'
    else:
        text += "<p>No disulfide-compatible pairs were found.</p>"

    if result.warnings:
        text += f'<p class="warn">⚠ {len(result.warnings)} warning(s).</p>'

    return _wrap_narrative(name, "Disulfide Engineering", text)


def _cavity_fill_narrative(name: str, result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None
                and any(m.source_step == name for m in c.mutations)]

    text = (
        "<p>Internal cavities were identified by analyzing residue burial "
        "(Cα neighbor count within 10 Å). Small buried residues (Gly, Ala, Val, Ser, Thr) "
        "adjacent to cavities were targeted for conservative size-increasing mutations "
        "(e.g., Ala→Val, Val→Ile) to improve core packing and thermostability.</p>"
    )
    if variants:
        muts = set()
        for v in variants:
            for m in v.mutations:
                if m.source_step == name:
                    muts.add(f"{m.wt}{m.position}{m.mut}")
        text += (
            f"<p>Proposed <strong>{len(muts)} cavity-filling mutations</strong>: "
            f"{', '.join(sorted(muts))}.</p>"
        )
    else:
        text += "<p>No cavity-lining residues suitable for filling were found.</p>"

    if result.warnings:
        text += f'<p class="warn">⚠ {len(result.warnings)} warning(s).</p>'

    return _wrap_narrative(name, "Cavity Filling", text)


def _surface_patch_narrative(name: str, result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None
                and any(m.source_step == name for m in c.mutations)]

    text = (
        "<p>Solvent-accessible surface area (SASA) was computed to identify exposed "
        "hydrophobic patches — clusters of surface-exposed Ile, Leu, Met, Phe, Val, or Trp "
        "residues. These patches promote aggregation and reduce solubility. "
        "Conservative substitutions to polar/charged residues were proposed.</p>"
    )
    if variants:
        patches_set = set()
        muts = set()
        for v in variants:
            for m in v.mutations:
                if m.source_step == name:
                    muts.add(f"{m.wt}{m.position}{m.mut}")
                    patches_set.add(m.metadata.get("patch_size", 0))
        text += (
            f"<p>Found hydrophobic patches of size {', '.join(str(p) for p in sorted(patches_set) if p)}. "
            f"Proposed <strong>{len(muts)} surface mutations</strong>: "
            f"{', '.join(sorted(muts))}.</p>"
        )
    else:
        text += "<p>No concerning hydrophobic patches found on the surface.</p>"

    if result.warnings:
        text += f'<p class="warn">⚠ {len(result.warnings)} warning(s).</p>'

    return _wrap_narrative(name, "Surface Patch Analysis", text)


def _combine_variants_narrative(name: str, result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]
    combos = [c for c in variants if c.metadata.get("combination_order", 0) >= 2]

    text = (
        "<p>Top-ranked individual mutations from all prior steps were combined "
        "into multi-point variant libraries using a combinatorial approach. "
        "Combinations are scored by the additive sum of individual mutation scores.</p>"
    )
    if combos:
        by_order = {}
        for c in combos:
            order = c.metadata.get("combination_order", 2)
            by_order.setdefault(order, []).append(c)

        for order in sorted(by_order):
            text += (
                f"<p>Generated <strong>{len(by_order[order])} "
                f"{'double' if order == 2 else 'triple' if order == 3 else f'{order}-point'} "
                f"mutants</strong>.</p>"
            )

        # Show top 5 combinations
        top_combos = sorted(combos, key=lambda c: c.scores.get("combo_sum", 0), reverse=True)[:5]
        if top_combos:
            text += (
                '<table style="margin:0.5rem 0"><thead><tr>'
                '<th>Variant</th><th>Mutations</th><th>Combined Score</th>'
                '</tr></thead><tbody>'
            )
            for c in top_combos:
                muts = ", ".join(c.metadata.get("component_mutations", []))
                score = c.scores.get("combo_sum", 0)
                text += (
                    f'<tr><td>{c.name}</td>'
                    f'<td class="mutations">{muts}</td>'
                    f'<td>{score:.2f}</td></tr>'
                )
            text += '</tbody></table>'
    else:
        text += "<p>No combinatorial variants were generated.</p>"

    text += f"<p><strong>Output:</strong> {len(result.candidates)} total candidates.</p>"
    if result.warnings:
        text += f'<p class="warn">⚠ {len(result.warnings)} warning(s).</p>'

    return _wrap_narrative(name, "Combinatorial Variant Library", text)


def _e1_score_narrative(name: str, result: StepResult) -> str:
    variants = [c for c in result.candidates if "e1_fitness" in c.scores]
    scores = [c.scores["e1_fitness"] for c in variants if "e1_fitness" in c.scores]
    if scores:
        avg = sum(scores) / len(scores)
        best = max(scores)
        return _wrap_narrative(
            name,
            "Profluent E1 Scoring",
            f"<p>Variants were scored using the <strong>Profluent E1</strong> protein "
            f"language model (masked-marginal likelihood). This model evaluates mutation "
            f"fitness based on evolutionary plausibility learned from millions of protein "
            f"sequences.</p>"
            f"<p>Scored <strong>{len(scores)}</strong> variants: average fitness "
            f"= {avg:.4f}, best = {best:.4f}.</p>"
            f"<p><strong>Output:</strong> {len(result.candidates)} candidates.</p>",
        )
    return _generic_narrative(name, result)


def _esm1v_score_narrative(name: str, result: StepResult) -> str:
    variants = [c for c in result.candidates if "esm1v_delta" in c.scores]
    scores = [c.scores["esm1v_delta"] for c in variants if "esm1v_delta" in c.scores]
    if scores:
        avg = sum(scores) / len(scores)
        best = max(scores)
        return _wrap_narrative(
            name,
            "ESM-1v Scoring",
            f"<p>Mutation effects were predicted using the <strong>ESM-1v</strong> "
            f"5-model ensemble (masked-marginal scoring). Positive Δ scores indicate "
            f"mutations predicted to be more fit than wild type.</p>"
            f"<p>Scored <strong>{len(scores)}</strong> variants: average Δ = {avg:.4f}, "
            f"best Δ = {best:.4f}.</p>"
            f"<p><strong>Output:</strong> {len(result.candidates)} candidates.</p>",
        )
    return _generic_narrative(name, result)


def _generic_narrative(name: str, result: StepResult) -> str:
    """Fallback narrative for steps without a custom narrator."""
    variants = [c for c in result.candidates if c.parent_id is not None]
    parents = [c for c in result.candidates if c.parent_id is None]

    # Collect unique mutations from this step
    unique_muts = set()
    for v in variants:
        for m in v.mutations:
            if m.source_step == name:
                unique_muts.add((m.position, m.wt, m.mut))

    title = name.replace("_", " ").title()
    text = (
        f"<p>The <strong>{title}</strong> step processed "
        f"{len(parents)} input protein{'s' if len(parents) != 1 else ''} and produced "
        f"{len(result.candidates)} candidates "
        f"({len(variants)} variant{'s' if len(variants) != 1 else ''})."
    )
    if unique_muts:
        text += (
            f" Proposed <strong>{len(unique_muts)} unique mutation{'s' if len(unique_muts) != 1 else ''}"
            f"</strong> at {len(set(p for p, _, _ in unique_muts))} positions."
        )
    text += "</p>"

    if result.warnings:
        text += f'<p class="warn">⚠ {len(result.warnings)} warning(s) generated.</p>'

    return _wrap_narrative(name, title, text)


def generate_recommendation_summary(results: dict[str, StepResult]) -> str:
    """Generate a cross-tier mutation recommendation summary.

    Consolidates all proposed mutations across steps, identifies overlaps
    (mutations supported by multiple analyses), and ranks them by confidence.
    """
    # Collect all mutations from all steps with their sources and scores
    mutation_evidence: dict[tuple[int, str, str], dict] = {}
    # key: (position, wt, mut) → {sources: [(step, score)], ...}

    for step_name, result in results.items():
        if step_name == "input":
            continue
        for c in result.candidates:
            if c.parent_id is None:
                continue
            for m in c.mutations:
                key = (m.position, m.wt, m.mut)
                if key not in mutation_evidence:
                    mutation_evidence[key] = {
                        "position": m.position,
                        "wt": m.wt,
                        "mut": m.mut,
                        "sources": [],
                        "scores": {},
                    }
                source = m.source_step or step_name
                if source not in [s for s, _ in mutation_evidence[key]["sources"]]:
                    mutation_evidence[key]["sources"].append(
                        (source, m.score or 0.0)
                    )

    if not mutation_evidence:
        return ""

    # Classify mutations
    multi_support = []   # supported by ≥2 independent analyses
    single_support = []  # supported by 1 analysis

    for key, info in mutation_evidence.items():
        n_sources = len(info["sources"])
        # Compute a combined confidence score
        source_names = [s for s, _ in info["sources"]]
        max_score = max(sc for _, sc in info["sources"])

        # Categorize the mutation type
        wt, mut = info["wt"], info["mut"]
        if wt == "C":
            cat = "Cysteine removal"
        elif wt == "M" and mut == "L":
            cat = "Oxidation fix"
        elif wt == "N" and mut == "Q":
            cat = "Deamidation fix"
        elif wt == "D" and mut == "E":
            cat = "Isomerization fix"
        elif wt in ("K", "R") and mut == "Q":
            cat = "Proteolysis fix"
        elif "consensus_design" in source_names:
            cat = "Consensus reversion"
        elif "pssm_analysis" in source_names:
            cat = "PSSM-guided"
        else:
            cat = "Other"

        entry = {
            **info,
            "n_sources": n_sources,
            "source_names": source_names,
            "max_score": max_score,
            "category": cat,
        }

        if n_sources >= 2:
            multi_support.append(entry)
        else:
            single_support.append(entry)

    multi_support.sort(key=lambda x: (x["n_sources"], x["max_score"]), reverse=True)
    single_support.sort(key=lambda x: x["max_score"], reverse=True)

    lines = []
    lines.append('<h2>Recommended Mutations</h2>')
    lines.append(
        '<p style="color:#8b949e;font-size:0.9rem;margin-bottom:1rem">'
        'Cross-tier consolidation of all proposed mutations, ranked by confidence. '
        'Mutations supported by multiple independent analyses are highest priority.</p>'
    )

    # Summary stats
    total_unique = len(mutation_evidence)
    n_multi = len(multi_support)
    unique_positions = len(set(k[0] for k in mutation_evidence))
    lines.append(
        f'<div class="stats">'
        f'<div class="stat-card"><div class="number">{total_unique}</div>'
        f'<div class="label">Unique Mutations</div></div>'
        f'<div class="stat-card"><div class="number">{unique_positions}</div>'
        f'<div class="label">Positions Targeted</div></div>'
        f'<div class="stat-card"><div class="number good">{n_multi}</div>'
        f'<div class="label">Multi-Evidence</div></div>'
        f'</div>'
    )

    # Multi-evidence mutations (highest confidence)
    if multi_support:
        lines.append(
            '<h3 style="color:#3fb950;margin-top:1.5rem">'
            '★ High-Confidence Mutations (supported by multiple analyses)</h3>'
        )
        lines.append(
            '<p>These mutations were independently identified by two or more '
            'pipeline steps, providing strong confidence they are safe and beneficial.</p>'
        )
        lines.append(
            '<table style="margin:0.5rem 0"><thead><tr>'
            '<th>Rank</th><th>Mutation</th><th>Category</th>'
            '<th>Supporting Evidence</th><th>Recommendation</th>'
            '</tr></thead><tbody>'
        )
        for rank, m in enumerate(multi_support, 1):
            label = f'{m["wt"]}{m["position"]}{m["mut"]}'
            evidence = ", ".join(
                f'{s} ({sc:.2f})' for s, sc in m["sources"]
            )
            # Recommendation strength
            if m["n_sources"] >= 3:
                rec = "Strongly recommended"
                rec_class = "good"
            elif m["n_sources"] == 2 and m["max_score"] >= 3.0:
                rec = "Recommended"
                rec_class = "good"
            else:
                rec = "Likely beneficial"
                rec_class = "warn"

            lines.append(
                f'<tr><td>{rank}</td>'
                f'<td class="mutations" style="font-weight:bold">{label}</td>'
                f'<td>{m["category"]}</td>'
                f'<td>{evidence}</td>'
                f'<td class="{rec_class}">{rec}</td></tr>'
            )
        lines.append('</tbody></table>')

    # Single-evidence mutations by category
    if single_support:
        # Group by category
        by_cat: dict[str, list] = {}
        for m in single_support:
            by_cat.setdefault(m["category"], []).append(m)

        lines.append(
            '<h3 style="margin-top:1.5rem">'
            'Single-Evidence Mutations (by category)</h3>'
        )
        lines.append(
            '<p>These mutations were identified by one analysis step. They are '
            'reasonable candidates but should be validated experimentally or with '
            'additional computational methods.</p>'
        )

        for cat, muts in by_cat.items():
            muts.sort(key=lambda x: x["max_score"], reverse=True)
            lines.append(
                f'<h4>{cat} ({len(muts)} mutation'
                f'{"s" if len(muts) != 1 else ""})</h4>'
            )
            lines.append(
                '<table style="margin:0.5rem 0"><thead><tr>'
                '<th>Mutation</th><th>Source</th><th>Score</th>'
                '</tr></thead><tbody>'
            )
            for m in muts[:15]:  # Top 15 per category
                label = f'{m["wt"]}{m["position"]}{m["mut"]}'
                source = m["source_names"][0]
                score = m["sources"][0][1]
                lines.append(
                    f'<tr><td class="mutations">{label}</td>'
                    f'<td>{source}</td>'
                    f'<td>{score:.2f}</td></tr>'
                )
            lines.append('</tbody></table>')
            if len(muts) > 15:
                lines.append(
                    f'<p><em>Showing top 15 of {len(muts)}. '
                    f'See JSON output for full list.</em></p>'
                )

    return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────


def _wrap_narrative(step_name: str, title: str, content: str) -> str:
    """Wrap narrative content in a styled HTML section."""
    return f"""
<div class="narrative-block" id="narrative-{html.escape(step_name)}">
  <h3>
    <span class="step-tag">Step</span> {html.escape(title)}
  </h3>
  <div class="narrative-content">
    {content}
  </div>
</div>
"""


# ── Registry ─────────────────────────────────────────────────────────

_NARRATORS: dict[str, Any] = {
    "cysteine_scan": _cysteine_scan_narrative,
    "motif_scan": _motif_scan_narrative,
    "sequence_complexity": _sequence_complexity_narrative,
    "find_homologs": _find_homologs_narrative,
    "consensus_design": _consensus_design_narrative,
    "pssm_analysis": _pssm_analysis_narrative,
    "predict_structure": _predict_structure_narrative,
    "stability_ddg": _stability_ddg_narrative,
    "disulfide_design": _disulfide_design_narrative,
    "cavity_fill": _cavity_fill_narrative,
    "surface_patch": _surface_patch_narrative,
    "e1_score": _e1_score_narrative,
    "esm1v_score": _esm1v_score_narrative,
    "combine_variants": _combine_variants_narrative,
}
