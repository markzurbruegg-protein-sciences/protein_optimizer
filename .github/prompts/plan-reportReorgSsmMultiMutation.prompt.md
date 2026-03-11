# Plan: Report Reorganization + SSM Scanning + Multi-Mutation Optimization

Major three-phase reorganization: (1) restructure report sidebar and body narratives for logical biological flow, remove duplication, add ThermMPNN; (2) add targeted ProtSolM SSM alongside existing E1/ThermMPNN SSM; (3) build a full-combinatorial mutation optimizer that pools top mutations from all sources and scores all combinations with Profluent E1.

---

## Phase 1: Report Reorganization

### 1A. Sidebar Metrics Panel — `src/protein_optimizer/reporting/html_report.py`

Current issues: "Aggregation & Solubility" and "Solubility" are two redundant sections; GRAVY (a solubility indicator) is buried in "Physicochemical"; ThermMPNN stability data is absent; identity metrics (pI, ε₂₈₀) are mixed into Physicochemical.

**Proposed sidebar** (biological flow: identity → quality → stability → solubility → handling → producibility → features → fold):

| # | Section | Key Changes |
|---|---------|-------------|
| 1 | **Sequence Identity** | New section: Length, MW, pI, ε₂₈₀, Net Charge — *moved from Physicochemical* |
| 2 | **Structure Quality** | Unchanged: pLDDT |
| 3 | **Thermal Stability** | Est. Tm, Instability Index, Aliphatic Index + **NEW: ThermMPNN Best ΔΔG|
| 4 | **Solubility & Aggregation** | **Merged** two sections + GRAVY moved here: ProtSolM, Ensemble, GRAVY, W-H, β-Agg Regions, Nucleation Cores, APRs, SAP, etc. |
| 5 | **Physicochemical** | Slimmed: pH Range, Charge vs pH, Aromaticity, Cysteines, Disulfide Bonds |
| 6 | **Expression (E. coli)** | Unchanged |
| 7 | **Sequence Features** | Unchanged |
| 8 | **Fold & Architecture** | Unchanged |
| 9 | **Legend** | Unchanged |

ThermMPNN metrics calculated for this sequence

### 1B. Body Narrative Reorganization — `src/protein_optimizer/reporting/narratives.py`

Currently narratives render in pipeline execution order. Restructure into **thematic sections** with headers:

1. **Executive Summary** (enhanced with ThermMPNN + SSM highlights)
2. **Sequence Analysis**: Protein Characterization → Sequence Complexity → Motif Scan → Cysteine Scan
3. **Evolutionary Analysis**: Homolog Discovery → Consensus Design → PSSM Analysis
4. **Structural Analysis**: Structure Prediction → Stability ΔΔG → Disulfide Design → Cavity Fill → Surface Patch
5. **In Silico Mutagenesis** ← NEW: E1 SSM → ThermMPNN SSM → ProtSolM SSM → cross-tool agreement
6. **Design & Generation**: this section is removed and proteinMpnn and RFdiffusion narratives are not run in ths pipeline. all mentions of these are removed
7. **Multi-Mutation Optimization** ← NEW: evidence-pooled singles table → combinatorial library → top multi-mutants
8. **PLM Validation**: only E1 is used for calidation
9. **Full Variant Library** (existing)

Add `NARRATIVE_ORDER` constant defining this grouping; modify `_analysis_narrative_section()` to render with section dividers.

---

## Phase 2: SSM Scanning Pipeline

### 2A. E1 SSM (existing — `src/protein_optimizer/steps/e1_score.py`)
- Increase default `top_k_positions` from 5 → 20 for broader scanning
- Store summary stats in step metadata for report/sidebar consumption

### 2B. ThermMPNN SSM (existing — `src/protein_optimizer/steps/stability_ddg.py`)
- Already runs full SSM. Add: store `best_ddg`, `n_stabilizing`, `top_5_mutations` in step metadata
- These feed into the new SSM section

### 2C. ProtSolM Targeted SSM ← NEW
- **New file:** `src/protein_optimizer/steps/solubility_ssm.py` (Tier 3, requires `predict_structure` + `protein_characterization`)
- **Modified file:** `scripts/protsolm_helper.py` — add batch prediction mode

**How it works:**
1. Collect target positions from prior results: surface-exposed hydrophobic residues (from SASA/surface_patch) + APR positions (from characterization aggregation analysis). Typically ~30-80 positions for a 300-residue protein
2. For each target position, generate 19 mutant sequences (all alternative AAs)
3. Call protsolm_helper in **batch mode**: same PDB backbone, pre-compute graph + structural features once, only re-run ESM2 embeddings + AA composition features per variant
4. Compute ΔSolubility = P(soluble)_variant - P(soluble)_WT
5. Create candidates for mutations with ΔSolubility > 0

**Performance:** ~950 variants, batched ESM2 inference → ~2-5 min on A100

**New narrative:** `_solubility_ssm_narrative()` — top solubility-improving mutations, hotspot positions, cross-reference with APRs

---

## Phase 3: Multi-Mutation Optimization

**New file:** `src/protein_optimizer/steps/mutation_optimizer.py` (Tier 5, after all SSM + scoring)

### Sub-steps:

**i. Mutation Pool Construction**
Collect all beneficial mutations from E1 SSM (gain > 0), ThermMPNN SSM (ΔΔG < threshold), ProtSolM SSM (ΔSol > 0), consensus design, PSSM, motif fixes, cysteine removal, cavity fill, surface patch. Deduplicate by position+mutation; merge evidence when same mutation found by multiple tools.

**ii. Evidence Scoring**
Build evidence matrix — each mutation scored on: `normalized_e1_gain`, `normalized_ddg` (inverted), `normalized_delta_solubility`, `evidence_breadth` (count of tools supporting it). Rank by weighted evidence score; select top-20 (configurable). No two mutations at same position.

**iii. Full Combinatorial Generation**
From top-20 pool, generate ALL combinations up to order 5: ~21,700 combinations total. Apply biological filters: max net charge shift ±5, respect protected residues.

**iv. E1 Scoring**
Score entire combinatorial library with Profluent E1 (wildtype-marginal — handles multi-mutations natively, captures epistasis). Deduplicate sequences before scoring. ~35 min with batching for ~21K sequences.

**v. Additive Model Comparison**
For each combination compute: `additive_ddg` (sum of individual ΔΔGs), `additive_sol` (dampened average), `epistasis_signal` (E1_combo - Σ E1_singles) — measures non-additive effects.

**vi. Output Tables**

| Table | Columns |
|-------|---------|
| **Single Mutation Evidence** | Position, WT, MUT, E1 Gain, ΔΔG, ΔSol, Evidence Sources, Evidence Score |
| **Top Multi-Mutants** (top 50-100) | Rank, Mutations, E1 Fitness, Additive ΔΔG, Additive ΔSol, Epistasis Signal, N_mut |

---

## Phase 4: Configuration & Integration

- **Config updates:** Add `solubility_ssm` and `mutation_optimizer` to `src/protein_optimizer/config.py` `DEFAULT_CONFIG`, `configs/full_pipeline.yaml`, `configs/ecoli_enzyme.yaml`
- **Scoring aggregator:** Add `delta_solubility` weight (0.10) and `evidence_score` weight (0.05) to `src/protein_optimizer/scoring/aggregator.py`
- **Pipeline order:** solubility_ssm runs parallel with Stage 3 structural steps; mutation_optimizer runs as final Stage 6 after all scoring

---

## Relevant Files

### Modify:
- `src/protein_optimizer/reporting/html_report.py` — sidebar reorder, thematic sections, new SSM/optimizer sections
- `src/protein_optimizer/reporting/narratives.py` — narrative reorder, `NARRATIVE_ORDER`, new narrative generators
- `src/protein_optimizer/steps/e1_score.py` — broader SSM defaults, summary metadata
- `src/protein_optimizer/steps/stability_ddg.py` — store summary stats for sidebar
- `scripts/protsolm_helper.py` — add batch prediction mode
- `src/protein_optimizer/config.py` — DEFAULT_CONFIG for new steps
- `src/protein_optimizer/scoring/aggregator.py` — new score weights
- `configs/full_pipeline.yaml` — add new pipeline stages
- `configs/ecoli_enzyme.yaml` — add new steps

### Create:
- `src/protein_optimizer/steps/solubility_ssm.py` — ProtSolM targeted SSM step
- `src/protein_optimizer/steps/mutation_optimizer.py` — combinatorial optimization + E1 scoring

### Reference (use as implementation templates):
- `src/protein_optimizer/steps/stability_ddg.py` — SSM step pattern
- `src/protein_optimizer/steps/combine_variants.py` — mutation collection + combination logic
- `src/protein_optimizer/steps/base.py` — `BaseStep` interface

---

## Verification

1. **Unit tests:** `test_solubility_ssm.py` (mock ProtSolM, verify position targeting + ΔSolubility calc), `test_mutation_optimizer.py` (mock E1, verify evidence matrix + combinatorial generation + biological filters), `test_report_reorganization.py` (verify sidebar order + no duplicates + ThermMPNN present)
2. **Integration test:** Run full pipeline on jcDRM — verify `solubility_ssm.json` and `mutation_optimizer.json` created with expected structure
3. **Report visual QC:** Sidebar flows logically, no duplicate metrics, ThermMPNN in Thermal Stability, SSM section shows all three tools, multi-mutation table sortable and color-coded
4. **Regression:** All existing `pytest tests/` pass
5. **Performance:** ProtSolM SSM < 10 min; mutation_optimizer with E1 < 60 min for ~21K combinations

---

## Decisions

- ThermMPNN sidebar shows best ΔΔG + count of stabilizing mutations
- ProtSolM SSM is targeted (surface + APR positions) for practical compute time
- Full combinatorial + E1 scoring (up to order 5, top-20 pool)
- Both sidebar AND body narratives reorganized
- E1 handles epistasis natively; additive ΔΔG provided as comparison only
- **Excluded:** training a new ML model; modifying RFdiffusion/ProteinMPNN; changing existing aggregator weight values

## Open Considerations

1. **Library size cap:** Top-20 pool at order-5 yields ~21K variants. If GPU memory is tight, reduce to order-4 (~6K) or top-15 pool (~4.9K). Recommend making `max_combination_order` configurable.
2. **ESM-1v cross-validation:** Optional ESM-1v scoring of multi-mutants adds ~2 hours but provides independent validation. Recommend off by default, configurable.
3. **ProtSolM conda environment:** The SSM step needs the same conda env as the ProtSolM helper — verify env name in deployment config.
