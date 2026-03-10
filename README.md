# protein-optimizer

A modular protein optimization pipeline for industrial enzyme engineering targeting *E. coli* expression. Progresses from simple sequence heuristics through evolutionary analysis, structure-based engineering, AI/ML scoring, and generative protein design.

Each step is independently runnable via CLI **or** composable into end-to-end pipelines with automatic dependency resolution.

---

## Overview

```
Tier 1 — Sequence Scan        Tier 2 — Evolutionary       Tier 3 — Structure
  cysteine_scan                  find_homologs               predict_structure (Boltz-2)
  motif_scan                     consensus_design            stability_ddg (Rosetta/FoldX)
  sequence_complexity            pssm_analysis               disulfide_design
                                                             cavity_fill
                                                             surface_patch

Tier 4 — AI/ML Scoring         Tier 5 — Generative Design
  e1_score (Profluent E1)        rfdiffusion_diversify
  esm1v_score (5-model)          design_validate (MPNN→Boltz-2)
  proteinmpnn_design             motif_scaffold
  esmif1_score
  combine_variants
```

**19 steps** organized into 5 tiers, with score aggregation, quality filtering, diversity selection, and HTML reporting.

## Installation

```bash
# Core (Tiers 1–3 sequence/evolutionary steps)
pip install -e .

# With AI/ML models (Tier 4)
pip install -e ".[tier4]"

# With Boltz-2 structure prediction
pip install -e ".[boltz]"

# Everything + plotting
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

### External tool requirements

| Tool | Steps | Install |
|------|-------|---------|
| **MMseqs2** | `find_homologs` | `conda install -c bioconda mmseqs2` |
| **MAFFT** | `find_homologs` | `conda install -c bioconda mafft` |
| **Boltz-2** | `predict_structure`, `design_validate` | `pip install boltz[cuda]` |
| **ESMFold** | `predict_structure` (fallback) | `pip install fair-esm` |
| **PyRosetta** | `stability_ddg` | Academic license from pyrosetta.org |
| **FoldX** | `stability_ddg` (alternative) | Academic license from foldxsuite.crg.eu |
| **Profluent E1** | `e1_score` | `pip install git+https://github.com/Profluent-AI/E1.git` |
| **ProteinMPNN** | `proteinmpnn_design`, `design_validate` | `git clone https://github.com/dauparas/ProteinMPNN` + set `PROTEINMPNN_DIR` |
| **RFdiffusion** | `rfdiffusion_diversify`, `motif_scaffold` | SE3nv conda env + weights from IPD; set `RFDIFFUSION_DIR` |

> **Tiers 1–2 require only BioPython** — no GPU or external tools needed.

## Quick Start

### Run a single step

```bash
# Scan cysteines in a FASTA file
protein-opt step cysteine_scan -i my_enzyme.fasta -o cys_result.json

# Score motifs (deamidation, oxidation, proteolysis, aggregation)
protein-opt step motif_scan -i my_enzyme.fasta

# Chain steps: use previous output as input
protein-opt step consensus_design -i find_homologs_result.json -o consensus.json
```

### Run the full pipeline

```bash
# Just point at your FASTA — results and report appear next to it
protopt run my_enzyme.fasta

# With a custom config
protopt run my_enzyme.fasta -c configs/custom.yaml
```

This creates:
```
my_enzyme.fasta            # your input
my_enzyme_results/         # step JSONs, structures, logs
my_enzyme_report.html      # interactive HTML report with 3D viewer
```

### Regenerate the report

```bash
protopt report my_enzyme.fasta
```

### Rank and inspect candidates

```bash
protopt rank -i my_enzyme_results/combine_variants.json -n 20
```

### List available steps

```bash
protein-opt list-steps
```

### Protect residues from mutation

```bash
# Protect active-site residues (1-based positions)
protein-opt step cysteine_scan -i enzyme.fasta --protected-residues 45,112,198
```

## Configuration

Pipeline behavior is controlled via YAML config files. A complete default config is at [`configs/default.yaml`](configs/default.yaml).

```yaml
pipeline:
  steps:
    - cysteine_scan
    - motif_scan
    - sequence_complexity
    - find_homologs
    - consensus_design
    - pssm_analysis

global:
  protected_residues: [45, 112, 198]  # active site
  expression_host: ecoli
  output_dir: ./results

# Step-specific overrides
cysteine_scan:
  replacement_aas: ["S", "A"]

e1_score:
  model: Profluent-Bio/E1-600m
  retrieval_augmented: true    # use homologs for context
  site_saturation: false

scoring:
  weights:
    e1_fitness: 0.25
    esm1v_delta: 0.20
    esmif1_delta: 0.15
    consensus_score: 0.10
```

Override individual step parameters from the CLI:

```bash
protein-opt step e1_score -i input.json sampling_temp=0.2 top_k_positions=10
```

## Architecture

### Data flow

Every step reads a **`StepResult`** (JSON) and writes a **`StepResult`**:

```
FASTA → StepResult → [Step 1] → StepResult → [Step 2] → ... → StepResult → Report
```

`StepResult` contains a list of `ProteinCandidate` objects, each carrying:
- Amino acid sequence
- Parent lineage tracking (candidate_id / parent_id)
- Mutation history
- Scores from each analysis step
- Optional structure path (PDB)
- Arbitrary metadata

### Step abstraction

```python
from protein_optimizer.steps.base import BaseStep
from protein_optimizer.models import StepResult

class MyCustomStep(BaseStep):
    name = "my_step"
    tier = 2
    title = "My Custom Step"
    description = "Does something useful."
    requires = []  # dependency step names

    def run(self, step_input: StepResult, config: dict) -> StepResult:
        # Your logic here
        return StepResult(step_name=self.name, candidates=[...])
```

Steps auto-register via `__init_subclass__` — just define the class and it appears in the CLI.

### Scoring & filtering

After the pipeline runs, candidates are:
1. **Score-aggregated** — weighted z-score normalization across E1, ESM-1v, ESM-IF1, ΔΔG, MPNN, consensus, PSSM scores
2. **Filtered** — configurable thresholds (max mutations, min pLDDT, max ΔΔG, no new Cys, no Pro in helix)
3. **Diversity-selected** — Hamming-distance deduplication
4. **Reported** — self-contained HTML with ranked tables, mutation frequency maps, and warnings

## Project Structure

```
protein_optimizer/
├── configs/
│   └── default.yaml              # Full default configuration
├── src/protein_optimizer/
│   ├── cli.py                    # Click CLI (run, step, list-steps, rank, report)
│   ├── config.py                 # YAML config loading with deep-merge
│   ├── io_utils.py               # FASTA/JSON I/O
│   ├── models.py                 # Mutation, ProteinCandidate, StepResult
│   ├── pipeline.py               # Orchestrator with dependency resolution
│   ├── scoring/
│   │   ├── aggregator.py         # Weighted composite scoring
│   │   └── filters.py            # Quality + diversity filters
│   ├── reporting/
│   │   ├── html_report.py        # Self-contained HTML report
│   │   └── plots.py              # matplotlib visualizations
│   └── steps/
│       ├── base.py               # BaseStep ABC + registry
│       ├── cysteine_scan.py      # Tier 1: Cys→Ser/Ala
│       ├── motif_scan.py         # Tier 1: deamidation/oxidation/proteolysis
│       ├── sequence_complexity.py # Tier 1: homopolymers, Pro runs, entropy
│       ├── find_homologs.py      # Tier 2: MMseqs2 + MAFFT MSA
│       ├── consensus_design.py   # Tier 2: back-to-consensus mutations
│       ├── pssm_analysis.py      # Tier 2: PSSM log-odds scoring
│       ├── predict_structure.py  # Tier 3: Boltz-2 / ESMFold
│       ├── stability_ddg.py      # Tier 3: Rosetta/FoldX ΔΔG
│       ├── disulfide_design.py   # Tier 3: geometric SS-bond finder
│       ├── cavity_fill.py        # Tier 3: core packing improvements
│       ├── surface_patch.py      # Tier 3: hydrophobic patch reduction
│       ├── e1_score.py           # Tier 4: Profluent E1 masked-marginal
│       ├── esm1v_score.py        # Tier 4: ESM-1v 5-model ensemble
│       ├── proteinmpnn_design.py # Tier 4: SolubleMPNN design
│       ├── esmif1_score.py       # Tier 4: inverse folding scoring
│       ├── combine_variants.py   # Tier 4: combinatorial library
│       ├── rfdiffusion_diversify.py # Tier 5: partial diffusion
│       ├── design_validate.py    # Tier 5: MPNN→Boltz-2 validation
│       └── motif_scaffold.py     # Tier 5: active site grafting
├── tests/
│   ├── fixtures/example.fasta
│   ├── test_models.py
│   ├── test_io_utils.py
│   ├── test_tier1_steps.py
│   ├── test_scoring.py
│   ├── test_pipeline.py
│   └── test_reporting.py
└── pyproject.toml
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
