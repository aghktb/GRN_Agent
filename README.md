# GRNAgent: A Multimodal Graph Reasoning Agent for Gene Regulatory Inference

![GRNAgent overview](<docs/Figure1_Overview (2).png>)

**GRNAgent** is a **multi-agent** system for gene regulatory network (GRN) inference. Agents cooperate under explicit contracts: acquisition and ingest prepare data, a split agent defines leakage-safe train/val/test boundaries, window builders assemble **TF-centered evidence graphs** per neighborhood sample, and the **TF-EAGER** agent performs **graph reasoning** over each window to score TF→gene edges. Training, inference, and evaluation scripts orchestrate those handoffs from YAML configs.

**TF-EAGER** is GRNAgent’s **graph reasoning agent**: it consumes a compact typed evidence graph per TF window built from expression, optional accessibility and motif signals, and related regulatory evidence, runs the window model, and produces calibrated edge scores. It scores all candidate genes in a sampled TF neighborhood jointly, then exports scored edges and flat evidence for downstream evaluation.

For semantics (labeling, negatives, split policy, knobs), see [`docs/TF-EAGER.md`](docs/TF-EAGER.md).

---

## Agents and handoffs

Implementation lives under `src/grn_agent/` (agents, acquisition, I/O, schemas, `models/tf_eager/`). Typical **integrated** orchestration is `scripts/run_integrated_tf_eager_workflow.py`, which wires stages in order when enabled in your master YAML.

| Agent / stage | Role | Typical artifact |
|---------------|------|-------------------|
| **Acquisition** | Resolves RNA and optional ATAC/motif inputs; writes a multimodal manifest when enabled. | `multimodal_manifest.json` |
| **Ingest & harmonize** | Loads expression (and optional multimodal indexes) aligned to gene symbols and context. | In-memory + paths in resolved configs |
| **Split** | Builds a TF-aware holdout manifest so train/val/test respect your strategy (e.g. leave-one-TF-out). | `split_manifest.csv` |
| **Window builder** | For each split subset, samples TF-centered subgraphs, labels candidates, and materializes **one evidence graph per window** (JSONL lines). | `train_windows.jsonl`, `test_windows.jsonl` |
| **TF-EAGER (graph reasoning)** | Trains or runs the window model over those graphs; learns cross-gene structure within each window. | `tf_eager_bootstrap_v2.pt` (checkpoint) |
| **Score & export** | Writes per-edge scores, an optional thresholded network, and flattened evidence for eval. | `test_scored_edges.csv`, `test_network.csv`, `test_flat_evidence.jsonl` |
| **Evaluation** | Held-out metrics with split constraints and optional negative-ratio sweeps (`scripts/eval_grn_agent.py`). | e.g. `evaluation/eval_test_by_ratio.json` |

---

## Install

From the repository root:

```bash
pip install -e .
pip install -e ".[torch]"          # training and GPU inference for TF-EAGER
pip install -e ".[acquisition]"   # optional: multimodal acquisition (ATAC/motif, etc.)
pip install -e ".[dev]"           # optional: tests
```

---

## How to run (by goal)

| Goal | Command | Config entry points |
|------|---------|---------------------|
| Full single-dataset pipeline (acquire → split → windows → train TF-EAGER → test infer → eval) | `python scripts/run_integrated_tf_eager_workflow.py --config <YAML>` | [Integrated single-context](#config-file-templates) |
| One shared TF-EAGER model across many contexts | `python scripts/run_multicontext_tf_eager_workflow.py --config <YAML>` | [Multicontext](#multicontext) |
| Blind inference / blind eval over many DataContext folders | `python scripts/run_blind_tf_eager_datacontext_eval.py --config <YAML>` | [Blind datacontext](#blind-datacontext-inference-and-evaluation) |
| Manual build → train → infer (debugging or custom wiring) | `build_tf_eager_windows.py` → `train_tf_eager.py` → `infer_tf_eager.py` | [Manual pipeline](#manual-pipeline-build-train-infer) |

Artifacts live under `artifacts/` (exact layout follows `workflow.id` and `artifact_root` in your YAML).

---

## Config file templates

Copy a template and point it at your expression matrix, `TFs.csv`, gold edges, and genome settings as needed.

### Generic integrated template

| File | Purpose |
|------|---------|
| [`conf/tf_eager_integrated_standard.yml`](conf/tf_eager_integrated_standard.yml) | Annotated master YAML: `workflow`, `acquisition`, `dataset`, `split`, window build, `train_tf_eager`, `infer_tf_eager`, `evaluation`. |

### Integrated single-context

| File | Purpose |
|------|---------|
| [`conf/mESC_tf500_tf_eager_integrated.yml`](conf/mESC_tf500_tf_eager_integrated.yml) | mESC tf500 integrated workflow. |
| [`conf/mESC/mESC_tf_eager_integrated.yml`](conf/mESC/mESC_tf_eager_integrated.yml) | mESC integrated variant. |
| [`conf/mHSC-GM_celltype_specific_chipseq_Tf1000/tf_eager_integrated_mHSC-GM.yml`](conf/mHSC-GM_celltype_specific_chipseq_Tf1000/tf_eager_integrated_mHSC-GM.yml) | mHSC–GM example. |
| [`conf/ecoli/tf_eager_integrated_ecoli.yml`](conf/ecoli/tf_eager_integrated_ecoli.yml) | *E. coli* example. |

### Train / infer YAML (manual or scripted steps)

| File | Purpose |
|------|---------|
| [`conf/mESC_tf500_tf_eager_train.yml`](conf/mESC_tf500_tf_eager_train.yml) | Window build + train + infer settings for `build_tf_eager_windows.py`, `train_tf_eager.py`, `infer_tf_eager.py`. |

### Multicontext

| File | Purpose |
|------|---------|
| [`conf/multicontext_tf_eager/all_datacontext_contexts_neg2.yml`](conf/multicontext_tf_eager/all_datacontext_contexts_neg2.yml) | Neg2 variant. |
| [`conf/multicontext_tf_eager/all_datacontext_contexts_neg2_single_stage.yml`](conf/multicontext_tf_eager/all_datacontext_contexts_neg2_single_stage.yml) | Neg2 single-stage. |
| [`conf/multicontext_tf_eager/all_datacontext_contexts_neg2_functional_only.yml`](conf/multicontext_tf_eager/all_datacontext_contexts_neg2_functional_only.yml) | Neg2 functional-only. |

**Resume or partial recompute:**

```bash
python scripts/run_multicontext_tf_eager_workflow.py \
  --config conf/multicontext_tf_eager/all_datacontext_contexts.yml \
  --force-recompute \
  --start-from <context_id>
```

### Blind datacontext

| File | Purpose |
|------|---------|
| [`conf/blind_tf_eager_datacontext_eval_neg2.yml`](conf/blind_tf_eager_datacontext_eval_neg2.yml) | Neg2 blind. |
| [`conf/blind_tf_eager_datacontext_eval_neg2_exhaustive.yml`](conf/blind_tf_eager_datacontext_eval_neg2_exhaustive.yml) | Exhaustive neg2 blind sweep. |

---

## Integrated workflow (single YAML)

Runs the agent pipeline end-to-end when stages are enabled in the config.

```bash
python scripts/run_integrated_tf_eager_workflow.py --config conf/mESC_tf500_tf_eager_integrated.yml
```

**Stages:** optional acquisition → split manifest → train windows → **train TF-EAGER graph reasoning agent** → test windows → infer (score + export) → evaluation.

**Default inference outputs** (paths can be overridden in YAML):

- `test_scored_edges.csv`
- `test_network.csv`
- `test_flat_evidence.jsonl`

**Evaluation:** e.g. `evaluation/eval_test_by_ratio.json` under the workflow artifact directory when ratio sweeps are configured.

---
| **Literature Validation** | LLM-based verification of top predictions against PubMed abstracts. | `literature_validated.csv`, `literature_classifications/` |

Add NCBI email to  `lit_config.py`  NCBI_EMAIL section.
- For increased speed, Get your own api key at https://www.ncbi.nlm.nih.gov/account/settings/ and create .env with feild NCBI_API_KEY

## Literature Validation Artifacts

If you run the literature validation stage (e.g., via `scripts/run_literature_validation.py`), the following files are produced in your artifact directory:

- **`literature_validated.csv`**: The primary results table. It contains the original model scores plus literature-derived metrics: `lit_score`, `n_supporting` (papers), `pmids`, and `evidence_types` (e.g., ChIP-seq, knockdown).
- **`literature_classifications/`**: A directory containing detailed audit logs for every interaction.
  - Files are named `{TF}_{Target}_classifications.json`.
  - **Key fields**:
    - `evidence_sentence`: The **exact quote** from the paper that supports the interaction.
    - `cell_type_sentences`: Quotes from the paper that grounded the study in your specific cell type.
    - `effective_support`: A boolean flag indicating if the paper passed all quality gates (grounded, correct direction, not negated).
    - `confidence`: The LLM's confidence score for that specific abstract.
___

## Multicontext

Trains **one** TF-EAGER model on combined training windows from multiple contexts, then runs test inference and evaluation **per context**.

```bash
python scripts/run_multicontext_tf_eager_workflow.py \
  --config conf/multicontext_tf_eager/all_datacontext_contexts.yml
```

---

## Blind datacontext (inference and evaluation)

Batch blind runs across prepared DataContext directories (see [`scripts/run_blind_tf_eager_datacontext_eval.py`](scripts/run_blind_tf_eager_datacontext_eval.py)):

```bash
python scripts/run_blind_tf_eager_datacontext_eval.py --config conf/blind_tf_eager_datacontext_eval.yml
```

Use the `neg2` YAML variants to compare negative-construction settings.

The integrated workflow can also build **blind-style** test windows when no split manifest is configured; see [`scripts/run_integrated_tf_eager_workflow.py`](scripts/run_integrated_tf_eager_workflow.py) and your YAML.

---

## Manual pipeline (build → train → infer)

```bash
python scripts/build_tf_eager_windows.py --config conf/mESC_tf500_tf_eager_train.yml
python scripts/train_tf_eager.py --config conf/mESC_tf500_tf_eager_train.yml
python scripts/infer_tf_eager.py --config conf/mESC_tf500_tf_eager_train.yml
```

---

## Train, infer, and evaluate (scripts)

| Step | Script | Notes |
|------|--------|--------|
| Build windows | `scripts/build_tf_eager_windows.py` | YAML with `tf_eager` and paths for split subset and output JSONL |
| Train TF-EAGER | `scripts/train_tf_eager.py` | Requires training `windows_jsonl` and checkpoint `out` in config |
| Infer + export | `scripts/infer_tf_eager.py` | Checkpoint + test `windows_jsonl`; writes CSV / optional JSONL |
| Evaluate | `scripts/eval_grn_agent.py` | Invoked from integrated configs for held-out, ratio-based eval; needs scored outputs, gold edges, and split manifest when doing split-aware eval |

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

Relevant tests include `tests/test_tf_eager_*.py` and `tests/test_integrated_tf_eager_workflow.py`.

---

## Further reading

| Doc | Content |
|-----|---------|
| [`docs/TF-EAGER.md`](docs/TF-EAGER.md) | TF-EAGER behavior, knobs, minimal commands |
| [`docs/TF_EAGER_CONFIG_TUTORIAL.md`](docs/TF_EAGER_CONFIG_TUTORIAL.md) | Config walkthrough |
| [`src/grn_agent/acquisition/README.md`](src/grn_agent/acquisition/README.md) | Multimodal acquisition |
