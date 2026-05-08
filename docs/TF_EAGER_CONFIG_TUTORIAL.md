# TF-EAGER Config Tutorial

This document explains how TF-EAGER configuration works in this repository, with a focus on the workflows used for the NeurIPS manuscript.

It is meant to answer three questions:

1. Which config file should I edit?
2. Which section controls which stage?
3. How do I run ablations without overwriting outputs?

## 1. Mental model

There are two common ways to run TF-EAGER in this repo:

1. `scripts/run_multicontext_tf_eager_workflow.py`
   Use this when training one shared checkpoint across many DataContext contexts.

2. `scripts/run_integrated_tf_eager_workflow.py`
   Use this when running one end-to-end dataset workflow such as mESC.

Both workflows generate smaller resolved configs and then call stage scripts like:

- `scripts/build_tf_eager_windows.py`
- `scripts/train_tf_eager.py`
- `scripts/infer_tf_eager.py`
- `scripts/eval_grn_agent.py`

So the master YAML is not the only config that exists during a run. The workflow script materializes per-stage configs under the workflow artifact directory, then launches the stage scripts with those resolved configs.

## 2. Recommended way to think about configs

Think in layers:

1. `workflow-level config`
   Defines the run identity, artifact root, and which big workflow script is orchestrating the stages.

2. `dataset / context config`
   Defines the biological input, expression files, TF file, species, and context grouping.

3. `window-building config`
   Defines candidate construction, rescue rules, thresholds, and output window JSONL files.

4. `training config`
   Defines TF-EAGER architecture, optimization, DDP, and checkpoint naming.

5. `inference config`
   Defines scored CSV path, network CSV path, threshold, and device.

6. `evaluation config`
   Defines ratio sweeps, report path, and W&B naming.

## 3. Which file should I edit?

### A. Multi-context training

Edit a file like:

- `conf/multicontext_tf_eager/all_datacontext_contexts.yml`
- `conf/multicontext_tf_eager/all_datacontext_contexts_neg1.yml`
- `conf/multicontext_tf_eager/all_datacontext_contexts_neg2.yml`

and run:

```bash
python scripts/run_multicontext_tf_eager_workflow.py --config <that-yaml>
```

Use this for manuscript-scale shared-checkpoint experiments.

### B. Single integrated dataset workflow

Edit a file like:

- `conf/mESC/mESC_tf_eager_integrated.yml`
- `conf/mESC_tf500_tf_eager_integrated.yml`
- `conf/hHep/hHep_only.yml`

and run:

```bash
python scripts/run_integrated_tf_eager_workflow.py --config <that-yaml>
```

Use this for one dataset or one benchmark pipeline.

## 4. Config precedence

In practice, values can come from several places.

### `train_tf_eager.py`

Training resolves values roughly in this order:

1. CLI args
2. `tf_eager.train`
3. `tf_eager`
4. top-level config
5. top-level `train`
6. top-level `scoring`

The TF-EAGER model architecture itself is built from fields found in:

- `tf_eager.train.model`
- `tf_eager.model`
- top-level `model`
- matching top-level fields

Only fields that exist in `TfEagerConfig` are used for the model.

### `build_tf_eager_windows.py`

Window building mainly resolves from:

1. CLI args
2. `tf_eager.build`
3. `tf_eager`
4. top-level config

It also reads from sections like:

- `dataset`
- `candidates`
- `cell_context`
- `scoring`
- top-level workflow defaults

### `infer_tf_eager.py`

Inference resolves roughly from:

1. CLI args
2. `tf_eager.infer`
3. `tf_eager`
4. top-level config
5. top-level `scoring`

Important: inference loads the checkpoint's saved TF-EAGER model config. That means ablation model settings such as `decoder_mode` and `drop_token_kinds` do not need to be duplicated in the inference section if the checkpoint was trained with them.

### `eval_grn_agent.py`

Evaluation is simpler. It reads:

1. CLI args
2. top-level config

So evaluation resolved configs are usually flat.

## 5. Main sections in the multi-context master config

Using `conf/multicontext_tf_eager/all_datacontext_contexts.yml` as the template:

### `workflow`

Controls run identity and artifact root.

Example:

```yaml
workflow:
  id: multicontext_tf_eager/all_datacontext_contexts
  seed: 42
  artifact_root: artifacts
```

Meaning:

- artifacts go under `artifacts/<workflow.id>/`
- random seed defaults from here unless a stage overrides it

### `common`

Shared defaults merged into each expanded context.

Example:

```yaml
common:
  modalities:
    - scrna
    - atac
```

### `acquisition`

Controls multimodal acquisition.

Useful fields:

- `enabled`
- `reuse_if_exists`
- `skip_atac_search`
- `skip_motif`
- `strict`
- `min_promoter_coverage`
- `max_atac_candidates`

If you are running no-motif acquisition for a quick experiment, this is where that happens. But for the manuscript ablations we implemented, you generally do not change acquisition. You change model consumption instead.

### `split`

Controls the split manifest strategy.

Common fields:

- `split_name` or strategy-like fields
- `node_split_mode`
- `target_gene_policy`
- `train_ratio`
- `val_ratio`
- `test_ratio`

This is where leave-one-TF-out behavior is anchored for the manuscript.

### `build_windows`

Controls how train/val/test window JSONLs are built.

Common nested fields:

```yaml
build_windows:
  devices: cpu
  tf_workers: 8
  candidates:
    mode: tf_centered_window
    expression_transform: arcsinh
    corr_threshold: 0.25
    negative_ratio: 5
    train_window_neighbors: 200
    train_subgraph_bootstraps: 5
    val_subgraph_bootstraps: 5
    test_subgraph_bootstraps: 5
    motif_score_threshold: 0.0
    accessibility_threshold: 0.0
    linkage_threshold: 0.0
    rescue_motif: true
    rescue_accessibility: true
```

Important intuition:

- `corr_threshold` affects initial candidate selection
- `negative_ratio` affects window composition and evaluation sampling conventions elsewhere in the workflow
- `rescue_motif` and `rescue_accessibility` affect candidate inclusion at build time
- `train_subgraph_bootstraps` controls how many sampled windows are produced per TF during train-time generation

### `scoring`

Usually carries a device default used by downstream stages.

Example:

```yaml
scoring:
  device: cuda
```

### `context_groups`

This is the heart of the multi-context workflow.

It expands a compact dataset pattern into many contexts.

Example structure:

```yaml
context_groups:
  - base_dir: Data/DataContext
    require_all: true
    variants:
      - nonspecific_chipseq_500
      - string_tf500
    cells:
      - prefix: hHep
        species: human
        cell_type: hepatocyte
```

This means the workflow will enumerate combinations of:

- base directory
- context prefixes
- variant names

and build/train/evaluate across all valid discovered directories.

### `train_tf_eager`

Controls TF-EAGER training.

Typical example:

```yaml
train_tf_eager:
  enabled: true
  model:
    token_layout: edge_compact
    gene_vocab: 1024
  epochs: 100
  lr: 0.0001
  weight_decay: 0.05
  dropout: 0.3
  val_frac: 0.0
  device: cuda
  batch_size: 32
  num_workers: 8
  early_stopping_patience: 10
  early_stopping_min_delta: 0.0001
  wandb: true
  wandb_project: grn-agent-tf-eager
  wandb_run_name: all_datacontext_contexts
  distributed:
    nproc_per_node: 2
    backend: nccl
    standalone: true
```

Key points:

- `model` controls TF-EAGER architecture/config fields
- the rest control optimization and logging
- `distributed` affects whether the workflow uses `torch.distributed.run`

### `infer_tf_eager`

Controls inference and export.

Typical fields:

- `enabled`
- `devices`
- `threshold`
- `topk_per_tf`
- `device`

The multicontext workflow runs inference separately inside each context directory, usually writing outputs to an `evaluation/` subdirectory under that context.

### `evaluation`

Controls evaluation sweep behavior.

Typical fields:

- `enabled`
- `parallel_workers`
- `negative_ratios`
- `k_values`

## 6. Main sections in the integrated workflow config

Using `conf/mESC/mESC_tf_eager_integrated.yml` as the mental model:

### `workflow`

Adds a useful flag:

- `single_artifact_dir: true`

When true, outputs are grouped under one workflow directory.

### `acquisition`

Defines explicit acquisition inputs for one dataset.

Example fields:

- `expr`
- `species`
- `cell_type`
- `cell_line`
- `gold_network`
- `out_manifest`

### `dataset`

Defines how expression is loaded.

Common fields:

- `mode: beeline_csv` or `mode: npy`
- `dataset_id`
- `species`
- `expression_path`
- `tf_file`
- `modalities`

### `cell_context`

Controls context labeling and optional Scanpy use.

### `candidates`

This is the window-building candidate policy for the integrated path.

It includes fields such as:

- `corr_threshold`
- `negative_ratio`
- `train_window_neighbors`
- `train_subgraph_bootstraps`
- `train_include_positives`
- `expr_weak_percentile`
- `expr_probable_percentile`
- `rescue_motif`
- `rescue_accessibility`
- `rescue_max_per_tf`

### `split`

Defines gold edge file, output split manifest path, fold id, and ratios.

### `tf_eager`

This section carries TF-EAGER-specific common values across build/train/infer.

Typical fields:

- `strategy`
- `fold_id`
- `checkpoint`
- `train_windows_jsonl`
- `test_windows_jsonl`
- `window_size`

### `build_train_windows` and `build_test_windows`

These stage sections mainly control whether those build steps are enabled and whether outputs are reused.

### `train_tf_eager`

Same idea as the multicontext workflow, but usually simpler.

### `infer_tf_eager`

Defines:

- `scored_csv`
- `network_csv`
- `evidence_jsonl`
- `threshold`
- `topk_per_tf`
- `device`

### `evaluation`

Defines:

- `gold_edges`
- `strategy`
- `fold_id`
- `subset`
- `negative_ratios`
- `out_report`
- optional W&B fields

## 7. TF-EAGER model config fields

These fields come from `TfEagerConfig` and belong under a `model:` block, usually inside `train_tf_eager`.

Current supported model fields are:

```yaml
model:
  d_model: 128
  n_heads: 4
  n_encoder_layers: 2
  dropout: 0.1
  tf_vocab: 8192
  gene_vocab: 8192
  context_vocab: 1024
  target_pos_vocab: 101
  token_layout: evidence_tokens
  use_tf_identity: true
  use_gene_identity: true
  use_context_identity: true
  drop_token_kinds: []
  decoder_mode: staged
```

Most common ones you will edit:

- `token_layout`
- `gene_vocab`
- `dropout`
- `drop_token_kinds`
- `decoder_mode`

### `token_layout`

Allowed values:

- `evidence_tokens`
- `edge_compact`

Current manuscript multicontext configs use `edge_compact`.

### `drop_token_kinds`

This is the new ablation knob.

Allowed names:

- `motif`
- `accessibility`
- `linkage`
- `literature`
- also short aliases like `acc`, `link`, `lit`

Example functional-only mechanistic ablation:

```yaml
train_tf_eager:
  model:
    drop_token_kinds:
      - motif
      - accessibility
      - linkage
```

This works for both `evidence_tokens` and `edge_compact`.

### `decoder_mode`

This is the other new ablation knob.

Allowed values:

- `staged`
- `single_stage`

Example:

```yaml
train_tf_eager:
  model:
    decoder_mode: single_stage
```

## 8. Output naming and overwrite avoidance

This matters a lot for ablations.

### Multi-context workflow output naming

By default, the multicontext workflow writes:

- checkpoint: `tf_eager/tf_eager_bootstrap_v2.pt`
- per-context scored CSV: `evaluation/test_scored_edges.csv`
- per-context network CSV: `evaluation/test_network.csv`
- per-context flat evidence: `evaluation/test_flat_evidence.jsonl`
- per-context eval report: `evaluation/eval_test_by_ratio.json`

We added new optional fields so ablations can coexist.

### New multi-context naming fields

Under `train_tf_eager`:

- `checkpoint_name`

Under `infer_tf_eager`:

- `output_suffix`
- `scored_filename`
- `network_filename`
- `evidence_filename`
- `resolved_config_name`

Under `evaluation`:

- `report_filename`
- `resolved_config_name`

### Recommended ablation naming pattern

Functional-only run:

```yaml
workflow:
  id: multicontext_tf_eager/all_datacontext_contexts_functional_only

train_tf_eager:
  checkpoint_name: tf_eager_functional_only.pt
  wandb_run_name: all_datacontext_contexts_functional_only
  model:
    token_layout: edge_compact
    gene_vocab: 1024
    drop_token_kinds:
      - motif
      - accessibility
      - linkage

infer_tf_eager:
  output_suffix: _functional_only

evaluation:
  report_filename: eval_test_by_ratio_functional_only.json
```

Single-stage run:

```yaml
workflow:
  id: multicontext_tf_eager/all_datacontext_contexts_single_stage

train_tf_eager:
  checkpoint_name: tf_eager_single_stage.pt
  wandb_run_name: all_datacontext_contexts_single_stage
  model:
    token_layout: edge_compact
    gene_vocab: 1024
    decoder_mode: single_stage

infer_tf_eager:
  output_suffix: _single_stage

evaluation:
  report_filename: eval_test_by_ratio_single_stage.json
```

## 9. How stage outputs are connected

### Window builder output

`build_tf_eager_windows.py` writes JSONL windows.

Those become input to training and inference.

### Trainer output

`train_tf_eager.py` writes a checkpoint `.pt` file containing:

- `model_state`
- saved model `config`
- best validation metrics
- training metadata

### Inference output

`infer_tf_eager.py` writes:

- scored per-edge CSV
- optional thresholded/top-k network CSV
- flat evidence JSONL

### Evaluation output

`eval_grn_agent.py` consumes:

- scored or network CSV
- flat evidence JSONL
- split manifest and/or gold edges

and writes a JSON report.

## 10. What to edit for common tasks

### Change train hyperparameters

Edit `train_tf_eager`:

- `epochs`
- `lr`
- `weight_decay`
- `batch_size`
- `num_workers`
- `early_stopping_patience`

### Change architecture

Edit `train_tf_eager.model`:

- `token_layout`
- `gene_vocab`
- `d_model`
- `n_heads`
- `n_encoder_layers`
- `drop_token_kinds`
- `decoder_mode`

### Change candidate/window construction

Edit:

- multi-context: `build_windows.candidates`
- integrated: `candidates`

### Change inference threshold or top-k

Edit `infer_tf_eager`:

- `threshold`
- `topk_per_tf`

### Change evaluation ratio sweep

Edit `evaluation`:

- `negative_ratios`
- `k_values`
- `negative_repeats`

## 11. Safe ablation workflow recipe

When creating a new manuscript ablation config, change at least these:

1. `workflow.id`
2. `train_tf_eager.checkpoint_name` or checkpoint path
3. `train_tf_eager.wandb_run_name`
4. `infer_tf_eager.output_suffix`
5. `evaluation.report_filename`

That avoids collisions in:

- artifact directories
- checkpoint names
- per-context inference outputs
- evaluation JSON reports
- W&B runs

## 12. Common mistakes

### Mistake 1: editing inference config to change the model ablation

For TF-EAGER architectural ablations, edit `train_tf_eager.model`, not `infer_tf_eager`.
The checkpoint carries the model config.

### Mistake 2: changing only the checkpoint name but not output files

In multicontext runs, each context writes evaluation files under its own `evaluation/` folder. If filenames stay the same, one run can overwrite another.

### Mistake 3: mixing build-time and model-time ablations

- Build-time changes: rescue rules, thresholds, candidate pool
- Model-time changes: token dropping, decoder mode, embedding/vocab/layout changes

These answer different scientific questions.

### Mistake 4: forgetting that `edge_compact` and `evidence_tokens` behave differently

The same high-level ablation may still be implemented differently internally.
In this repo, `drop_token_kinds` now works for both, but the compact layout zeroes slices inside one token, while the evidence-token layout removes whole token types.

## 13. Minimal examples

### A. Minimal multi-context config idea

```yaml
workflow:
  id: multicontext_tf_eager/example_run
  seed: 42
  artifact_root: artifacts

build_windows:
  candidates:
    corr_threshold: 0.25
    negative_ratio: 5

train_tf_eager:
  enabled: true
  model:
    token_layout: edge_compact
    gene_vocab: 1024
  epochs: 100
  lr: 0.0001
  batch_size: 32
  device: cuda

infer_tf_eager:
  enabled: true
  threshold: 0.05
  topk_per_tf: 100

evaluation:
  enabled: true
  negative_ratios: "1,2,5,10"
  k_values: "10,50,100"
```

### B. Minimal functional-only ablation block

```yaml
train_tf_eager:
  model:
    drop_token_kinds:
      - motif
      - accessibility
      - linkage
```

### C. Minimal single-stage ablation block

```yaml
train_tf_eager:
  model:
    decoder_mode: single_stage
```

## 14. How to inspect what a run actually used

Best practice:

1. keep the master YAML you launched
2. inspect the resolved YAMLs written by the workflow
3. inspect the checkpoint metadata

Useful places:

- multi-context workflow directory:
  - `tf_eager_train.resolved.yml`
  - per-context `infer*.resolved.yml`
  - per-context `evaluation*.resolved.yml`

- integrated workflow directory:
  - `split.resolved.yml`
  - `build_train_windows.resolved.yml`
  - `train_tf_eager.resolved.yml`
  - `infer_tf_eager.resolved.yml`
  - `evaluation.resolved.yml`

These resolved configs are often the best debugging tool because they show what the workflow actually passed to each stage.

## 15. Bottom line

If you are working on the manuscript and want to stay sane:

1. edit the master YAML for the workflow you are using
2. keep model ablations inside `train_tf_eager.model`
3. keep data/candidate ablations in `build_windows.candidates` or `candidates`
4. always change `workflow.id` and output names for ablations
5. inspect resolved YAMLs when behavior is confusing
