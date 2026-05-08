TF-EAGER
========

Overview
--------

TF-EAGER is the TF-centered windowed training and inference path for GRN edge
scoring. It builds one evidence graph per TF subgraph sample, scores all
candidate genes in that subgraph jointly, and exports per-edge predictions plus
flattened evidence for evaluation.

Current behavior
----------------

1. Expression preprocessing:
   - expression is transformed with `arcsinh` before coexpression is computed
   - after a TF subgraph is selected, expression is z-score normalized across
     genes for each cell inside that subgraph

2. Candidate subgraph construction:
   - start from coexpression neighbors using `abs(corr) >= corr_threshold`
   - add motif/accessibility rescue genes when enabled
   - add low-coexpression background genes so the TF subgraph also contains
     reliable negative candidates
   - cap the subgraph by random sampling from the candidate pool
   - gold edges do not control which genes enter the TF subgraph

3. Labeling:
   - positives come from the allowed split subset only
   - globally positive TF-gene pairs outside the allowed subset are excluded
     from negative assignment to avoid leakage
   - non-positives are classified as:
     - `reliable_negative`
     - `probable_negative`
     - `ambiguous`

4. Split policy:
   - source TF must be in `TFs.csv`
   - source TF must be present in the expression universe
   - target gene must be present in the expression universe
   - current integrated configs use `node_split_mode: expression` with
     `target_gene_policy: audit`

5. Train/val/test traversal:
   - train uses bootstrap TF subgraph samples
   - val/test currently build one TF subgraph sample per TF
   - val/test still pass through the same candidate labeling and sampling path,
     so they do not simply dump every raw candidate neighbor

Key config knobs
----------------

- `candidates.expression_transform`: currently `arcsinh`
- `candidates.corr_threshold`: coexpression threshold
- `candidates.train_window_neighbors`: TF subgraph size cap, commonly `200`
- `candidates.train_subgraph_bootstraps`: TF bootstrap count, commonly:
  - `50` for the mESC integrated run
  - `5` for the multicontext run
- `candidates.rescue_motif`
- `candidates.rescue_accessibility`
- `negative_ratio`
- `split.node_split_mode`
- `split.target_gene_policy`

Window outputs
--------------

`scripts/build_tf_eager_windows.py` writes JSONL records with:

- `source_tf`
- `context`
- `genes`
- `evidence_graph`

The `evidence_graph` carries:

- expression evidence
- motif evidence
- accessibility evidence
- linkage evidence

Training semantics
------------------

- `sample_weight=1.0`: positive or reliable negative
- `sample_weight=0.5`: probable negative
- `sample_weight=0.0`: ambiguous, ignored by the loss

Reliable negatives are modality-aware:

- multi-omics: low coexpression + low accessibility + low/no motif
- incomplete datasets with accessibility only: low coexpression + low accessibility
- expression-only datasets: low coexpression and not positive in gold

Minimal command flow
--------------------

Single-context:

```bash
python scripts/build_tf_eager_windows.py --config conf/mESC_tf500_tf_eager_train.yml
python scripts/train_tf_eager.py --config conf/mESC_tf500_tf_eager_train.yml
python scripts/infer_tf_eager.py --config conf/mESC_tf500_tf_eager_train.yml
```

Integrated single-context workflow:

```bash
python scripts/run_integrated_tf_eager_workflow.py --config conf/mESC_tf500_tf_eager_integrated.yml
```

Integrated multicontext workflow:

```bash
python scripts/run_multicontext_tf_eager_workflow.py \
  --config conf/multicontext_tf_eager/all_datacontext_contexts.yml
```

To force recomputation from a specific context onward while reusing earlier
artifacts:

```bash
python scripts/run_multicontext_tf_eager_workflow.py \
  --config conf/multicontext_tf_eager/all_datacontext_contexts.yml \
  --force-recompute \
  --start-from hESC
```

Integrated workflow stages
--------------------------

Single-context integrated workflow:

1. optional multimodal acquisition
2. split manifest construction
3. train window build
4. TF-EAGER training
5. test window build
6. inference
7. evaluation

Multicontext workflow:

1. per-context optional multimodal acquisition
2. per-context split manifest construction
3. per-context train/val/test window build
4. combine and shuffle train windows across contexts
5. combine validation windows across contexts
6. train one shared TF-EAGER model
7. run test inference/evaluation per context

Multimodal inputs
-----------------

When `multimodal_manifest` is provided, TF-EAGER consumes:

- `motif_hits`
- `promoter_accessibility`
- manifest QC metadata

Acquisition failures in motif scanning do not have to abort the full workflow:
timeouts from `bedtools getfasta` or `fimo` are downgraded into motif acquisition
failure states so the run can continue without motif support.

Current integrated configs
--------------------------

- [conf/mESC/mESC_tf_eager_integrated.yml](/home/aghktb/GRN_Agent/conf/mESC/mESC_tf_eager_integrated.yml)
- [conf/mESC_tf500_tf_eager_integrated.yml](/home/aghktb/GRN_Agent/conf/mESC_tf500_tf_eager_integrated.yml)
- [conf/multicontext_tf_eager/all_datacontext_contexts.yml](/home/aghktb/GRN_Agent/conf/multicontext_tf_eager/all_datacontext_contexts.yml)

Evaluation artifacts
--------------------

Inference writes:

- `test_scored_edges.csv`
- `test_network.csv`
- `test_flat_evidence.jsonl`

Evaluation is run with `scripts/eval_grn_agent.py` against the split manifest
and held-out subset.
