from __future__ import annotations

import torch

from grn_agent.models.tf_eager import TfEagerConfig, TfEagerWindowModel, window_record_to_batch
from grn_agent.models.tf_eager.window_batch import (
    EDGE_COMPACT_MAX_TOKENS,
    MAX_TOKENS,
    NO_TARGET_POSITION,
    TF_EAGER_WINDOW_SIZE,
    TOKEN_LAYOUT_EDGE_COMPACT,
    VALUE_DIM,
    TfEagerTokenKind,
)


def _window_record() -> dict:
    return {
        "schema": "tf_eager_window_v1",
        "source_tf": "tf_a",
        "context": {
            "context_id": "ctx_1",
            "cell_type": "stem",
            "n_cells": 12,
            "n_module_genes": 3,
        },
        "window_index": 0,
        "window_size": TF_EAGER_WINDOW_SIZE,
        "genes": [
            {
                "target_gene": "gene_b",
                "label": 1,
                "sample_weight": 2.0,
                "evidence": {
                    "correlation": 0.72,
                    "in_same_module": True,
                    "z_t": 0.4,
                    "z_g": 0.2,
                    "activity_t": 0.72,
                    "motif_present": True,
                    "accessibility": 1.5,
                    "ensemble_prior": 0.61,
                },
                "expression": {
                    "mean_expr_t": 3.2,
                    "mean_expr_g": 1.4,
                    "dropout_t": 0.1,
                    "dropout_g": 0.2,
                },
                "network": {"shared_neighbors": 4},
                "motif": {"motif_present": True, "motif_score": 9.5, "n_motif_regions": 2},
                "accessibility": {"peak_accessibility": 1.5, "celltype_specificity": 0.8},
                "linkage": {"peak_to_gene_linked": True, "linkage_score": 0.7},
                "prior": {"ensemble_prior": 0.61, "bootstrap_stability": 0.5},
                "orthology": {"ortholog_support": 1.0, "conserved_in_mouse": True},
            },
            {
                "target_gene": "gene_c",
                "label": 0,
                "sample_weight": 1.0,
                "evidence": {"correlation": -0.03, "in_same_module": True},
                "expression": {"z_t": 0.4, "z_g": -0.1, "activity_t": -0.03},
                "network": {"pearson_r": -0.03, "shared_neighbors": 0},
                "motif": {"motif_present": False, "motif_score": 0.0, "n_motif_regions": 0},
                "accessibility": {"peak_accessibility": 0.0},
                "linkage": {"peak_to_gene_linked": False, "linkage_score": 0.0},
                "prior": {"ensemble_prior": 0.02},
                "orthology": {},
            },
        ],
    }


def test_window_record_to_batch_shapes_masks_and_modalities():
    batch = window_record_to_batch(_window_record())

    assert batch.token_kind.shape == (1, MAX_TOKENS)
    assert batch.x_value.shape == (1, MAX_TOKENS, VALUE_DIM)
    assert batch.token_target_pos.shape == (1, MAX_TOKENS)
    assert batch.gene_idx.shape == (1, TF_EAGER_WINDOW_SIZE)
    assert batch.gene_pos[0, :3].tolist() == [0, 1, NO_TARGET_POSITION]
    assert batch.gene_mask[0, :3].tolist() == [1.0, 1.0, 0.0]
    assert batch.labels[0, :2].tolist() == [1.0, 0.0]
    assert batch.sample_weight[0, :2].tolist() == [2.0, 1.0]
    assert batch.modality.tolist() == [[1.0, 1.0, 1.0, 1.0]]
    assert int((batch.token_kind == TfEagerTokenKind.TF).sum()) == 1
    assert int((batch.token_kind == TfEagerTokenKind.CTX).sum()) == 1
    assert batch.token_target_pos[0, 0].item() == NO_TARGET_POSITION
    assert 0 in batch.token_target_pos[batch.token_kind == TfEagerTokenKind.EXPR].tolist()
    assert 1 in batch.token_target_pos[batch.token_kind == TfEagerTokenKind.EXPR].tolist()
    assert int(batch.mech_mask.sum()) > 0
    assert int(batch.func_mask.sum()) > 0


def test_tf_eager_forward_backward_smoke():
    batch = window_record_to_batch(_window_record())
    model = TfEagerWindowModel(
        TfEagerConfig(
            d_model=16,
            n_heads=4,
            n_encoder_layers=1,
            dropout=0.0,
            tf_vocab=8192,
            gene_vocab=8192,
            context_vocab=1024,
        )
    )

    logits = model(batch)
    assert logits.shape == (1, TF_EAGER_WINDOW_SIZE)
    assert torch.isfinite(logits).all()

    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.labels, reduction="none")
        * batch.gene_mask
    ).sum() / batch.gene_mask.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None


def test_compact_edge_token_layout_and_reduced_gene_vocab():
    cfg = TfEagerConfig(
        d_model=16,
        n_heads=4,
        n_encoder_layers=1,
        dropout=0.0,
        token_layout=TOKEN_LAYOUT_EDGE_COMPACT,
        tf_vocab=1024,
        gene_vocab=128,
        context_vocab=128,
    )
    batch = window_record_to_batch(
        _window_record(),
        token_layout=cfg.token_layout,
        tf_vocab=cfg.tf_vocab,
        gene_vocab=cfg.gene_vocab,
        context_vocab=cfg.context_vocab,
    )

    assert batch.token_kind.shape == (1, EDGE_COMPACT_MAX_TOKENS)
    assert int(batch.token_mask.sum().item()) == 4
    assert int((batch.token_kind == TfEagerTokenKind.GENE).sum().item()) == 2
    assert int((batch.token_kind == TfEagerTokenKind.EXPR).sum().item()) == 0
    assert int((batch.token_kind == TfEagerTokenKind.MOTIF).sum().item()) == 0
    assert int(batch.gene_idx.max().item()) < cfg.gene_vocab

    model = TfEagerWindowModel(cfg)
    logits = model(batch)
    assert logits.shape == (1, TF_EAGER_WINDOW_SIZE)
    assert torch.isfinite(logits).all()
