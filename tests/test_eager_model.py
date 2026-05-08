import torch

from grn_agent.models.eager.eager_model import EagerRegulator
from grn_agent.models.eager.graph_batch import EagerGraphBatch


def _fake_batch(modality=(1.0, 1.0, 1.0, 1.0)) -> EagerGraphBatch:
    kinds = [0, 3, 5] + [9] * 29
    return EagerGraphBatch(
        node_kind=torch.tensor([kinds], dtype=torch.long),
        x_value=torch.zeros((1, 32, 32), dtype=torch.float32),
        conf=torch.zeros((1, 32), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_type=torch.tensor([0, 1, 2], dtype=torch.long),
        node_mask=torch.tensor([[1.0, 1.0, 1.0] + [0.0] * 29], dtype=torch.float32),
        modality=torch.tensor([list(modality)], dtype=torch.float32),
        mech_mask=torch.tensor([[0.0, 0.0, 1.0] + [0.0] * 29], dtype=torch.float32),
        func_mask=torch.tensor([[0.0, 1.0, 0.0] + [0.0] * 29], dtype=torch.float32),
        context_idx=torch.tensor([1], dtype=torch.long),
        tf_idx=torch.tensor([2], dtype=torch.long),
        gene_idx=torch.tensor([3], dtype=torch.long),
    )


def test_h0_shape():
    m = EagerRegulator()
    b = _fake_batch()
    h0 = m._h0(b)
    assert tuple(h0.shape) == (1, 32, m.cfg.d_model)


def test_forward_logit_shape():
    m = EagerRegulator()
    b = _fake_batch()
    y = m(b)
    assert tuple(y.shape) == (1,)


def test_stage1_gate_when_no_mechanistic_modalities():
    m = EagerRegulator()
    b = _fake_batch(modality=(1.0, 0.0, 0.0, 0.0))
    y = m(b)
    assert torch.isfinite(y).all()


def test_binary_bce_loss_runs():
    m = EagerRegulator()
    b = _fake_batch()
    y = m(b)
    t = torch.tensor([1.0], dtype=torch.float32)
    loss = torch.nn.BCEWithLogitsLoss()(y, t)
    assert float(loss.item()) >= 0.0
