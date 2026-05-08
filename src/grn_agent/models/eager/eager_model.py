"""EAGER: node init + typed MPNN + staged cross-attention + binary logit."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from grn_agent.models.eager.graph_batch import NUM_NODE_KINDS, VALUE_DIM
from grn_agent.models.eager.graph_batch import EagerGraphBatch
from grn_agent.schemas import RelationType


def _n_edge_types() -> int:
    return len(list(RelationType))


@dataclass
class EagerRegulatorConfig:
    d_model: int = 128
    n_mpnn_layers: int = 3
    num_edge_types: int = field(default_factory=_n_edge_types)
    n_heads: int = 4
    dropout: float = 0.1
    tf_vocab: int = 8192
    gene_vocab: int = 8192
    context_vocab: int = 1024
    conf_bins: int = 32


class EagerRegulator(nn.Module):
    """
    h_i^(0) = e_type + e_value + e_conf + e_context
    H^graph = typed MPNN layers
    z0 = e_tf + e_gene + e_ctx; three cross-attn stages; logit = linear(z3)
    """

    def __init__(self, cfg: EagerRegulatorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EagerRegulatorConfig()
        c = self.cfg
        d = c.d_model
        n_rel = c.num_edge_types
        self.node_kind_emb = nn.Embedding(NUM_NODE_KINDS, d)
        self.value_mlp = nn.Sequential(
            nn.Linear(VALUE_DIM, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.conf_proj = nn.Linear(1, d)
        self.ctx_node_emb = nn.Embedding(c.context_vocab, d)
        self.emb_tf = nn.Embedding(c.tf_vocab, d)
        self.emb_gene = nn.Embedding(c.gene_vocab, d)
        self.emb_ctx_edge = nn.Embedding(c.context_vocab, d)

        self.msg_lin = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_rel)])
        self.mpnn_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(c.n_mpnn_layers)])
        self.mpnn_self = nn.ModuleList([nn.Linear(d, d) for _ in range(c.n_mpnn_layers)])

        self.stage1 = nn.MultiheadAttention(d, c.n_heads, dropout=c.dropout, batch_first=True)
        self.stage2 = nn.MultiheadAttention(d, c.n_heads, dropout=c.dropout, batch_first=True)
        self.stage3 = nn.MultiheadAttention(d, c.n_heads, dropout=c.dropout, batch_first=True)
        self.ln_s1 = nn.LayerNorm(d)
        self.ln_s2 = nn.LayerNorm(d)
        self.ln_s3 = nn.LayerNorm(d)
        self.ff1 = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ff2 = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ff3 = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Linear(d, 1)
        self.dropout = nn.Dropout(c.dropout)

    def _h0(self, batch: EagerGraphBatch) -> torch.Tensor:
        b, n = batch.node_kind.shape
        d = self.cfg.d_model
        et = self.node_kind_emb(batch.node_kind)
        ev = self.value_mlp(batch.x_value)
        ec = self.conf_proj(batch.conf.unsqueeze(-1))
        ectx = self.ctx_node_emb(batch.context_idx).unsqueeze(1).expand(b, n, d)
        return et + ev + ec + ectx

    def _mpnn(self, h: torch.Tensor, batch: EagerGraphBatch) -> torch.Tensor:
        if h.shape[0] != 1:
            raise NotImplementedError("MPNN in this build supports batch size 1 per forward")
        edge_index = batch.edge_index
        edge_type = batch.edge_type
        node_mask = batch.node_mask[0:1]  # (1,N)
        h1 = h[0]
        n = h1.shape[0]
        for layer in range(self.cfg.n_mpnn_layers):
            agg = torch.zeros_like(h1)
            ecount = torch.zeros(n, 1, device=h.device, dtype=h.dtype)
            for r in range(len(self.msg_lin)):
                rel_mask = edge_type == r
                if not bool(rel_mask.any().item()):
                    continue
                src = edge_index[0, rel_mask].long()
                dst = edge_index[1, rel_mask].long()
                msg = self.msg_lin[r](h1[src])
                agg = agg.index_add(0, dst, msg)
                ones = torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype)
                ecount = ecount.index_add(0, dst, ones)
            ecount = torch.clamp(ecount, min=1.0)
            agg = agg / ecount
            upd = self.mpnn_self[layer](h1) + agg
            upd = self.mpnn_norms[layer](upd)
            h1 = h1 + F.gelu(upd)
            m = node_mask.squeeze(0).unsqueeze(-1)
            h1 = h1 * m
        return h1.unsqueeze(0)

    def forward(
        self,
        batch: EagerGraphBatch,
        *,
        return_attention_summary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Return logits (B,) and optional attention summary. Batch size must be 1."""
        if batch.node_kind.shape[0] != 1:
            raise NotImplementedError("EagerRegulator forward supports batch size 1 in this build")
        h0 = self._h0(batch)
        h = self._mpnn(h0, batch)
        z0 = self.emb_tf(batch.tf_idx) + self.emb_gene(batch.gene_idx) + self.emb_ctx_edge(batch.context_idx)
        n_real = batch.node_mask[0] > 0.5
        mask_pad = ~n_real
        m_acc_mot = (batch.modality[0, 1] + batch.modality[0, 2]) > 0.5
        has_mech_tok = (batch.mech_mask[0] * batch.node_mask[0]).sum() > 0.5
        use_mech = bool(m_acc_mot.item() and has_mech_tok.item())
        mech_m = batch.mech_mask[0] > 0.5
        bad1 = mask_pad | (~mech_m)
        z = z0
        attn1 = None
        if use_mech:
            out1, attn1 = self.stage1(
                z.unsqueeze(1),
                h,
                h,
                key_padding_mask=bad1.unsqueeze(0),
                need_weights=return_attention_summary,
            )
            z1p = self.ln_s1(z + self.dropout(out1.squeeze(1)))
            z = z1p + self.dropout(self.ff1(z1p))

        has_func = (batch.func_mask[0] * batch.node_mask[0]).sum() > 0.5
        if bool(has_func.item()):
            fn = batch.func_mask[0] > 0.5
            bad2 = mask_pad | (~fn)
        else:
            bad2 = mask_pad
        out2, attn2 = self.stage2(
            z.unsqueeze(1),
            h,
            h,
            key_padding_mask=bad2.unsqueeze(0),
            need_weights=return_attention_summary,
        )
        z = self.ln_s2(z + self.dropout(out2.squeeze(1)))
        z = z + self.dropout(self.ff2(z))

        out3, attn3 = self.stage3(
            z.unsqueeze(1),
            h,
            h,
            key_padding_mask=mask_pad.unsqueeze(0),
            need_weights=return_attention_summary,
        )
        z = self.ln_s3(z + self.dropout(out3.squeeze(1)))
        z = z + self.dropout(self.ff3(z))
        logit = self.head(z).squeeze(-1)
        if not return_attention_summary:
            return logit

        summ: dict[str, float] = {}
        summ.update(self._attention_stats("stage2", attn2, batch))
        summ.update(self._attention_stats("stage3", attn3, batch))
        if attn1 is not None:
            summ.update(self._attention_stats("stage1", attn1, batch))
        return logit, summ

    def _attention_stats(self, prefix: str, attn: torch.Tensor | None, batch: EagerGraphBatch) -> dict[str, float]:
        if attn is None:
            return {}
        # attn shape with batch_first=True and tgt_len=1:
        # either (B, tgt_len, src_len) or (B, num_heads, tgt_len, src_len)
        if attn.dim() == 4:
            w = attn.mean(dim=1)  # average across heads
        else:
            w = attn
        wv = w[0, 0]  # (src_len,)
        node_mask = (batch.node_mask[0] > 0.5)
        mech_mask = (batch.mech_mask[0] > 0.5) & node_mask
        func_mask = (batch.func_mask[0] > 0.5) & node_mask
        real = node_mask
        s_real = float(wv[real].sum().item()) if bool(real.any().item()) else 0.0
        s_mech = float(wv[mech_mask].sum().item()) if bool(mech_mask.any().item()) else 0.0
        s_func = float(wv[func_mask].sum().item()) if bool(func_mask.any().item()) else 0.0
        return {
            f"{prefix}_attn_real": s_real,
            f"{prefix}_attn_mech": s_mech,
            f"{prefix}_attn_func": s_func,
        }
