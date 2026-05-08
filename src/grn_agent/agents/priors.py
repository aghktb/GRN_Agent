from __future__ import annotations

import numpy as np

from grn_agent.agents import priors_engines as pe
from grn_agent.schemas import PriorBundle


def compute_priors_for_pairs(
    expression_full: np.ndarray,
    train_mask: np.ndarray,
    gene_symbols: list[str],
    pairs: list[tuple[str, str]],
    split_id: str,
    n_bootstrap: int = 8,
    seed: int = 0,
    device: str | None = None,
    chunk_size: int = 8192,
) -> dict[tuple[str, str], PriorBundle]:
    """
    Fit proxy priors for many TF-target pairs on one split.

    The formulas match ``compute_priors_for_pair`` but run correlations and
    bootstrap stability in batches. If ``device`` requests CUDA and torch is
    available, the numeric correlation batches run on GPU.
    """
    _ = split_id
    norm_pairs = [(str(tf).strip().upper(), str(g).strip().upper()) for tf, g in pairs]
    out = {pair: PriorBundle(ensemble_prior=0.0) for pair in norm_pairs}
    if not norm_pairs:
        return out

    gene_to_idx = {str(g).strip().upper(): i for i, g in enumerate(gene_symbols)}
    ti = np.asarray([gene_to_idx.get(tf, -1) for tf, _ in norm_pairs], dtype=np.int64)
    gi = np.asarray([gene_to_idx.get(g, -1) for _, g in norm_pairs], dtype=np.int64)
    valid = (ti >= 0) & (gi >= 0)
    if not np.any(valid):
        return out

    backend = _resolve_numeric_backend(device)
    expr_t = np.asarray(expression_full[train_mask], dtype=np.float64)
    if backend == "torch_cuda":
        r_valid = _pairwise_abs_corr_torch(expr_t, ti[valid], gi[valid], chunk_size=chunk_size)
        bs_valid = _bootstrap_stability_many_torch(
            expression_full,
            train_mask,
            ti[valid],
            gi[valid],
            n_boot=n_bootstrap,
            seed=seed,
            chunk_size=chunk_size,
        )
    else:
        r_valid = _pairwise_abs_corr_numpy(expr_t, ti[valid], gi[valid], chunk_size=chunk_size)
        bs_valid = _bootstrap_stability_many_numpy(
            expression_full,
            train_mask,
            ti[valid],
            gi[valid],
            n_boot=n_bootstrap,
            seed=seed,
            chunk_size=chunk_size,
        )

    r = np.zeros(len(norm_pairs), dtype=np.float64)
    bs = np.zeros(len(norm_pairs), dtype=np.float64)
    r[valid] = r_valid
    bs[valid] = bs_valid

    p_grnboost = np.clip(r, 0.0, 1.0)
    p_genie3 = np.minimum(1.0, (r + 0.1) / 1.1 * 0.95)
    p_pidc = np.minimum(1.0, r * 0.85)
    p_scenic = 0.5 * r
    ens = np.clip((p_grnboost + p_genie3 + p_pidc + p_scenic) / 4.0 * (0.5 + 0.5 * bs), 0.0, 1.0)

    for i, pair in enumerate(norm_pairs):
        if not valid[i]:
            continue
        out[pair] = PriorBundle(
            p_grnboost=float(p_grnboost[i]),
            p_genie3=float(p_genie3[i]),
            p_pidc=float(p_pidc[i]),
            scenic_regulon_support=float(p_scenic[i]),
            bootstrap_stability=float(bs[i]),
            ensemble_prior=float(ens[i]),
        )
    return out


def _resolve_numeric_backend(device: str | None) -> str:
    dev = str(device or "").strip().lower()
    if dev in {"cuda", "gpu", "torch_cuda"} or dev.startswith("cuda:"):
        try:
            import torch

            if torch.cuda.is_available():
                return "torch_cuda"
        except Exception:
            return "numpy"
    return "numpy"


def _pairwise_abs_corr_numpy(
    expr: np.ndarray,
    ti: np.ndarray,
    gi: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    if expr.shape[0] < 2 or len(ti) == 0:
        return np.zeros(len(ti), dtype=np.float64)
    x = np.asarray(expr, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    out = np.empty(len(ti), dtype=np.float64)
    for start in range(0, len(ti), chunk_size):
        end = min(start + chunk_size, len(ti))
        a = x[:, ti[start:end]]
        b = x[:, gi[start:end]]
        num = np.sum(a * b, axis=0)
        den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
        out[start:end] = np.divide(np.abs(num), den, out=np.zeros(end - start, dtype=np.float64), where=den > 1e-12)
    return np.clip(out, 0.0, 1.0)


def _pairwise_abs_corr_torch(
    expr: np.ndarray,
    ti: np.ndarray,
    gi: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    if expr.shape[0] < 2 or len(ti) == 0:
        return np.zeros(len(ti), dtype=np.float64)
    import torch

    dev = torch.device("cuda")
    x = torch.as_tensor(expr, dtype=torch.float64, device=dev)
    x = x - x.mean(dim=0, keepdim=True)
    ti_t = torch.as_tensor(ti, dtype=torch.long, device=dev)
    gi_t = torch.as_tensor(gi, dtype=torch.long, device=dev)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(ti), chunk_size):
        end = min(start + chunk_size, len(ti))
        a = x.index_select(1, ti_t[start:end])
        b = x.index_select(1, gi_t[start:end])
        num = torch.sum(a * b, dim=0).abs()
        den = torch.sqrt(torch.sum(a * a, dim=0) * torch.sum(b * b, dim=0))
        chunks.append(torch.where(den > 1e-12, num / den, torch.zeros_like(num)))
    return torch.clamp(torch.cat(chunks), 0.0, 1.0).detach().cpu().numpy()


def _bootstrap_stability_many_numpy(
    expr_full: np.ndarray,
    train_mask: np.ndarray,
    ti: np.ndarray,
    gi: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray:
    train_idx = np.where(train_mask)[0]
    if len(train_idx) < 5 or n_boot <= 0:
        return np.zeros(len(ti), dtype=np.float64)
    rng = np.random.default_rng(seed)
    vals = np.empty((n_boot, len(ti)), dtype=np.float64)
    for b in range(n_boot):
        samp = rng.choice(train_idx, size=len(train_idx), replace=True)
        r = _pairwise_abs_corr_numpy(np.asarray(expr_full[samp, :], dtype=np.float64), ti, gi, chunk_size=chunk_size)
        vals[b, :] = _proxy_ensemble_without_scenic(r)
    return np.maximum(0.0, 1.0 - vals.std(axis=0))


def _bootstrap_stability_many_torch(
    expr_full: np.ndarray,
    train_mask: np.ndarray,
    ti: np.ndarray,
    gi: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray:
    train_idx = np.where(train_mask)[0]
    if len(train_idx) < 5 or n_boot <= 0:
        return np.zeros(len(ti), dtype=np.float64)
    rng = np.random.default_rng(seed)
    vals = np.empty((n_boot, len(ti)), dtype=np.float64)
    for b in range(n_boot):
        samp = rng.choice(train_idx, size=len(train_idx), replace=True)
        r = _pairwise_abs_corr_torch(np.asarray(expr_full[samp, :], dtype=np.float64), ti, gi, chunk_size=chunk_size)
        vals[b, :] = _proxy_ensemble_without_scenic(r)
    return np.maximum(0.0, 1.0 - vals.std(axis=0))


def _proxy_ensemble_without_scenic(r: np.ndarray) -> np.ndarray:
    return (
        np.clip(r, 0.0, 1.0)
        + np.minimum(1.0, (r + 0.1) / 1.1 * 0.95)
        + np.minimum(1.0, r * 0.85)
    ) / 3.0


def compute_priors_for_pair(
    expression_full: np.ndarray,
    train_mask: np.ndarray,
    gene_symbols: list[str],
    source_tf: str,
    target_gene: str,
    split_id: str,
    n_bootstrap: int = 8,
    seed: int = 0,
) -> PriorBundle:
    """
    Fit priors using **only** rows where train_mask is True (SDD §3.5).

    `split_id` is included in cache keys by callers; passed for provenance in payloads.
    """
    _ = split_id
    expr_t = expression_full[train_mask]
    try:
        ti = gene_symbols.index(source_tf)
        gi = gene_symbols.index(target_gene)
    except ValueError:
        return PriorBundle(ensemble_prior=0.0)

    pg = pe.proxy_grnboost(expr_t, ti, gi)
    pg3 = pe.proxy_genie3(expr_t, ti, gi)
    pp = pe.proxy_pidc(expr_t, ti, gi)
    ps = pe.proxy_scenic(expr_t, ti, gi)
    bs = pe.bootstrap_stability(expression_full, train_mask, ti, gi, n_boot=n_bootstrap, seed=seed)
    ens = float(np.clip((pg + pg3 + pp + ps) / 4.0 * (0.5 + 0.5 * bs), 0.0, 1.0))
    return PriorBundle(
        p_grnboost=pg,
        p_genie3=pg3,
        p_pidc=pp,
        scenic_regulon_support=ps,
        bootstrap_stability=bs,
        ensemble_prior=ens,
    )
