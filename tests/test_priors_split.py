import numpy as np

from grn_agent.agents.priors import compute_priors_for_pair, compute_priors_for_pairs


def test_priors_depend_on_train_mask():
    rng = np.random.default_rng(0)
    expr = rng.standard_normal((30, 10))
    genes = [f"G{i}" for i in range(10)]
    m1 = np.zeros(30, dtype=bool)
    m1[:20] = True
    m2 = np.zeros(30, dtype=bool)
    m2[10:] = True
    p1 = compute_priors_for_pair(expr, m1, genes, "G0", "G1", split_id="a", seed=1)
    p2 = compute_priors_for_pair(expr, m2, genes, "G0", "G1", split_id="b", seed=1)
    assert p1.ensemble_prior != p2.ensemble_prior


def test_vectorized_priors_match_pairwise():
    rng = np.random.default_rng(2)
    expr = rng.standard_normal((40, 12))
    genes = [f"G{i}" for i in range(12)]
    mask = np.zeros(40, dtype=bool)
    mask[:30] = True
    pairs = [("G0", "G1"), ("G2", "G3"), ("G4", "G5")]

    batched = compute_priors_for_pairs(expr, mask, genes, pairs, split_id="x", seed=7)

    for tf, gene in pairs:
        single = compute_priors_for_pair(expr, mask, genes, tf, gene, split_id="x", seed=7)
        got = batched[(tf, gene)]
        assert np.isclose(got.p_grnboost, single.p_grnboost)
        assert np.isclose(got.p_genie3, single.p_genie3)
        assert np.isclose(got.p_pidc, single.p_pidc)
        assert np.isclose(got.scenic_regulon_support, single.scenic_regulon_support)
        assert np.isclose(got.bootstrap_stability, single.bootstrap_stability)
        assert np.isclose(got.ensemble_prior, single.ensemble_prior)
