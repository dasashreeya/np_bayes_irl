import numpy as np

from evaluate import UNMATCHED_L2, evaluate, matched_weight_error
from summarize import PosteriorSummary

W1 = np.array([1.0, -1.0, 0.0])
W2 = np.array([-1.0, 1.0, 0.0])


def test_extra_clusters_are_counted_not_hidden():
    # The old best_match_error scored this as perfect. Hungarian matching
    # still gives 0 error on the matched pairs but reports 2 extra clusters.
    rec = [W1, W2, W1 + 0.01, np.array([0.0, 0.0, 1.0])]
    m = matched_weight_error([W1, W2], [10, 10], rec)
    assert m["l2_matched"] < 1e-12
    assert m["n_extra"] == 2 and m["n_unmatched_true"] == 0


def test_under_clustering_is_penalized_by_size():
    m = matched_weight_error([W1, W2], [30, 10], [W1])
    assert m["n_unmatched_true"] == 1
    assert np.isclose(m["l2_penalized"], 10 / 40 * UNMATCHED_L2)


def test_matching_is_one_to_one():
    # Both true types are nearest to the same recovered vector; only one may use it.
    m = matched_weight_error([W1, 0.9 * W1 + 0.1 * W2], [5, 5], [W1, W2])
    assert sorted(c for _, c in m["match"]) == [0, 1]


def test_evaluate_end_to_end():
    s = PosteriorSummary(
        k_values=np.array([2, 3]), k_probs=np.array([0.75, 0.25]), psm=np.eye(4),
        point_clustering=np.array([0, 0, 1, 1]), point_weights=[W2, W1],
        map_assignments=np.array([0, 0, 1, 2]), map_weights=[W2, W1, W1],
        map_log_post=-3.0, rhat={"K": 1.01, "log_post": 1.02},
        ess={"K": 400.0, "log_post": 380.0}, accept_rate=0.25, n_chains=4, n_kept=100)
    out = evaluate(s, [1, 1, 0, 0], [W1, W2])
    assert out["ari_point"] == 1.0 and out["ari_map"] < 1.0
    assert out["p_K_true"] == 0.75 and out["K_mode"] == 2 and np.isclose(out["K_mean"], 2.25)
    assert out["l2_matched"] < 1e-12 and out["n_extra"] == 0
    assert out["rhat_K"] == 1.01 and out["ess_log_post"] == 380.0
