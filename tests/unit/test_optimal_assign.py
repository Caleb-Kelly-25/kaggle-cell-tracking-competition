"""Tests for globally optimal assignment and the motion-corrected distance.

These exist because the geometry diagnostic showed the learned linker is beaten
by plain distance on the full fold, so the remaining leverage is in the
*decision rule* and the *cost*, not the model. Both changes here are therefore
load-bearing and must be pinned.
"""

import numpy as np
import pytest

linking = pytest.importorskip("tracking_cellmot.linking")
pytest.importorskip("scipy")

optimal_assign = linking.optimal_assign
greedy_select = linking.greedy_select


def test_hungarian_beats_greedy_on_the_case_greedy_gets_wrong() -> None:
    """The concrete reason to prefer optimal assignment.

    Source 0 slightly prefers target 0, but source 1 can *only* use target 0.
    Greedy takes (0,0) first and strands source 1; the optimal matching gives
    target 0 to source 1 and target 1 to source 0, for a higher total.
    """
    probs = np.array([
        [0.90, 0.85],   # source 0 is nearly indifferent
        [0.80, 0.01],   # source 1 needs target 0
    ])

    g = greedy_select(probs, 0.5, max_children=1, max_parents=1)
    h = optimal_assign(probs, 0.5)

    # Greedy takes (0,0)=0.90 first; that consumes source 0's only child slot
    # AND target 0's only parent slot, so source 1 is stranded with nothing.
    assert sorted((i, j) for i, j, _ in g) == [(0, 0)], g
    # Optimal gives up 0.05 on source 0 to buy source 1 a link worth 0.80.
    assert sorted((i, j) for i, j, _ in h) == [(0, 1), (1, 0)], h

    # The payoff is edge COUNT, not summed likelihood: the metric is edge
    # Jaccard, so two correct links beat one. (Summed log-probability is not
    # comparable across different cardinalities -- every extra edge makes the
    # sum more negative, so it would always favour linking less.)
    assert len(h) == 2 and len(g) == 1, (h, g)


def test_assignment_is_one_to_one() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random((9, 7))
    picked = optimal_assign(probs, 0.3)
    srcs = [i for i, _, _ in picked]
    tgts = [j for _, j, _ in picked]
    assert len(srcs) == len(set(srcs)), "a source got two children"
    assert len(tgts) == len(set(tgts)), "a target got two parents"


def test_below_threshold_pairs_are_never_emitted() -> None:
    """A node must be left unlinked rather than forced into a bad match."""
    probs = np.array([[0.9, 0.01], [0.01, 0.02]])
    picked = optimal_assign(probs, 0.5)
    assert picked == [(0, 0, 0.9)], picked


def test_all_forbidden_returns_nothing_rather_than_failing() -> None:
    probs = np.array([[0.1, 0.2], [0.05, 0.15]])
    assert optimal_assign(probs, 0.5) == []


def test_empty_input_is_handled() -> None:
    assert optimal_assign(np.empty((0, 0)), 0.5) == []
    assert optimal_assign(np.empty((0, 4)), 0.5) == []


@pytest.mark.parametrize("shape", [(3, 9), (9, 3), (6, 6)])
def test_rectangular_inputs_match_the_smaller_side_at_most(shape) -> None:
    rng = np.random.default_rng(2)
    probs = rng.random(shape) * 0.5 + 0.5   # all above threshold
    picked = optimal_assign(probs, 0.4)
    assert len(picked) == min(shape)


def test_zero_probability_does_not_produce_a_nan_or_inf_cost() -> None:
    """log(0) would be -inf and make the solver fail rather than skip the pair."""
    probs = np.array([[0.0, 0.9], [0.8, 0.0]])
    picked = optimal_assign(probs, 0.5)
    assert sorted((i, j) for i, j, _ in picked) == [(0, 1), (1, 0)]


# ---------------------------------------------------------------------------
# Motion-corrected distance
# ---------------------------------------------------------------------------

def _motion_distance(cs, ct, vel, alpha, voxel):
    """The expression used in predict_video, isolated for testing."""
    pred = cs + alpha * vel
    return np.linalg.norm((pred[:, None, :] - ct[None, :, :]) * voxel, axis=-1)


VOXEL = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)


def test_alpha_zero_is_exactly_the_static_distance() -> None:
    """The default must not perturb the measured 0.7152 geometry result."""
    rng = np.random.default_rng(3)
    cs = rng.random((5, 3)).astype(np.float32) * 50
    ct = rng.random((4, 3)).astype(np.float32) * 50
    vel = rng.random((5, 3)).astype(np.float32) * 10
    static = np.linalg.norm((cs[:, None, :] - ct[None, :, :]) * VOXEL, axis=-1)
    assert np.array_equal(_motion_distance(cs, ct, vel, 0.0, VOXEL), static)


def test_motion_prefers_the_cell_that_continued_its_trajectory() -> None:
    """A cell moving steadily in +x should match the node ahead of it.

    Static distance is ambiguous here by construction: both candidates sit the
    same distance away, one ahead and one behind. Only velocity breaks the tie,
    which is exactly the information the static linker throws away.
    """
    cs = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)     # at x=10
    vel = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)     # moving +5 in x
    ct = np.array([
        [0.0, 0.0, 15.0],   # ahead  -- the continuation
        [0.0, 0.0, 5.0],    # behind -- equally distant statically
    ], dtype=np.float32)

    static = _motion_distance(cs, ct, vel, 0.0, VOXEL)[0]
    assert np.isclose(static[0], static[1]), "test setup must be a genuine tie"

    moved = _motion_distance(cs, ct, vel, 1.0, VOXEL)[0]
    assert moved[0] < moved[1], moved
    assert np.isclose(moved[0], 0.0, atol=1e-5), "full extrapolation should land on it"


def test_a_parentless_node_falls_back_to_static_distance() -> None:
    """Newly appeared tracks have no velocity; they must not be penalised."""
    cs = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
    ct = np.array([[0.0, 0.0, 12.0]], dtype=np.float32)
    zero_vel = np.zeros((1, 3), dtype=np.float32)
    assert np.array_equal(
        _motion_distance(cs, ct, zero_vel, 1.0, VOXEL),
        _motion_distance(cs, ct, zero_vel, 0.0, VOXEL),
    )
