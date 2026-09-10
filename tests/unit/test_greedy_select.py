"""Tests for the shared greedy edge-selection rule and the selective-division gate.

Two things are pinned here:

1. Extracting the rule out of `predict_video` must not have changed any
   decision -- otherwise every measured score becomes incomparable.
2. `division_threshold` must trade divisions off *by confidence*. The plain cap
   cannot: measured, `max_children=1` gives TP=0/FP=0/FN=7 (no division credit
   at all) and `max_children=2` gives TP=2/FP=234 (J=0.008) while also costing
   edge Jaccard. Since the score weights divisions at 0.1*TP/(TP+FP+FN),
   precision is the only thing that pays.
"""

import numpy as np
import pytest

linking = pytest.importorskip("tracking_cellmot.linking")
greedy_select = linking.greedy_select


def _legacy_greedy(probs, threshold, max_children, max_parents):
    """Frozen copy of the inline loop as it stood before extraction."""
    sel = np.argwhere(probs > threshold)
    if sel.size:
        sel = sel[np.argsort(-probs[sel[:, 0], sel[:, 1]])]
    candidates = [(float(probs[i, j]), int(i), int(j)) for i, j in sel]
    children_count, parents_count, out = {}, {}, []
    for prob, i, j in candidates:
        n_ch = children_count.get(i, 0)
        n_pa = parents_count.get(j, 0)
        if max_children is not None and n_ch >= max_children:
            continue
        if max_parents is not None and n_pa >= max_parents:
            continue
        out.append((i, j, prob))
        children_count[i] = n_ch + 1
        parents_count[j] = n_pa + 1
    return out


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("max_children", [1, 2])
def test_extraction_is_behaviour_preserving(seed: int, max_children: int) -> None:
    """Defaults (division_threshold=None) must reproduce the pre-extraction rule."""
    rng = np.random.default_rng(seed)
    probs = rng.random((11, 9))
    got = greedy_select(probs, 0.5, max_children=max_children, max_parents=1)
    want = _legacy_greedy(probs, 0.5, max_children, 1)
    assert got == want


def test_threshold_excludes_low_probability_edges() -> None:
    probs = np.array([[0.9, 0.2], [0.1, 0.8]])
    picked = greedy_select(probs, 0.5, max_children=1, max_parents=1)
    assert sorted((i, j) for i, j, _ in picked) == [(0, 0), (1, 1)]


def test_highest_probability_wins_the_contested_slot() -> None:
    """Both sources want target 0; with max_parents=1 the stronger must win."""
    probs = np.array([[0.6, 0.0], [0.95, 0.0]])
    picked = greedy_select(probs, 0.5, max_children=1, max_parents=1)
    assert picked == [(1, 0, 0.95)]


def test_cap_of_one_forbids_every_division() -> None:
    """This is the current setting, and why division credit is exactly zero."""
    probs = np.array([[0.99, 0.98]])  # one parent, two strong children
    picked = greedy_select(probs, 0.5, max_children=1, max_parents=1)
    assert len(picked) == 1


def test_division_threshold_admits_only_a_confident_second_child() -> None:
    """The whole point: buy division recall a few edges at a time."""
    # Parent 0 has a strong second child (0.85); parent 1's is weak (0.55).
    probs = np.array([
        [0.99, 0.85, 0.00, 0.00],
        [0.00, 0.00, 0.99, 0.55],
    ])
    picked = greedy_select(probs, 0.5, max_children=1, max_parents=1,
                           division_threshold=0.8)
    by_parent: dict[int, int] = {}
    for i, _, _ in picked:
        by_parent[i] = by_parent.get(i, 0) + 1
    assert by_parent == {0: 2, 1: 1}, picked


def test_division_threshold_is_off_by_default() -> None:
    probs = np.array([[0.99, 0.98]])
    assert len(greedy_select(probs, 0.5, max_children=1, max_parents=1)) == 1


def test_division_never_exceeds_its_ceiling() -> None:
    """Even with everything above the bar, a fork stays a fork."""
    probs = np.array([[0.99, 0.98, 0.97, 0.96]])
    picked = greedy_select(probs, 0.5, max_children=1, max_parents=1,
                           division_threshold=0.5, division_max_children=2)
    assert len(picked) == 2


def test_a_high_division_threshold_degenerates_to_no_divisions() -> None:
    """Sanity: the knob spans the full range between the two measured regimes."""
    rng = np.random.default_rng(0)
    probs = rng.random((8, 12))
    strict = greedy_select(probs, 0.5, max_children=1, max_parents=1,
                           division_threshold=1.01)
    plain = greedy_select(probs, 0.5, max_children=1, max_parents=1)
    assert strict == plain


def test_a_zero_division_threshold_matches_a_plain_cap_of_two() -> None:
    """The other end: admitting every second child equals max_children=2."""
    rng = np.random.default_rng(1)
    probs = rng.random((8, 12))
    loose = greedy_select(probs, 0.5, max_children=1, max_parents=1,
                          division_threshold=0.0, division_max_children=2)
    cap2 = greedy_select(probs, 0.5, max_children=2, max_parents=1)
    assert loose == cap2


def test_parent_cap_still_binds_on_an_admitted_division() -> None:
    """A fork must never create a merge -- two parents for one child is invalid."""
    # Source 0 takes target 0. Source 1 would like target 0 as a 2nd child.
    probs = np.array([
        [0.99, 0.00],
        [0.95, 0.90],
    ])
    picked = greedy_select(probs, 0.5, max_children=1, max_parents=1,
                           division_threshold=0.5)
    targets = [j for _, j, _ in picked]
    assert len(targets) == len(set(targets)), f"a target got two parents: {picked}"


def test_dual_softmax_separates_a_division_from_an_ambiguity() -> None:
    """What makes `division_threshold` principled rather than merely swept.

    Under sqrt(p_col * p_row) a dividing parent splits its row mass evenly while
    each child keeps one clear parent, giving 1/sqrt(2); genuine 2x2 confusion
    gives 0.5. If these ever stop separating, the threshold has no defensible
    value and this knob should be reconsidered rather than re-tuned.
    """
    torch = pytest.importorskip("torch")
    import math

    def p(logits):
        return linking.edge_probs(torch.tensor(logits, dtype=torch.float32),
                                  "dual_softmax")

    division = float(p([[5.0, 5.0]])[0, 1])
    confusion = float(p([[5.0, 5.0], [5.0, 5.0]])[0, 1])
    single = float(p([[5.0, -5.0], [-5.0, 5.0]])[0, 0])

    assert math.isclose(division, 1 / math.sqrt(2), rel_tol=1e-3), division
    assert math.isclose(confusion, 0.5, rel_tol=1e-3), confusion
    assert math.isclose(single, 1.0, rel_tol=1e-3), single
    assert confusion < division < single

    # The documented band admits the division and rejects the confusion.
    for thr in (0.55, 0.60, 0.65, 0.70):
        assert confusion < thr < division, thr
