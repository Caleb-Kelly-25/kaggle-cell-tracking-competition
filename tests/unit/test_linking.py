"""Unit tests for the shared linking primitives.

The key contract: the normalisation the training loss optimises must be exactly
the normalisation inference applies. These tests pin that, plus the null-slot
behaviour that motivated the whole change.
"""

import math

import pytest
import torch

linking = pytest.importorskip("tracking_cellmot.linking")


def _raw(seed: int = 0, n_src: int = 6, n_tgt: int = 5) -> torch.Tensor:
    return torch.randn(n_src, n_tgt, generator=torch.Generator().manual_seed(seed))


def test_softmax_activation_is_exactly_legacy() -> None:
    """The default activation must stay bit-identical to plain column softmax."""
    raw = _raw()
    assert torch.equal(linking.edge_probs(raw, "softmax"), torch.softmax(raw, dim=0))


def test_sigmoid_activation() -> None:
    raw = _raw()
    assert torch.equal(linking.edge_probs(raw, "sigmoid"), torch.sigmoid(raw))


def test_null_log_softmax_matches_previous_inference_formula() -> None:
    """`softmax(a,0) * sigmoid(logsumexp(a,0) - b)` == `exp(null_log_softmax(a,0,b))`.

    The old inference code used the left form; the loss now uses the right one.
    Both equal exp(a) / (exp(b) + sum exp(a)), and this test is what guarantees
    training and inference stay in lockstep.
    """
    raw = _raw(seed=3)
    for b in (-2.0, 0.0, 1.5):
        old = torch.softmax(raw, dim=0) * torch.sigmoid(
            torch.logsumexp(raw, dim=0, keepdim=True) - b)
        new = linking.null_log_softmax(raw, 0, b).exp()
        assert torch.allclose(old, new, atol=1e-6), b


def test_null_slot_leaves_mass_unassigned() -> None:
    """With a null slot each column sums to LESS than one: a child may have no parent."""
    raw = _raw(seed=1)
    p = linking.edge_probs(raw, "softmax_null", 0.0)
    col_sums = p.sum(dim=0)
    assert (col_sums < 1.0).all(), col_sums
    assert (col_sums > 0.0).all()


def test_null_slot_recovers_legacy_as_null_logit_goes_to_minus_inf() -> None:
    raw = _raw(seed=2)
    p = linking.edge_probs(raw, "softmax_null", -60.0)
    assert torch.allclose(p, torch.softmax(raw, dim=0), atol=1e-6)


def test_weak_evidence_column_stays_near_zero_only_with_null_slot() -> None:
    """The core pathology: with no null slot, a column of uniformly terrible
    candidates still renormalises to a confident parent."""
    raw = torch.full((3, 1), -20.0)
    legacy = linking.edge_probs(raw, "softmax")
    parental = linking.edge_probs(raw, "softmax_null", 0.0)
    assert torch.allclose(legacy, torch.full((3, 1), 1.0 / 3.0)), legacy
    assert float(parental.max()) < 1e-6, parental


def test_dual_penalises_a_second_child() -> None:
    """A parent that spreads mass over two children should be penalised by the row term."""
    raw = torch.tensor([[0.0, 3.0, 2.5], [0.0, 0.5, 0.4]])
    col = linking.edge_probs(raw, "softmax")
    dual = linking.edge_probs(raw, "dual_softmax")
    assert col[0, 1] > 0.5 and col[0, 2] > 0.5, "setup: source 0 wins both columns"
    assert dual[0, 2] < col[0, 2]


def test_pair_distance_um_accounts_for_anisotropy() -> None:
    """One voxel in z is 4x the physical distance of one voxel in x at native scale."""
    a = torch.tensor([[0.0, 0.0, 0.0]])
    z = torch.tensor([[1.0, 0.0, 0.0]])
    x = torch.tensor([[0.0, 0.0, 1.0]])
    vs = (1.625, 0.40625, 0.40625)
    assert math.isclose(float(linking.pair_distance_um(a, z, vs)), 1.625, rel_tol=1e-6)
    assert math.isclose(float(linking.pair_distance_um(a, x, vs)), 0.40625, rel_tol=1e-6)


def test_distance_gate_inf_is_a_no_op() -> None:
    d = torch.rand(4, 3) * 50
    assert linking.distance_gate(d, float("inf")).all()
    assert not linking.distance_gate(d, 0.0).any()
