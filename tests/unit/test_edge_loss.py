"""Unit tests for the edge/linking loss.

These run on CPU with real torch and no data, so they can be executed locally.
The central guarantee is that the DEFAULT code path stays bit-identical to the
loss that produced the measured baseline (score 0.6016) — otherwise every A/B
against that baseline is invalid.
"""

import pytest
import torch
import torch.nn.functional as F

train = pytest.importorskip("train_unet_transformer")


def _legacy_compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Frozen copy of compute_loss as it stood when the 0.6016 baseline was measured.

    Deliberately duplicated rather than imported: it is the reference the live
    implementation is checked against, so it must not move when the real one does.
    """
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    probs = torch.softmax(logits, dim=0)
    bce = F.binary_cross_entropy(probs, target, reduction="none")
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = ((1 - p_t) ** 2) * bce

    div_rows = target.sum(dim=1) > 1
    weight = torch.ones_like(loss)
    weight[div_rows] = 1.0

    return (loss * weight)[mask].mean()


def _random_pair(gen: torch.Generator, n_src: int, n_tgt: int, n_pos: int):
    """A random (logits, target) pair with sparse positives, like real data."""
    logits = torch.randn(n_src, n_tgt, generator=gen)
    target = torch.zeros(n_src, n_tgt)
    for _ in range(n_pos):
        i = int(torch.randint(n_src, (1,), generator=gen))
        j = int(torch.randint(n_tgt, (1,), generator=gen))
        target[i, j] = 1.0
    return logits, target


@pytest.mark.parametrize("seed", range(20))
def test_default_loss_is_bit_identical_to_baseline(seed: int) -> None:
    """The default path must reproduce the baseline loss EXACTLY, not approximately."""
    gen = torch.Generator().manual_seed(seed)
    n_src = int(torch.randint(2, 30, (1,), generator=gen))
    n_tgt = int(torch.randint(2, 30, (1,), generator=gen))
    n_pos = int(torch.randint(0, 5, (1,), generator=gen))
    logits, target = _random_pair(gen, n_src, n_tgt, n_pos)

    expected = _legacy_compute_loss(logits, target)
    actual = train.compute_loss(logits, target)
    assert torch.equal(actual, expected), (
        f"default compute_loss drifted from the baseline: {actual} != {expected}"
    )


def test_empty_mask_returns_zero() -> None:
    """An all-zero target masks everything out; must not NaN."""
    logits = torch.randn(4, 5)
    target = torch.zeros(4, 5)
    loss = train.compute_loss(logits, target)
    assert torch.isfinite(loss) and float(loss) == 0.0


@pytest.mark.parametrize("mode", ["colsoftmax", "parental", "dual", "sigmoid"])
def test_every_mode_is_finite_and_differentiable(mode: str) -> None:
    gen = torch.Generator().manual_seed(7)
    logits, target = _random_pair(gen, 8, 6, 3)
    logits.requires_grad_(True)
    loss = train.compute_loss(logits, target, mode=mode)
    assert torch.isfinite(loss), (mode, loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all(), mode


def test_unknown_mode_raises() -> None:
    # Needs a non-empty target: an all-zero target masks everything out and
    # returns early, before the mode is ever inspected.
    target = torch.zeros(3, 3)
    target[0, 1] = 1.0
    with pytest.raises(ValueError, match="Unknown edge loss mode"):
        train.compute_loss(torch.randn(3, 3), target, mode="nope")


def test_gate_removes_far_pairs_from_supervision() -> None:
    """A gated-out entry must not influence the loss at all."""
    gen = torch.Generator().manual_seed(11)
    logits, target = _random_pair(gen, 6, 6, 2)
    gate = torch.ones(6, 6, dtype=torch.bool)
    gate[0, :] = False  # gate out an entire source row

    base = train.compute_loss(logits, target, mode="dual", gate=gate)
    bumped = logits.clone()
    bumped[0, :] += 25.0  # change only gated entries
    after = train.compute_loss(bumped, target, mode="dual", gate=gate)
    assert torch.allclose(base, after, atol=1e-6), (base, after)


def test_gate_with_a_true_edge_outside_it_stays_finite() -> None:
    """A positive beyond the gate must not produce inf/NaN (it is simply unsupervised)."""
    logits = torch.randn(4, 4, generator=torch.Generator().manual_seed(5))
    target = torch.zeros(4, 4)
    target[0, 0] = 1.0
    gate = torch.ones(4, 4, dtype=torch.bool)
    gate[0, 0] = False
    loss = train.compute_loss(logits, target, mode="dual", gate=gate)
    assert torch.isfinite(loss)


def test_sigmoid_scores_are_invariant_to_extra_candidate_sources() -> None:
    """Softmax edge probabilities rescale with n_src; sigmoid ones do not.

    Detection runs at a very different operating point in training (raw logit
    > 0.3) than at inference (sigmoid > 0.99), so the number of candidate parents
    differs wildly.  Because the column softmax normalises over exactly that set,
    every edge probability is rescaled by the detector's threshold -- while
    sigmoid scores are unaffected.  That is the argument for the `sigmoid` arm.

    Asserted on the activation, not the loss: the loss is a mean over the
    supervised mask, which necessarily changes when the mask grows.
    """
    from tracking_cellmot import linking

    raw = torch.randn(5, 4, generator=torch.Generator().manual_seed(13))
    raw_big = torch.cat([raw, torch.full((6, 4), -10.0)], dim=0)

    sig = linking.edge_probs(raw, "sigmoid")
    sig_big = linking.edge_probs(raw_big, "sigmoid")[:5]
    assert torch.equal(sig, sig_big), "sigmoid must be per-edge and context-free"

    col = linking.edge_probs(raw, "softmax")
    col_big = linking.edge_probs(raw_big, "softmax")[:5]
    assert not torch.allclose(col, col_big, atol=1e-6), (
        "column softmax should be rescaled by the extra candidates"
    )


def test_division_row_weight_is_a_no_op() -> None:
    """`weight[div_rows] = 1.0` on an ones_like tensor is dead code.

    Multiplying by exactly 1.0 is exact in IEEE754, so removing it must be
    bit-identical even when a division row is present.
    """
    logits = torch.randn(5, 5, generator=torch.Generator().manual_seed(0))
    target = torch.zeros(5, 5)
    target[1, 2] = 1.0
    target[1, 3] = 1.0  # a dividing row: two children
    assert (target.sum(dim=1) > 1).any(), "test setup must contain a division"
    assert torch.equal(train.compute_loss(logits, target),
                       _legacy_compute_loss(logits, target))


# ---------------------------------------------------------------------------
# Held-out evaluation must use the arm's own objective
# ---------------------------------------------------------------------------

def _legacy_evaluate_pair(logits, target):
    """Frozen copy of the pre-fix body, for bit-identity of the defaults."""
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    if not active_rows.any():
        return 0.0, 0, 0
    loss = train.compute_loss(logits, target).item()
    probs = torch.softmax(logits, dim=0)
    preds = (probs > 0.5).float()
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    return loss, (preds[mask] == target[mask]).sum().item(), mask.sum().item()


@pytest.mark.parametrize("seed", range(10))
def test_evaluate_pair_defaults_are_bit_identical_to_legacy(seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(9, 7, generator=g) * 3
    target = (torch.rand(9, 7, generator=g) > 0.85).float()
    got = train._evaluate_pair(logits, target)
    want = _legacy_evaluate_pair(logits, target)
    assert got[1:] == want[1:]
    assert got[0] == want[0], (got[0], want[0])


def test_evaluate_pair_actually_changes_with_the_arm_objective() -> None:
    """The bug: the held-out loss ignored `mode`, so every arm got the same number.

    A null-slot arm scored under the legacy column softmax is being judged by a
    likelihood it never optimised, which makes --model-select loss incomparable
    across arms. These must differ.
    """
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(12, 10, generator=g) * 3
    target = (torch.rand(12, 10, generator=g) > 0.85).float()
    losses = {
        m: train._evaluate_pair(logits, target, mode=m)[0]
        for m in ("colsoftmax", "parental", "dual", "sigmoid")
    }
    assert len(set(losses.values())) == len(losses), losses
    assert all(x == x and abs(x) != float("inf") for x in losses.values()), losses


def test_evaluate_forwards_the_objective_to_evaluate_pair() -> None:
    """A parameter that stops at `evaluate()` would silently restore the bug."""
    import inspect
    sig = inspect.signature(train.evaluate).parameters
    for name in ("edge_loss_mode", "edge_activation", "edge_sigmoid_weight",
                 "edge_focal_gamma", "edge_null_logit", "train_gate_um"):
        assert name in sig, f"evaluate() does not accept {name}"
    body = inspect.getsource(train.evaluate)
    for name in ("mode=edge_loss_mode", "activation=edge_activation",
                 "null_logit=edge_null_logit"):
        assert name in body, f"evaluate() never forwards {name} to _evaluate_pair"


def test_train_scores_the_heldout_set_under_its_own_activation() -> None:
    """`train()` must pass the arm's inference activation, not the default."""
    import inspect
    body = inspect.getsource(train.train)
    assert "edge_activation=_INFERENCE_ACTIVATION[edge_loss_mode]" in body, (
        "train() calls evaluate() without mapping its loss mode to an activation"
    )
