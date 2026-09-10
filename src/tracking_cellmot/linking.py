"""Shared primitives for turning pair logits into edges.

Training and inference must agree on how a raw ``(n_src, n_tgt)`` logit matrix
becomes edge probabilities.  They previously each had their own copy, which is
how the two drifted: training optimised a column-softmax likelihood while
inference applied a different activation.  Everything here is torch/numpy only
(no polars/zarr/tracksdata), so it imports and unit-tests in a bare environment.

Background — why the null slot matters
--------------------------------------
The legacy activation is ``softmax(logits, dim=0)``: it normalises down each
*column*, so every detection at t+1 spreads exactly one unit of probability
across candidate parents at t.  With no "no-parent" option, a child whose real
parent was never detected still hands its full unit of mass to some neighbour —
which shows up as a spurious division.  Adding a null logit to the denominator
(Trackastra's "parental softmax", ECCV 2024) lets a detection have no parent::

    Phi(a)_ij = exp(a_ij) / (exp(null) + sum_i' exp(a_i'j))

The column form regulates *in-degree*.  Our measured failure mode is *out-degree*
(236 predicted forks against 7 real divisions), so the ``dual`` variants also
apply the row-wise term, penalising a parent that splits its mass over several
children.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Finite fill for gated-out pairs.  Deliberately not -inf: a fully-masked column
# would then produce NaN under softmax, and downstream sigmoids stay well-defined.
GATE_FILL = -1e4


def null_log_softmax(
    logits: torch.Tensor,
    dim: int,
    null_logit: float = 0.0,
) -> torch.Tensor:
    """``log( exp(a) / (exp(null_logit) + sum(exp(a), dim)) )``.

    A softmax with an extra, non-competing "null" entry in the denominator, so
    the probabilities along *dim* sum to less than one — the remainder is the
    mass assigned to "no match".  As ``null_logit -> -inf`` this recovers an
    ordinary ``log_softmax``.
    """
    lse = torch.logsumexp(logits, dim=dim, keepdim=True)
    null = torch.full_like(lse, float(null_logit))
    return logits - torch.logaddexp(null, lse)


def focal_bce_from_logp(
    log_p: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Element-wise focal BCE given ``log p(edge)``.

    Positives are ~0.05% of the supervised entries, so the focal modulation is
    doing real work suppressing easy negatives; ``gamma=0`` gives plain BCE.
    """
    p = log_p.exp()
    log1m = torch.log1p(-p.clamp(max=1.0 - 1e-7))
    bce = -(target * log_p + (1.0 - target) * log1m)
    if gamma == 0.0:
        return bce
    p_t = p * target + (1.0 - p) * (1.0 - target)
    return ((1.0 - p_t) ** gamma) * bce


def focal_bce_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Element-wise focal BCE on independent per-edge sigmoids.

    Unlike the softmax forms this is invariant to the number of candidate
    sources, which matters because the detector's operating point differs
    between training and inference.
    """
    log_p = F.logsigmoid(logits)
    log1m = F.logsigmoid(-logits)
    bce = -(target * log_p + (1.0 - target) * log1m)
    if gamma == 0.0:
        return bce
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    return ((1.0 - p_t) ** gamma) * bce


def edge_probs(
    raw: torch.Tensor,
    activation: str = "softmax",
    null_logit: float = 0.0,
) -> torch.Tensor:
    """Convert raw pair logits ``(n_src, n_tgt)`` into edge probabilities.

    ``softmax``            legacy: column-normalised, forces every child to take a parent
    ``sigmoid``            independent per-edge probability
    ``softmax_null``       column-normalised with a null-parent slot
    ``dual_softmax``       ``sqrt(col * row)`` — also regulates out-degree
    ``dual_softmax_null``  both, with null slots on each axis
    """
    if activation == "sigmoid":
        return torch.sigmoid(raw)

    use_null = activation.endswith("_null")
    col = (null_log_softmax(raw, 0, null_logit).exp() if use_null
           else torch.softmax(raw, dim=0))

    if activation in ("softmax", "softmax_null"):
        return col
    if activation in ("dual_softmax", "dual_softmax_null"):
        row = (null_log_softmax(raw, 1, null_logit).exp() if use_null
               else torch.softmax(raw, dim=1))
        return torch.sqrt(col * row)
    raise ValueError(f"Unknown edge_activation: {activation!r}")


# Number of relative-geometry channels fed to the pair MLP, per mode.
REL_CHANNELS = {"legacy": 3, "um": 5}


def rel_pair_features(
    coords_a: torch.Tensor,
    coords_b: torch.Tensor,
    mode: str = "legacy",
    voxel_size: tuple[float, ...] | torch.Tensor | None = None,
    scale_um: float = 4.0,
) -> torch.Tensor:
    """Relative-geometry features for every ``(a, b)`` pair.

    ``legacy`` — ``(a - b) / 100`` in raw voxel units.  Two defects, both measured
    on the trained checkpoint: the fixed ``/100`` puts a typical 1.8 um
    displacement at ~0.01-0.045 alongside LayerNorm'd appearance features of
    magnitude ~1 (the network compensated by growing these weights ~3x), and the
    units are anisotropic, so the same physical distance in z produces a ~4x
    smaller input than in x/y — leaving the z geometry signal ~2.7x weaker.

    ``um`` — ``[d_um/s, |d|/s, exp(-|d|/s)]`` (5 channels, ``s = scale_um``).
    Isotropic, O(1) across the realistic displacement range, and adds an explicit
    distance magnitude plus a smooth proximity kernel (a soft version of the
    distance gate).  At ``s = 4.0``: 1.8 um -> 0.46/0.63, 3.6 um -> 0.91/0.40,
    12 um -> 3.0/0.05.
    """
    d = coords_a.unsqueeze(-2) - coords_b.unsqueeze(-3)
    if mode == "legacy":
        return d / 100.0
    if mode != "um":
        raise ValueError(f"Unknown rel_mode: {mode!r}")
    if voxel_size is None:
        raise ValueError("rel_mode='um' requires voxel_size (microns per voxel)")
    if not isinstance(voxel_size, torch.Tensor):
        voxel_size = torch.tensor(voxel_size, dtype=d.dtype, device=d.device)
    d_um = d * voxel_size
    r = d_um.norm(dim=-1, keepdim=True)
    return torch.cat([d_um / scale_um, r / scale_um, torch.exp(-r / scale_um)], dim=-1)


def pair_distance_um(
    coords_a: torch.Tensor,
    coords_b: torch.Tensor,
    voxel_size: tuple[float, ...] | torch.Tensor,
) -> torch.Tensor:
    """Pairwise physical distance in microns between two sets of ``(..., 3)`` coords.

    Coordinates are voxel indices; *voxel_size* converts each axis to microns,
    which matters because the data is 4x anisotropic before downsampling.
    """
    if not isinstance(voxel_size, torch.Tensor):
        voxel_size = torch.tensor(voxel_size, dtype=coords_a.dtype, device=coords_a.device)
    d = (coords_a.unsqueeze(-2) - coords_b.unsqueeze(-3)) * voxel_size
    return d.norm(dim=-1)


def distance_gate(d_um: torch.Tensor, gate_um: float) -> torch.Tensor:
    """Boolean mask of pairs within *gate_um* microns (all True when not finite)."""
    if gate_um is None or not float(gate_um) == float(gate_um) or gate_um == float("inf"):
        return torch.ones_like(d_um, dtype=torch.bool)
    return d_um <= gate_um
