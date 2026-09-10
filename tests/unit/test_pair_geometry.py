"""Tests for the pair-MLP relative-geometry features and checkpoint migration."""

import math
from pathlib import Path

import pytest
import torch

linking = pytest.importorskip("tracking_cellmot.linking")
snt = pytest.importorskip("tracking_cellmot.models.simple_node_transformer")

VOXEL = (1.625, 0.40625, 0.40625)  # microns per ORIGINAL voxel (z, y, x)


def test_legacy_features_are_bit_identical() -> None:
    a = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(0))
    b = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(1))
    expected = (a.unsqueeze(2) - b.unsqueeze(1)) / 100.0
    assert torch.equal(linking.rel_pair_features(a, b, "legacy"), expected)


def test_um_features_are_isotropic_but_legacy_ones_are_not() -> None:
    """The actual defect: the same PHYSICAL distance must give the same feature.

    Under `legacy` a 1 um move in z yields a 4x smaller input than in x, because
    the features are raw voxel counts and the voxels are 4x anisotropic.
    """
    origin = torch.zeros(1, 1, 3)
    one_um_z = torch.tensor([[[1.0 / VOXEL[0], 0.0, 0.0]]])
    one_um_x = torch.tensor([[[0.0, 0.0, 1.0 / VOXEL[2]]]])

    um_z = linking.rel_pair_features(origin, one_um_z, "um", VOXEL).abs().max()
    um_x = linking.rel_pair_features(origin, one_um_x, "um", VOXEL).abs().max()
    assert math.isclose(float(um_z), float(um_x), rel_tol=1e-5), (um_z, um_x)

    # z voxels are 4x larger, so 1 um of motion is only 0.62 voxels in z versus
    # 2.46 in x -- i.e. the legacy z input is ~4x SMALLER for the same physical
    # displacement, which is why the trained model's z geometry signal is weakest.
    leg_z = linking.rel_pair_features(origin, one_um_z, "legacy").abs().max()
    leg_x = linking.rel_pair_features(origin, one_um_x, "legacy").abs().max()
    assert float(leg_x) / float(leg_z) > 3.5, (
        f"legacy should be ~4x anisotropic (z={leg_z}, x={leg_x})"
    )


def test_um_features_are_order_one_at_realistic_displacements() -> None:
    """Legacy inputs are ~0.01-0.05 next to LayerNorm'd features of magnitude ~1."""
    origin = torch.zeros(1, 1, 3)
    # 1.82 um (the measured median) along x
    p = torch.tensor([[[0.0, 0.0, 1.82 / VOXEL[2]]]])
    um = linking.rel_pair_features(origin, p, "um", VOXEL, scale_um=4.0).abs().max()
    legacy = linking.rel_pair_features(origin, p, "legacy").abs().max()
    assert float(um) > 0.3, um
    assert float(legacy) < 0.06, legacy


def test_um_requires_voxel_size() -> None:
    with pytest.raises(ValueError, match="requires voxel_size"):
        linking.rel_pair_features(torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), "um")


@pytest.mark.parametrize("mode,expected", [("legacy", 3), ("um", 5)])
def test_pair_mlp_input_width_tracks_the_mode(mode: str, expected: int) -> None:
    m = snt.SimpleNodeTransformer(feat_dim=8, hidden_dim=16, rel_mode=mode)
    assert m.pair_mlp[0].in_features == 16 * 2 + expected


def test_transfer_migration_is_function_preserving() -> None:
    """A legacy checkpoint migrated into the um model must compute the SAME function.

    This is what lets us change the architecture without invalidating the
    measured baseline: the geometry term is re-expressed in microns and the two
    new channels start at zero, so the output is unchanged.
    """
    torch.manual_seed(0)
    kw = dict(feat_dim=8, hidden_dim=16, n_heads=2, n_blocks=2, dropout=0.0)
    legacy = snt.SimpleNodeTransformer(rel_mode="legacy", **kw).eval()
    new = snt.SimpleNodeTransformer(rel_mode="um", rel_scale_um=4.0, **kw).eval()

    migrated = snt.migrate_pair_mlp_state(
        legacy.state_dict(), new, legacy_voxel_size=VOXEL, rel_scale_um=4.0,
        init="transfer",
    )
    missing, unexpected = new.load_state_dict(migrated, strict=True), None
    assert unexpected is None

    g = torch.Generator().manual_seed(3)
    ft = torch.randn(1, 6, 8, generator=g)
    ft1 = torch.randn(1, 5, 8, generator=g)
    ct = torch.randn(1, 6, 3, generator=g) * 20
    ct1 = torch.randn(1, 5, 3, generator=g) * 20

    with torch.no_grad():
        a = legacy(ft, ft1, ct, ct1)
        b = new(ft, ft1, ct, ct1, voxel_size=VOXEL)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_migration_is_function_preserving_through_UNetNodeTransformer() -> None:
    """The same guarantee, but through the class the checkpoint is actually saved from.

    `test_transfer_migration_is_function_preserving` covers the inner module;
    this covers the real warm-start path -- `UNetNodeTransformer.predict_edges`
    with a strict load -- which is what Phase 1 (`--epochs 0` must reproduce
    0.6561) depends on. It also pins that `predict_edges` forwards `voxel_size`:
    without the forward the `um` model raises, and with the wrong scale the
    outputs would not match.
    """
    train_mod = pytest.importorskip("train_unet_transformer")
    from tracking_cellmot.models import TemporalUNet3D

    torch.manual_seed(0)
    C, POS = 4, 8

    def build(mode: str):
        torch.manual_seed(7)  # identical init for the shared (non-pair-MLP) weights
        return train_mod.UNetNodeTransformer(
            unet=TemporalUNet3D(in_channels=1, out_channels=C, layers=[4, 8]),
            unet_out_channels=C, pos_feat_dim=POS,
            hidden_dim=16, n_heads=2, n_blocks=2, dropout=0.0,
            rel_mode=mode, rel_scale_um=4.0,
        ).eval()

    legacy, new = build("legacy"), build("um")
    migrated = snt.migrate_pair_mlp_state(
        legacy.state_dict(), new, legacy_voxel_size=VOXEL, rel_scale_um=4.0,
        init="transfer",
    )
    missing, unexpected = new.load_state_dict(migrated, strict=True)
    assert not missing and not unexpected, (missing, unexpected)

    g = torch.Generator().manual_seed(11)
    args = (
        torch.randn(1, 6, C, generator=g), torch.randn(1, 5, C, generator=g),
        torch.randn(1, 6, 3, generator=g) * 30, torch.randn(1, 5, 3, generator=g) * 30,
        torch.randn(1, 6, POS, generator=g), torch.randn(1, 5, POS, generator=g),
        torch.ones(1, 6, dtype=torch.bool), torch.ones(1, 5, dtype=torch.bool),
    )
    with torch.no_grad():
        a = legacy.predict_edges(*args)
        b = new.predict_edges(*args, voxel_size=VOXEL)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


_CKPT = Path(__file__).resolve().parents[2] / "checkpoints" / "baseline_split0"


@pytest.mark.skipif(not (_CKPT / "edge_predictor_best.pth").exists(),
                    reason="vendored baseline checkpoint not present")
def test_the_real_checkpoint_migrates_without_changing_its_predictions() -> None:
    """End-to-end on the actual weights behind the measured 0.6561.

    Synthetic models can hide a shape or key mismatch that only the real
    checkpoint has. This is the local stand-in for Phase 1 (`--epochs 0` must
    reproduce the baseline): if the migrated model's logits match, the
    architecture change cannot have moved the score.
    """
    predict_mod = pytest.importorskip("predict_unet_transformer")
    train_mod = pytest.importorskip("train_unet_transformer")
    from tracking_cellmot.models import TemporalUNet3D

    weights = _CKPT / "edge_predictor_best.pth"
    dev = torch.device("cpu")
    legacy, _, _ = predict_mod.load_model(weights, dev)
    assert legacy.transformer.rel_mode == "legacy", "v1 config must load as legacy"

    pos = 4 * predict_mod._POS_EMBED_DIM
    new = train_mod.UNetNodeTransformer(
        unet=TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128]),
        unet_out_channels=32, pos_feat_dim=pos, rel_mode="um", rel_scale_um=4.0,
    )
    state = torch.load(weights, map_location=dev, weights_only=True)
    migrated = snt.migrate_pair_mlp_state(
        state, new, legacy_voxel_size=VOXEL, rel_scale_um=4.0, init="transfer",
    )
    missing, unexpected = new.load_state_dict(migrated, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    legacy.eval(), new.eval()

    g = torch.Generator().manual_seed(42)
    ns, nt = 24, 21
    extent = torch.tensor([60.0, 1200.0, 1200.0])  # a realistic original-res volume
    args = (
        torch.randn(1, ns, 32, generator=g), torch.randn(1, nt, 32, generator=g),
        torch.rand(1, ns, 3, generator=g) * extent,
        torch.rand(1, nt, 3, generator=g) * extent,
        torch.randn(1, ns, pos, generator=g), torch.randn(1, nt, pos, generator=g),
        torch.ones(1, ns, dtype=torch.bool), torch.ones(1, nt, dtype=torch.bool),
    )
    with torch.no_grad():
        a = legacy.predict_edges(*args)
        b = new.predict_edges(*args, voxel_size=VOXEL)
    # Measured 3.8e-6 against logits spanning about [-26, -1].
    assert torch.allclose(a, b, atol=1e-3), (a - b).abs().max()


def test_um_model_rejects_a_missing_voxel_size() -> None:
    """A silent fallback here would be a units bug, so it must raise instead."""
    m = snt.SimpleNodeTransformer(feat_dim=8, hidden_dim=16, n_heads=2, n_blocks=1,
                                  rel_mode="um").eval()
    with pytest.raises(ValueError, match="requires voxel_size"):
        m(torch.randn(1, 3, 8), torch.randn(1, 2, 8),
          torch.randn(1, 3, 3), torch.randn(1, 2, 3))


def test_zeros_init_drops_only_the_geometry_term() -> None:
    torch.manual_seed(1)
    kw = dict(feat_dim=8, hidden_dim=16, n_heads=2, n_blocks=2, dropout=0.0)
    legacy = snt.SimpleNodeTransformer(rel_mode="legacy", **kw).eval()
    new = snt.SimpleNodeTransformer(rel_mode="um", **kw).eval()
    migrated = snt.migrate_pair_mlp_state(legacy.state_dict(), new, init="zeros")
    key = "pair_mlp.0.weight"
    n_feat = 16 * 2
    assert torch.equal(migrated[key][:, :n_feat], legacy.state_dict()[key][:, :n_feat])
    assert float(migrated[key][:, n_feat:].abs().max()) == 0.0
