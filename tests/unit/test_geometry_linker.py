"""Guards on the `--linker geometry` diagnostic ablation.

The diagnostic only answers its question -- "how much does the transformer add
over nearest-neighbour?" -- if it changes *exactly one* thing. If the geometry
branch also skipped the gate, or used a different threshold or degree cap, the
gap against a learned run would confound the linker with those changes and the
conclusion (invest in motion modelling vs. a bigger model) would be unfounded.
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

predict_mod = pytest.importorskip("predict_unet_transformer")
linking = pytest.importorskip("tracking_cellmot.linking")

_SOURCE = Path(predict_mod.__file__).read_text(encoding="utf-8")


def test_config_exposes_the_linker_knobs() -> None:
    cfg = predict_mod.PredictConfig()
    assert cfg.linker == "learned", "the ablation must be opt-in"
    assert cfg.geometry_temp_um > 0


@pytest.mark.parametrize("flag,dest", [
    ("--linker", "linker"), ("--geometry-temp-um", "geometry_temp_um"),
])
def test_flag_is_defined_and_forwarded(flag: str, dest: str) -> None:
    assert f'"{flag}"' in _SOURCE, f"{flag} is not defined in the parser"
    assert f"args.{dest}" in _SOURCE, f"{flag} is parsed but never reaches PredictConfig"


def test_geometry_run_is_distinguishable_in_the_results_csv() -> None:
    """Two rows that differ only in the linker must not look identical."""
    assert '"linker": cfg.linker' in _SOURCE


def test_geometry_scores_are_monotonically_decreasing_in_distance() -> None:
    """The whole premise: nearer pairs must score higher, at any temperature."""
    d_um = np.array([[0.0, 1.8, 3.6, 12.0]], dtype=np.float32)
    for temp in (0.5, 2.0, 10.0):
        raw = torch.from_numpy((-d_um / temp).astype(np.float32))
        probs = linking.edge_probs(raw, "sigmoid")[0].numpy()
        assert np.all(np.diff(probs) < 0), (temp, probs)


def test_geometry_temperature_controls_how_decisively_the_nearest_wins() -> None:
    """T is the microns-scale over which a competing parent stops mattering.

    Under the column softmax the scores become a Boltzmann distribution, so a
    small T makes the nearest parent win outright and a large T spreads mass --
    which is what makes the existing `--threshold` still meaningful here.
    """
    # One child (column), two candidate parents at 1 um and 5 um.
    d_um = np.array([[1.0], [5.0]], dtype=np.float32)
    sharp = linking.edge_probs(torch.from_numpy(-d_um / 0.5), "softmax")[0, 0]
    soft = linking.edge_probs(torch.from_numpy(-d_um / 20.0), "softmax")[0, 0]
    assert float(sharp) > 0.99, sharp
    assert float(soft) < 0.65, soft


def test_geometry_branch_changes_only_the_score_not_the_decision_rule() -> None:
    """Static guard: the ablation must not fork any downstream step.

    Everything after `raw` is assigned -- gating, activation, thresholding,
    degree caps -- has to be shared between the two branches, or the comparison
    measures more than the linker.
    """
    tree = ast.parse(_SOURCE)
    ifs = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and "geometry" in ast.unparse(n.test)
        and "linker" in ast.unparse(n.test)
    ]
    assert len(ifs) == 1, f"expected exactly one linker branch, found {len(ifs)}"

    branch_src = ast.unparse(ifs[0])
    for downstream in ("gate_um", "edge_probs", "cfg.threshold",
                       "max_children_per_node", "max_parents_per_node"):
        assert downstream not in branch_src, (
            f"the linker branch touches `{downstream}` -- the geometry ablation "
            f"must swap the score only, leaving every downstream decision shared"
        )


def test_geometry_branch_skips_the_transformer() -> None:
    """A bypass that still ran the model would make the diagnostic pointlessly slow."""
    tree = ast.parse(_SOURCE)
    branch = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and "geometry" in ast.unparse(n.test) and "linker" in ast.unparse(n.test)
    )
    assert "predict_edges" not in ast.unparse(branch.body), (
        "geometry mode still calls predict_edges"
    )
    assert "predict_edges" in ast.unparse(branch.orelse), (
        "the learned path must be the else branch"
    )


def test_motion_alpha_is_refused_where_it_would_be_a_no_op() -> None:
    """A flag that silently does nothing cost us a whole 19-video GPU arm.

    `pack_det096_mt6_a05` scored identically to four decimals -- same
    edge_jaccard too -- with motion "enabled", because the motion-corrected
    distance is only consumed inside the `linker == "geometry"` branch. The
    learned linker's logits come from the model, which never sees it. So the
    combination must raise rather than appear to work.
    """
    with pytest.raises(ValueError, match="has no effect"):
        predict_mod.PredictConfig(motion_alpha=0.5)          # linker defaults to "learned"
    with pytest.raises(ValueError, match="motion-alpha"):
        predict_mod.PredictConfig(motion_alpha=1.0, linker="learned")


def test_motion_alpha_is_allowed_with_the_geometry_linker() -> None:
    cfg = predict_mod.PredictConfig(motion_alpha=0.5, linker="geometry")
    assert cfg.motion_alpha == 0.5 and cfg.linker == "geometry"


def test_default_config_is_unaffected_by_the_guard() -> None:
    """alpha=0 must stay constructible -- it is every measured run to date."""
    cfg = predict_mod.PredictConfig()
    assert cfg.motion_alpha == 0.0 and cfg.linker == "learned"
    # And the winning submission config (ILP, no motion) must still build.
    assert predict_mod.PredictConfig(use_ilp=True, min_track_nodes=6).motion_alpha == 0.0
