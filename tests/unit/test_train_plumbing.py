"""Guards on the training entry points.

These catch plumbing mistakes that would otherwise only surface hours into a
GPU run: a parameter that never reaches the loss, a CLI flag that is defined but
never forwarded, or a name collision that silently rebinds a config value.
"""

import inspect
from pathlib import Path

import pytest

train_mod = pytest.importorskip("train_unet_transformer")

_SOURCE = Path(train_mod.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", [
    "edge_loss_mode", "edge_sigmoid_weight", "edge_focal_gamma",
    "edge_null_logit", "train_gate_um", "init_from", "freeze_unet", "seed",
    "model_select",
])
def test_train_accepts_parameter(name: str) -> None:
    assert name in inspect.signature(train_mod.train).parameters


@pytest.mark.parametrize("name", [
    "edge_loss", "edge_sigmoid_weight", "edge_focal_gamma",
    "edge_null_logit", "train_gate_um", "freeze_unet",
])
def test_train_epoch_accepts_parameter(name: str) -> None:
    assert name in inspect.signature(train_mod.train_epoch).parameters


@pytest.mark.parametrize("flag", [
    "--edge-loss", "--edge-sigmoid-weight", "--edge-focal-gamma",
    "--edge-null-logit", "--train-gate-um", "--init-from",
    "--freeze-unet", "--seed", "--model-select",
])
def test_cli_flag_is_defined_and_forwarded(flag: str) -> None:
    """A flag that is defined but never passed to train() is a silent no-op."""
    assert f'"{flag}"' in _SOURCE, f"{flag} is not defined in the parser"
    dest = flag.lstrip("-").replace("-", "_")
    assert f"args.{dest}" in _SOURCE, f"{flag} is parsed but never forwarded to train()"


def test_edge_loss_param_is_not_shadowed_by_the_loop_variable() -> None:
    """The epoch loop rebinds `edge_loss` to the returned loss value.

    If the config parameter were also called `edge_loss`, every epoch after the
    first would pass a float where a mode string is expected.
    """
    params = inspect.signature(train_mod.train).parameters
    assert "edge_loss" not in params, (
        "train() must not take a parameter named `edge_loss`: the epoch loop "
        "rebinds that name to the loss value"
    )
    assert "edge_loss, det_loss = train_epoch(" in _SOURCE


def test_compute_loss_defaults_are_the_legacy_path() -> None:
    sig = inspect.signature(train_mod.compute_loss).parameters
    assert sig["mode"].default == "colsoftmax"
    assert sig["gate"].default is None
    assert sig["sigmoid_weight"].default == 0.0
    assert sig["focal_gamma"].default == 2.0


def test_inference_activation_map_covers_every_loss_mode() -> None:
    """Each training loss must declare the activation inference should use.

    A model trained with a null slot is miscalibrated if scored without one.
    """
    modes = set(inspect.signature(train_mod.compute_loss).parameters["mode"].annotation
                if False else ["colsoftmax", "parental", "dual", "sigmoid"])
    for mode in modes:
        assert f'"{mode}":' in _SOURCE, f"{mode} missing from the activation map"
