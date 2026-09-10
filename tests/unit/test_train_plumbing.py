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
    "edge_loss_mode", "edge_sigmoid_weight", "edge_focal_gamma",
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


@pytest.mark.parametrize("func", ["train", "train_epoch"])
def test_no_function_takes_a_parameter_it_rebinds(func: str) -> None:
    """`edge_loss` is assigned inside BOTH train() and train_epoch().

    train():       `edge_loss, det_loss = train_epoch(...)`
    train_epoch(): `edge_loss = sum(block_losses) / len(block_losses)`

    A parameter of that name is therefore clobbered with a loss tensor after the
    first iteration, and the next call receives a tensor where a mode string is
    expected. This actually happened: a GPU run died with
    "Unknown edge loss mode: tensor(0.0017, ...)". Both must use `edge_loss_mode`.
    """
    params = inspect.signature(getattr(train_mod, func)).parameters
    assert "edge_loss" not in params, (
        f"{func}() must not take a parameter named `edge_loss` -- it rebinds that "
        f"name to a loss tensor, silently corrupting the config after one iteration"
    )
    assert "edge_loss_mode" in params


def test_the_rebinding_that_motivates_the_rename_still_exists() -> None:
    """If these assignments ever disappear, the rename above can be reconsidered."""
    assert "edge_loss, det_loss = train_epoch(" in _SOURCE
    assert "edge_loss = sum(block_losses)" in _SOURCE


_CONFIG_PARAMS = {
    # `edge_loss` is listed deliberately: it is the name that caused the original
    # failure, and both functions still assign it, so reintroducing it as a
    # parameter anywhere must fail this test.
    "edge_loss",
    "edge_loss_mode", "edge_sigmoid_weight", "edge_focal_gamma", "edge_null_logit",
    "train_gate_um", "detection_loss", "pu_gate_power", "model_select",
    "freeze_unet", "init_from", "method",
}


def test_config_parameters_are_never_reassigned_in_their_own_body() -> None:
    """Generic guard for the bug class that killed a GPU run.

    A configuration parameter that its own function reassigns is silently
    corrupted partway through: the value is right on the first iteration and
    wrong on every one after. Signature checks cannot see this, so scan the AST.

    Deliberately limited to config-like names -- rebinding a *data* parameter
    (e.g. `logits = logits.masked_fill(...)`) is idiomatic and fine.
    """
    import ast

    offenders = []
    for fn in (n for n in ast.walk(ast.parse(_SOURCE)) if isinstance(n, ast.FunctionDef)):
        params = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)}
        assigned: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        assigned.add(tgt.id)
                    elif isinstance(tgt, ast.Tuple):
                        assigned.update(e.id for e in tgt.elts if isinstance(e, ast.Name))
        clobbered = params & assigned & _CONFIG_PARAMS
        if clobbered:
            offenders.append((fn.name, sorted(clobbered)))

    assert not offenders, (
        f"config parameters reassigned inside their own function: {offenders}"
    )


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
    for mode in ("colsoftmax", "parental", "dual", "sigmoid"):
        assert f'"{mode}":' in _SOURCE, f"{mode} missing from the activation map"


# ---------------------------------------------------------------------------
# Pair-geometry plumbing (Step 6)
# ---------------------------------------------------------------------------

predict_mod = pytest.importorskip("predict_unet_transformer")
_PREDICT_SOURCE = Path(predict_mod.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["pair_rel_mode", "pair_rel_scale_um", "init_pair_mlp"])
def test_train_accepts_pair_geometry_parameter(name: str) -> None:
    assert name in inspect.signature(train_mod.train).parameters


@pytest.mark.parametrize("flag", ["--pair-rel-mode", "--pair-rel-scale-um", "--init-pair-mlp"])
def test_pair_geometry_flag_is_defined_and_forwarded(flag: str) -> None:
    assert f'"{flag}"' in _SOURCE, f"{flag} is not defined in the parser"
    dest = flag.lstrip("-").replace("-", "_")
    assert f"args.{dest}" in _SOURCE, f"{flag} is parsed but never forwarded to train()"


def test_pair_geometry_is_persisted_to_config() -> None:
    """`load_state_dict` is strict and the pair-MLP width depends on rel_mode.

    A checkpoint trained with `um` geometry is simply unloadable if the config
    does not record it, so this is a hard requirement, not a nicety.
    """
    for key in ("pair_rel_mode", "pair_rel_scale_um"):
        assert f'"{key}":' in _SOURCE, f"train never writes {key} to config.json"
        assert f'config["{key}"]' in _PREDICT_SOURCE, f"load_model never reads {key}"


def test_predict_default_config_makes_v1_checkpoints_load_as_legacy() -> None:
    """Configs written before this change have no pair_rel_mode key."""
    assert predict_mod._DEFAULT_CONFIG["pair_rel_mode"] == "legacy"


@pytest.mark.parametrize("module", ["train", "predict"])
def test_every_predict_edges_call_passes_original_resolution_voxel_size(module: str) -> None:
    """The scale-space trap: `voxel_size` in these scripts is NOT what to pass.

    Call sites hand `predict_edges` coords multiplied back up to ORIGINAL
    resolution (`coords * downsample`), so the microns-per-voxel it needs is the
    dataset's `scale`. The local named `voxel_size` is `scale * downsample`
    (used for pooling and gating), which is 4x too large in y/x here -- passing
    it would silently quadruple every physical displacement the pair MLP sees,
    with no error anywhere. Both scripts bind the correct value as
    `orig_voxel_size`; this pins that every call site uses it.
    """
    import ast

    source = _SOURCE if module == "train" else _PREDICT_SOURCE
    calls = [
        n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "predict_edges"
    ]
    assert calls, f"no predict_edges call found in {module} -- test is stale"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "voxel_size" in kw, (
            f"{module}:{call.lineno} calls predict_edges without voxel_size; "
            f"rel_mode='um' would raise"
        )
        val = kw["voxel_size"]
        assert isinstance(val, ast.Name) and val.id == "orig_voxel_size", (
            f"{module}:{call.lineno} passes "
            f"{ast.unparse(val)!r} -- must be `orig_voxel_size` (= scale), not the "
            f"downsampled `voxel_size` (= scale * downsample)"
        )
