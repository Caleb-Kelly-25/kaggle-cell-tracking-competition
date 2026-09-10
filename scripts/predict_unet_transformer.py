#!/usr/bin/env python
"""Run UNet + transformer edge prediction on datasets and export to .geff.

Usage:
    uv run scripts/predict_unet_transformer.py --split 0
"""

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

import tracksdata as td

from tracking_cellmot.io import open_dataset, save_graph

# Import model and helpers from companion training script.
sys.path.insert(0, str(Path(__file__).parent))
from train_unet_transformer import (
    DEFAULT_METHOD,
    UNetNodeTransformer,
    extract_pos_features,
    _POS_EMBED_DIM,
)
from tracking_cellmot.models import TemporalUNet3D

from dataspec import USERNAME, INTERACTIVE, WEIGHTS_PATH
from evaluate import evaluate_run
from tracking_cellmot.metrics import summarise


# =============================================================================
# Prediction config
# =============================================================================

@dataclass
class PredictConfig:
    """All hyperparameters that can affect prediction quality / score.

    Detection
    ---------
    det_threshold : float
        Minimum sigmoid probability for a local-max peak to be kept.

    Edge filtering
    --------------
    edge_activation : str
        Activation applied to raw edge logits: ``"sigmoid"`` (independent
        per-edge scores) or ``"softmax"`` (row-normalised over t+1 nodes).
    threshold : float
        Minimum edge probability to consider a link at all.
    max_parents_per_node : int
        Maximum number of incoming edges per node (typically 1).
    max_children_per_node : int
        Maximum number of outgoing edges per node (1 = no divisions, 2 = divisions allowed).
    """
    # Detection
    det_threshold: float = 0.5
    det_tta: bool = True  # flip-xy TTA for detection logits
    pool_kernel_um: float = 3.0  # max-pool kernel size in µm for detection peak extraction
    # Count calibration.  The score is scaled by (1 - 0.1*(N_pred - N_true)/N_true),
    # so aiming the *count* at N_true optimises the objective directly instead of
    # hoping a global probability threshold happens to land there.
    target_nodes: int | None = None        # total nodes to aim for across the video
    target_nodes_from_geff: bool = False   # read estimated_number_of_nodes from the .geff
    # Edge pruning.  The scorer keeps only consecutive-frame edges and caps
    # out-degree at 2 by *arbitrary* edge id; pruning ourselves by probability
    # means our best links are the ones that survive.  0 disables a cap.
    max_out_degree: int = 2                # <=2 children (a division)
    max_in_degree: int = 1                 # <=1 parent (merges are invalid)
    # Node support.  Only edges score, so an unlinked node earns nothing and still
    # pays the node-count penalty; 2 drops isolated nodes (strictly beneficial).
    min_track_nodes: int = 2
    # Edge filtering
    edge_activation: str = "softmax"  # see _edge_probs for the full set
    threshold: float = 0.5
    # Candidate gating.  Median true inter-frame displacement is ~1.8 um (p90 3.6),
    # so a gate of ~10-15 um keeps essentially every real edge while removing
    # implausible candidates -- and, because the softmax is normalised over SOURCE
    # nodes, it stops distant nodes from stealing probability mass.
    gate_um: float = float("inf")     # inf = no gate (legacy behaviour)
    null_logit: float = 0.0           # only used by the "*_null" activations

    # ILP post-processing
    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0

    max_parents_per_node: int | None = None
    max_children_per_node: int | None = None

    def __post_init__(self) -> None:
        # When ILP is enabled it handles parent/children constraints itself,
        # so greedy limits are left unconstrained (None).  When ILP is
        # disabled, default to 1/1 to avoid unconstrained edge assignment.
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            if self.max_children_per_node is None:
                self.max_children_per_node = 2



# =============================================================================
# Helpers
# =============================================================================


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# =============================================================================
# Graph building
# =============================================================================

def build_graph(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
) -> td.graph.InMemoryGraph:
    """Build a tracksdata graph from detection coords and predicted edges.

    Avoids ``add_node_attr_key`` to sidestep a tracksdata/Polars compatibility
    issue where the float default value is mistakenly used as a dtype.
    Probabilities are passed as-is (softmax output, already in [0, 1]).
    """
    graph = td.graph.InMemoryGraph()

    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in coords
    ])

    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": node_ids[src],
                "target_id": node_ids[tgt],
                "edge_prob": prob,
                "edge_dist": dist,
            }
            for src, tgt, prob, dist in edges
        ])

    return graph


def prune_edges(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
    max_out_degree: int = 2,
    max_in_degree: int = 1,
) -> list[tuple[int, int, float, float]]:
    """Drop edges the scorer discards, choosing survivors by predicted probability.

    The metric (a) keeps only edges with ``t_target == t_source + 1`` — backward
    and gap-spanning edges are thrown away — and (b) caps out-degree at 2 keeping
    the two *lowest edge ids*, an arbitrary tiebreak that can discard our best
    link.  Extra incoming edges are biologically impossible merges and score as
    false positives.  Pruning here, highest probability first, means we choose
    which edges survive instead of letting insertion order choose for us.

    A degree limit of ``0`` disables that cap (used when the ILP solver runs,
    since it enforces its own flow constraints).
    """
    if not edges:
        return edges

    t = coords[:, 0].astype(np.int64)

    # Consecutive frames only — everything else is discarded by the scorer.
    kept = [e for e in edges if t[e[1]] - t[e[0]] == 1]

    # Highest probability first, so each cap keeps the most confident links.
    kept.sort(key=lambda e: e[2], reverse=True)

    out_count: dict[int, int] = {}
    in_count: dict[int, int] = {}
    pruned: list[tuple[int, int, float, float]] = []
    for src, tgt, prob, dist in kept:
        if max_out_degree and out_count.get(src, 0) >= max_out_degree:
            continue
        if max_in_degree and in_count.get(tgt, 0) >= max_in_degree:
            continue
        out_count[src] = out_count.get(src, 0) + 1
        in_count[tgt] = in_count.get(tgt, 0) + 1
        pruned.append((src, tgt, prob, dist))

    n_dropped = len(edges) - len(pruned)
    if n_dropped:
        print(f"  pruned {n_dropped}/{len(edges)} edges "
              f"(non-consecutive / degree caps)", flush=True)
    return pruned


def prune_nodes_by_support(
    graph: td.graph.BaseGraph,
    min_track_nodes: int = 2,
) -> td.graph.BaseGraph:
    """Drop nodes belonging to track fragments smaller than *min_track_nodes*.

    Only *edges* score.  A node with no edges therefore cannot contribute a true
    positive, yet it still counts toward ``N_pred`` and so pays the node-count
    penalty ``(1 - 0.1*(N_pred - N_true)/N_true)``.  Dropping isolated nodes is
    therefore *strictly* score-improving: edge TP/FP/FN are untouched and the
    penalty shrinks.

    ``min_track_nodes=2`` (default) removes only isolated nodes — the provably
    beneficial case.  Larger values also discard short fragments, trading a
    possible true-positive edge against a smaller node count: sweep, don't assume.
    """
    if min_track_nodes <= 1 or graph.num_nodes() == 0:
        return graph

    node_ids = list(graph.node_ids())

    if min_track_nodes == 2:
        # Isolated nodes only — degrees suffice, no component search needed.
        out_d = graph.out_degree(node_ids)
        in_d = graph.in_degree(node_ids)
        keep = [n for n, o, i in zip(node_ids, out_d, in_d) if (o + i) > 0]
    else:
        from collections import deque
        keep = []
        remaining = set(node_ids)
        while remaining:
            seed = next(iter(remaining))
            comp = {seed}
            queue = deque([seed])
            while queue:
                cur = queue.popleft()
                for nb in graph.successors(cur) + graph.predecessors(cur):
                    if nb not in comp:
                        comp.add(nb)
                        queue.append(nb)
            remaining -= comp
            if len(comp) >= min_track_nodes:
                keep.extend(comp)

    n_dropped = graph.num_nodes() - len(keep)
    if n_dropped <= 0:
        return graph
    print(f"  pruned {n_dropped}/{graph.num_nodes()} unsupported nodes "
          f"(fragments < {min_track_nodes} nodes)", flush=True)
    return graph.filter(node_ids=keep).subgraph()


def _append_results_csv(path: Path, row: dict) -> None:
    """Append one run's config + score to a CSV, writing the header if new."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _edge_probs(
    raw: torch.Tensor,
    activation: str = "softmax",
    null_logit: float = 0.0,
) -> torch.Tensor:
    """Convert raw pair logits ``(n_src, n_tgt)`` into edge probabilities.

    ``softmax`` (legacy) normalises down each *column*, so every target at t+1
    spreads exactly one unit of probability across candidate parents at t. That
    forces every child to take a parent, which is the structural reason this
    model over-predicts divisions: a child whose real parent was never detected
    still hands its full unit of mass to some neighbouring node, creating a fork.

    ``*_null`` adds a null logit to the denominator so a detection may have NO
    parent -- Trackastra's "parental softmax", ``exp(a) / (exp(b) + sum exp(a))``.
    That is exactly ``col_softmax * sigmoid(logsumexp - null_logit)``.

    ``dual_*`` multiplies the column term by the row term, penalising a parent
    that is splitting its own mass across several children -- i.e. it regulates
    out-degree, which the column softmax never did.
    """
    if activation == "sigmoid":
        return torch.sigmoid(raw)

    use_null = activation.endswith("_null")
    col = torch.softmax(raw, dim=0)
    if use_null:
        col = col * torch.sigmoid(torch.logsumexp(raw, dim=0, keepdim=True) - null_logit)

    if activation in ("softmax", "softmax_null"):
        return col
    if activation in ("dual_softmax", "dual_softmax_null"):
        row = torch.softmax(raw, dim=1)
        if use_null:
            row = row * torch.sigmoid(torch.logsumexp(raw, dim=1, keepdim=True) - null_logit)
        return torch.sqrt(col * row)
    raise ValueError(f"Unknown edge_activation: {activation!r}")


# =============================================================================
# Model loading
# =============================================================================

_DEFAULT_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
}


def load_model(
    weights_path: Path, device: torch.device,
) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:
    """Reconstruct UNetNodeTransformer from saved config + weights.

    Reads ``config.json`` from the same directory as the weights file.
    Falls back to ``_DEFAULT_CONFIG`` if the file is missing.

    Returns ``(model, window_size, downsample)``.
    """
    config_path = weights_path.parent / "config.json"
    if config_path.exists():
        config = {**_DEFAULT_CONFIG, **json.loads(config_path.read_text())}
    else:
        print(f"Warning: config.json not found at {config_path}, using defaults.", flush=True)
        config = _DEFAULT_CONFIG

    # Support legacy configs that used "downsample_factor" (scalar).
    if "downsample_factor" in config and "downsample" not in config:
        df = config["downsample_factor"]
        config["downsample"] = [df, df, df]

    downsample = tuple(config["downsample"])

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=config["unet_out_channels"],
        layers=config["unet_layers"],
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=config["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config["window_size"], downsample


# =============================================================================
# Per-frame loading
# =============================================================================

def _load_frame(
    zarr_arr,
    t: int,
    target_shape: list[int],
    downsample: tuple[int, ...] = (1, 1, 1),
) -> torch.Tensor:
    """Load one frame from zarr with strided spatial downsample (no normalisation)."""
    dz, dy, dx = downsample
    raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
    frame = torch.from_numpy(raw)
    if list(frame.shape) != target_shape:
        frame = F.interpolate(
            frame[None, None], size=target_shape,
            mode="trilinear", align_corners=False,
        )[0, 0]
    return frame


# =============================================================================
# Inference
# =============================================================================

def pool_kernel_from_um(
    um: float,
    voxel_size: tuple[float, ...],
) -> tuple[int, ...]:
    """Convert a physical suppression distance (microns) to a per-axis voxel kernel.

    Each axis gets ``round(um / voxel_size_axis)`` voxels, forced to odd
    (for symmetric padding) and at least 1.

    Parameters
    ----------
    um : float
        Desired suppression distance in microns.
    voxel_size : tuple[float, ...]
        Per-axis voxel sizes in microns, e.g. ``(1.625, 0.40625, 0.40625)``.
    """
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def _read_estimated_nodes(ds_path: Path) -> float | None:
    """Read ``estimated_number_of_nodes`` from the dataset's ``.geff`` (None if absent)."""
    geff_path = ds_path.parent / f"{ds_path.stem}.geff"
    if not geff_path.exists():
        return None
    try:
        from geff import GeffMetadata
        val = (GeffMetadata.read(geff_path).extra or {}).get("estimated_number_of_nodes")
        return float(val) if val is not None else None
    except Exception:
        return None


def _detect_cells_pooled(
    det_logits: torch.Tensor,
    t: int,
    det_threshold: float = 0.5,
    pool_kernel: tuple[int, ...] = (3, 3, 3),
    top_k: int | None = None,
) -> np.ndarray:
    """Extract cell coordinates via max-pool local-max (same as training).

    Coordinates are returned in the downsampled grid.  The caller is
    responsible for scaling back to original resolution if needed.

    Parameters
    ----------
    det_logits : torch.Tensor
        (1, Z, Y, X) raw logits.
    t : int
        Time index to prepend as the first column.
    det_threshold : float
        Minimum sigmoid probability for a peak to be considered (default 0.5).
    pool_kernel : tuple[int, ...]
        Per-axis kernel size for local-max pooling,
        e.g. ``(3, 11, 11)`` for anisotropic data.

    Returns
    -------
    np.ndarray
        (N, 4) int16 array with columns [t, z, y, x] in downsampled space.
    """
    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)

    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)

    # Count calibration: keep only the *top_k* most confident peaks in this frame.
    if top_k is not None and peak_idx.shape[0] > top_k:
        scores = logits[0, 0][peak_idx[:, 0], peak_idx[:, 1], peak_idx[:, 2]]
        keep = torch.topk(scores, top_k).indices
        peak_idx = peak_idx[keep]

    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


@torch.no_grad()
def predict_video(
    model: UNetNodeTransformer,
    ds_path: Path,
    device: torch.device,
    cfg: PredictConfig,
    window_size: int = 2,
    max_frames: int | None = None,
    unet_batch_size: int = 4,
    downsample: tuple[int, ...] = (1, 4, 4),
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Run inference on a single video using sliding windows of W frames.

    Windows slide with stride ``W - 1`` so every consecutive pair is covered
    exactly once.  UNet features from each window are reused for edge
    prediction on all ``W - 1`` consecutive pairs within the window.

    Returns
    -------
    coords : np.ndarray
        Shape (N, 4) — columns [t, z, y, x] in original resolution.
    edges : list of (src_idx, tgt_idx, prob, distance) tuples
    """
    ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=downsample)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {ds_path}")
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])

    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)
    image_shape = (T,) + ds.image_shape[1:]
    target_shape = list(image_shape[1:])

    ds_arr = np.array(downsample, dtype=np.float32)  # for coord rescaling at the end
    ds_arr_t = torch.from_numpy(ds_arr).to(device)   # for predict_edges (original-space coords)
    pos_feat_dim = 4 * _POS_EMBED_DIM
    W = window_size
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)

    # Count calibration: convert a per-video node target into a per-frame quota.
    target_total = cfg.target_nodes
    if cfg.target_nodes_from_geff:
        target_total = _read_estimated_nodes(ds_path) or target_total
    top_k_per_frame = None
    if target_total:
        top_k_per_frame = max(1, int(round(float(target_total) / max(T, 1))))
        print(
            f"  count calibration: target {int(target_total)} nodes over {T} frames "
            f"-> top-{top_k_per_frame} peaks/frame",
            flush=True,
        )

    # Running node registry — each entry records the frame-t detections.
    # coord_offset[t] = (start, end) half-open range into the stacked array.
    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count: int = 0
    all_edges: list[tuple[int, int, float, float]] = []

    # Sliding windows with stride W-1 cover every consecutive pair exactly once.
    stride = max(W - 1, 1)
    window_starts = list(range(0, T - W + 1, stride))
    # Ensure the very last pair (T-2 → T-1) is covered.
    if not window_starts or window_starts[-1] + W < T:
        last = max(T - W, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for ws in tqdm(
        window_starts,
        desc="  windows",
        leave=False,
        disable=not INTERACTIVE,
    ):
        frame_indices = list(range(ws, ws + W))

        # --- UNet encode (single window, batch_size=1) ---
        imgs = torch.stack([
            _load_frame(zarr_arr, t, target_shape, downsample)
            for t in frame_indices
        ])  # (W, *spatial)
        # Quantile normalisation (0.1%–99.9%) to match training pipeline.
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        imgs = imgs.unsqueeze(0).to(device)   # (1, W, *spatial)

        unet_out, det_logits = model.encode(imgs)
        # unet_out: (1, W, C, *spatial_down), det_logits: list of W × (1, 1, *spatial_down)

        # Detection TTA: original + flip-x + flip-y + flip-xy, average logits.
        # TTA: flip along Y (-2) and X (-1) only.  Z is excluded because
        # the data is highly anisotropic (Z resolution ~4x coarser than XY),
        # so Z-flips would produce out-of-distribution inputs.
        if cfg.det_tta:
            tta_flips = [(-1,), (-2,), (-2, -1)]
            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4

        del imgs

        # --- Detect cells in each frame (dedup across windows) ---
        for f_idx, t in enumerate(frame_indices):
            if t not in seen_frames:
                arr = _detect_cells_pooled(
                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,
                    top_k=top_k_per_frame,
                )
                coord_offset[t] = (global_node_count, global_node_count + len(arr))
                global_node_count += len(arr)
                coord_lists.append(arr)
                seen_frames.add(t)

        coords_so_far = (
            np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
        )

        # --- Edge prediction for each consecutive pair in the window ---
        for f_idx in range(W - 1):
            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
            if (t_src, t_tgt) in seen_pairs:
                continue
            seen_pairs.add((t_src, t_tgt))

            if t_src not in coord_offset or t_tgt not in coord_offset:
                continue
            s_src, e_src = coord_offset[t_src]
            s_tgt, e_tgt = coord_offset[t_tgt]
            if e_src == s_src or e_tgt == s_tgt:
                continue

            c_src = coords_so_far[s_src:e_src]
            c_tgt = coords_so_far[s_tgt:e_tgt]
            n_src, n_tgt = len(c_src), len(c_tgt)
            idx_src = np.arange(s_src, e_src, dtype=np.int64)
            idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

            # Build tensors (batch_size=1).
            p_coords_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            p_coords_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            # Use window-relative time (f_idx, f_idx+1) normalised by W, not absolute frame index.
            window_shape = (W,) + image_shape[1:]
            c_src_rel = c_src.copy()
            c_src_rel[:, 0] = f_idx
            c_tgt_rel = c_tgt.copy()
            c_tgt_rel[:, 0] = f_idx + 1
            p_pos_src = torch.from_numpy(extract_pos_features(c_src_rel, window_shape)).unsqueeze(0).to(device)
            p_pos_tgt = torch.from_numpy(extract_pos_features(c_tgt_rel, window_shape)).unsqueeze(0).to(device)
            p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
            p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

            unet_feat_src = model._index_features(
                unet_out[:, f_idx], p_coords_src, p_mask_src,
            )
            unet_feat_tgt = model._index_features(
                unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,
            )
            edge_logits_pair = model.predict_edges(
                unet_feat_src, unet_feat_tgt,
                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                p_pos_src, p_pos_tgt,
                p_mask_src, p_mask_tgt,
            )  # (1, n_src, n_tgt)

            raw = edge_logits_pair[0]

            # Physical (um) distance between every candidate pair, for gating.
            cs = coords_so_far[idx_src, 1:].astype(np.float32)
            ct = coords_so_far[idx_tgt, 1:].astype(np.float32)
            vs_np = np.asarray(voxel_size, dtype=np.float32)
            d_um = np.linalg.norm((cs[:, None, :] - ct[None, :, :]) * vs_np, axis=-1)

            gate_np = None
            if np.isfinite(cfg.gate_um):
                gate_np = d_um <= cfg.gate_um
                raw = raw.masked_fill(
                    torch.from_numpy(~gate_np).to(raw.device), -1e4,
                )

            probs = _edge_probs(raw, cfg.edge_activation, cfg.null_logit).cpu().numpy()
            if gate_np is not None:
                # Zero gated pairs outright: a fully-masked column would otherwise
                # renormalise to a uniform (and possibly supra-threshold) value.
                probs = np.where(gate_np, probs, 0.0)

            sel = np.argwhere(probs > cfg.threshold)
            if sel.size:
                sel = sel[np.argsort(-probs[sel[:, 0], sel[:, 1]])]
            candidates = [(float(probs[i, j]), int(i), int(j)) for i, j in sel]

            children_count: dict[int, int] = {}
            parents_count: dict[int, int] = {}

            for prob, i, j in candidates:
                n_ch = children_count.get(i, 0)
                n_pa = parents_count.get(j, 0)
                if cfg.max_children_per_node is not None and n_ch >= cfg.max_children_per_node:
                    continue
                if cfg.max_parents_per_node is not None and n_pa >= cfg.max_parents_per_node:
                    continue

                gi, gj = int(idx_src[i]), int(idx_tgt[j])
                dist = float(np.linalg.norm(
                    coords_so_far[gi, 1:].astype(np.float32)
                    - coords_so_far[gj, 1:].astype(np.float32)
                ))
                all_edges.append((gi, gj, float(prob), dist))
                children_count[i] = n_ch + 1
                parents_count[j] = n_pa + 1

        del unet_out

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    # Scale spatial coords back to original resolution.
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    coords = coords.astype(np.int16)
    return coords, all_edges


# =============================================================================
# Prediction loop
# =============================================================================

def predict(
    data_dir: Path,
    fold: int,
    splits_file: Path,
    weights_path: Path,
    cfg: PredictConfig,
    method: str = DEFAULT_METHOD,
    debug_video: Path | None = None,
    unet_batch_size: int = 4,
    video_slice: slice | None = None,
    evaluate: bool = False,
    log_csv: str | None = None,
    run_name: str | None = None,
) -> None:
    """Run inference on the test split and save predictions as .geff files."""
    if debug_video is not None:
        test_names = [debug_video.name]
        data_dir = debug_video.parent
    else:
        folds = json.loads(splits_file.read_text())
        test_names = folds[fold]["test"]
        if video_slice is not None:
            test_names = test_names[video_slice]

    from dataspec import PREDICTIONS_PATH
    output_dir = PREDICTIONS_PATH / USERNAME / method / f"split_{fold}"
    if output_dir.exists():
        import shutil
        for old in output_dir.glob("*.geff"):
            if old.is_dir():
                shutil.rmtree(old)
            else:
                old.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, window_size, downsample = load_model(weights_path, device)
    print(
        f"Fold {fold}: {len(test_names)} datasets | "
        f"weights={weights_path} | device={device} | window_size={window_size} | pool_kernel_um={cfg.pool_kernel_um}",
        flush=True,
    )

    for name in tqdm(test_names, desc="Predicting", disable=not INTERACTIVE):
        ds_path = data_dir / name
        coords, edges = predict_video(
                model, ds_path, device,
                cfg=cfg,
                window_size=window_size,
                unet_batch_size=unet_batch_size,
                downsample=downsample,
            )
        edges = prune_edges(
            coords, edges,
            # The ILP enforces its own flow constraints, so only pre-filter for it.
            max_out_degree=0 if cfg.use_ilp else cfg.max_out_degree,
            max_in_degree=0 if cfg.use_ilp else cfg.max_in_degree,
        )
        graph = build_graph(coords, edges)
        if cfg.use_ilp and graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight,
            )
            with suppress_output():
                graph = solver.solve(graph)
        # Applied after the ILP too: the solver can leave nodes unlinked.
        graph = prune_nodes_by_support(graph, cfg.min_track_nodes)
        save_graph(graph, output_dir / f"{name}.geff")

    print(f"Saved {len(test_names)} predictions to {output_dir}", flush=True)

    if evaluate:
        run = {
            "username": USERNAME,
            "method": method,
            "split": f"split_{fold}",
            "dir": output_dir,
            "geffs": sorted(output_dir.glob("*.geff")),
        }
        results = evaluate_run(run)
        s = summarise(results)
        print(
            f"Evaluation ({len(results)} videos): "
            f"score={s['score']:.4f}  "
            f"edge_jaccard={s['edge_jaccard']:.4f}  "
            f"adj_edge_jaccard={s['adj_edge_jaccard']:.4f} (n_adj={s['n_adj']})  "
            f"division_jaccard={s['division_jaccard']:.4f} "
            f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})  "
            f"node_recall={s['node_recall']:.4f}  (n={s['n']})",
            flush=True,
        )

        if log_csv:
            import time
            _append_results_csv(Path(log_csv), {
                "run": run_name or method,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": method,
                "split": fold,
                "n_videos": s["n"],
                "score": s["score"],
                "edge_jaccard": s["edge_jaccard"],
                "adj_edge_jaccard": s["adj_edge_jaccard"],
                "division_jaccard": s["division_jaccard"],
                "node_recall": s["node_recall"],
                "division_tp": s["division_tp"],
                "division_fp": s["division_fp"],
                "division_fn": s["division_fn"],
                "det_threshold": cfg.det_threshold,
                "pool_kernel_um": cfg.pool_kernel_um,
                "target_nodes": cfg.target_nodes,
                "target_nodes_from_geff": cfg.target_nodes_from_geff,
                "max_out_degree": cfg.max_out_degree,
                "max_in_degree": cfg.max_in_degree,
                "min_track_nodes": cfg.min_track_nodes,
                "edge_activation": cfg.edge_activation,
                "threshold": cfg.threshold,
                "gate_um": cfg.gate_um,
                "null_logit": cfg.null_logit,
                "max_children_per_node": cfg.max_children_per_node,
                "max_parents_per_node": cfg.max_parents_per_node,
                "det_tta": cfg.det_tta,
                "use_ilp": cfg.use_ilp,
                "weights": str(weights_path),
            })
            print(f"  logged run to {log_csv}", flush=True)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run UNet + transformer edge prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Default: DATASET_PATH")
    parser.add_argument("--splits", type=str, default=None,
                        help="Default: DATASET_PATH/dataset_splits.json")
    parser.add_argument("--split", type=str, default="0",
                        help="Split index (0-4) or 'all'.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file. "
                             "Default: weights/{method}/split_{split}/edge_predictor_best.pth")
    parser.add_argument("--debug-video", type=str, default=None,
                        help="Path to a single dataset. Ignores fold/splits.")
    parser.add_argument("--slice", type=str, default=None,
                        help="Python slice of the test list, e.g. ':1' for first video, "
                             "'2:5' for videos 2-4.")
    parser.add_argument("--unet-batch-size", type=int, default=4,
                        help="Number of frame pairs per UNet forward pass (default: 4).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation against GT after saving predictions.")
    parser.add_argument("--det-threshold", type=float, default=0.99,
                        help="Min sigmoid probability for a detection peak to be kept. "
                             "Default 0.99: the detector is poorly calibrated because the "
                             "ground truth is sparse (only some cells annotated), so a high "
                             "threshold keeps precision up. Sweep it for your model.")
    parser.add_argument("--target-nodes", type=int, default=None,
                        help="Cap detections so the video's total node count lands near N "
                             "(per-frame quota = N/T, keeping the most confident peaks). "
                             "Targets the metric's node-count penalty directly.")
    parser.add_argument("--target-nodes-from-geff", action="store_true",
                        help="Read the per-video target from estimated_number_of_nodes in the "
                             "dataset's .geff (train/val only -- test videos have no .geff).")
    parser.add_argument("--max-out-degree", type=int, default=2,
                        help="Keep at most N outgoing edges per node, highest probability first "
                             "(default 2 = a division). The scorer caps out-degree at 2 by "
                             "arbitrary edge id, so pruning here keeps our best links. 0 disables.")
    parser.add_argument("--max-in-degree", type=int, default=1,
                        help="Keep at most N incoming edges per node (default 1; merges are "
                             "biologically invalid and score as false positives). 0 disables.")
    parser.add_argument("--edge-activation", type=str, default="softmax",
                        choices=["softmax", "sigmoid", "softmax_null",
                                 "dual_softmax", "dual_softmax_null"],
                        help="How pair logits become edge probabilities. 'softmax' (legacy) "
                             "forces every child to take a parent; '*_null' allows no parent; "
                             "'dual_*' also penalises a parent splitting mass across children.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum edge probability to emit an edge (default 0.5).")
    parser.add_argument("--gate-um", type=float, default=float("inf"),
                        help="Discard candidate links longer than N microns before scoring "
                             "(default inf = no gate). Median true displacement is ~1.8 um.")
    parser.add_argument("--null-logit", type=float, default=0.0,
                        help="Null-parent logit for the '*_null' activations (default 0.0).")
    parser.add_argument("--max-children-per-node", type=int, default=None,
                        help="Greedy cap on outgoing edges per node (default 2; 1 = no divisions).")
    parser.add_argument("--max-parents-per-node", type=int, default=None,
                        help="Greedy cap on incoming edges per node (default 1).")
    parser.add_argument("--min-track-nodes", type=int, default=2,
                        help="Drop nodes in track fragments smaller than N (default 2 = drop "
                             "isolated nodes, which can never score an edge but still pay the "
                             "node-count penalty). 1 disables.")
    parser.add_argument("--log-csv", type=str, default=None,
                        help="Append this run's config and CV score to a CSV (with --evaluate), "
                             "so experiments are comparable.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Label for this run in --log-csv (defaults to --method).")
    parser.add_argument("--no-det-tta", dest="det_tta", action="store_false", default=True,
                        help="Disable flip-xy detection TTA (ON by default). TTA runs 4 forward "
                             "passes per window, so disabling it makes the detection pass ~4x "
                             "faster -- useful for wide inference sweeps, and lets us ablate it.")
    parser.add_argument("--use-ilp", action="store_true",
                        help="Post-process the predicted graph with the tracksdata ILP "
                             "solver (global, flow-consistent linking) instead of greedy "
                             "assignment. Needs pyscipopt; produces cleaner tracks.")
    parser.add_argument("--ilp-edge-weight", type=float, default=-1.0,
                        help="ILP: weight on edge_prob (default -1.0).")
    parser.add_argument("--ilp-appearance-weight", type=float, default=0.1,
                        help="ILP: cost of a track appearing (default 0.1).")
    parser.add_argument("--ilp-disappearance-weight", type=float, default=0.1,
                        help="ILP: cost of a track disappearing (default 0.1).")
    parser.add_argument("--ilp-division-weight", type=float, default=1.0,
                        help="ILP: cost of a division; lower to allow more splits (default 1.0).")

    args = parser.parse_args()

    from dataspec import DATASET_PATH
    data_dir = Path(args.data_dir) if args.data_dir else Path(DATASET_PATH)
    splits_file = Path(args.splits) if args.splits else data_dir / "dataset_splits.json"
    debug_video = Path(args.debug_video) if args.debug_video else None
    video_slice = (
        slice(*[int(x) if x else None for x in args.slice.split(":")])
        if args.slice else None
    )
    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        target_nodes=args.target_nodes,
        target_nodes_from_geff=args.target_nodes_from_geff,
        max_out_degree=args.max_out_degree,
        max_in_degree=args.max_in_degree,
        min_track_nodes=args.min_track_nodes,
        det_tta=args.det_tta,
        edge_activation=args.edge_activation,
        threshold=args.threshold,
        gate_um=args.gate_um,
        null_logit=args.null_logit,
        max_children_per_node=args.max_children_per_node,
        max_parents_per_node=args.max_parents_per_node,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )

    folds = range(5) if args.split == "all" else [int(args.split)]

    for fold in folds:
        weights_path = (
            Path(args.weights) if args.weights
            else WEIGHTS_PATH / args.method / f"split_{fold}" / "edge_predictor_best.pth"
        )
        predict(
            data_dir=data_dir,
            fold=fold,
            splits_file=splits_file,
            weights_path=weights_path,
            cfg=cfg,
            method=args.method,
            debug_video=debug_video,
            unet_batch_size=args.unet_batch_size,
            video_slice=video_slice,
            evaluate=args.evaluate,
            log_csv=args.log_csv,
            run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
