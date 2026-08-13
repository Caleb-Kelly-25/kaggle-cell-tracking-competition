#!/usr/bin/env python
"""Exploratory data analysis for the cell-tracking ground-truth graphs.

Characterises every ``{name}.geff`` in a data directory (the paired ``.zarr``
is read for image shape/scale only — no volume is loaded), reporting the
structural facts that drive the metric:

* **Size** — nodes, edges, timepoints, image ``(T, Z, Y, X)``.
* **Sparsity** — annotated nodes vs the GEFF ``estimated_number_of_nodes``
  (the ``T_true`` used by the adjusted-edge-Jaccard penalty).
* **Divisions** — dividing nodes (out-degree >= 2), division rate, and any
  structural anomalies (out-degree > 2, or in-degree >= 2 "merges").
* **Track structure** — track starts (in-degree 0), ends (out-degree 0),
  nodes per timepoint.
* **Linking difficulty** — per-edge temporal gap ``dt`` and physical
  displacement in microns (relevant to the 7 µm node-match radius).

Usage:
    python scripts/eda.py                         # uses dataspec.DATASET_PATH
    python scripts/eda.py --data-dir data/train
    python scripts/eda.py --data-dir /kaggle/input/competitions/biohub-cell-tracking-during-development/train
    python scripts/eda.py --out-csv results/eda.csv

This reports on *ground truth*; it does not need predictions. For the baseline
FP/FN breakdown, run ``scripts/evaluate.py`` which prints per-dataset
edge/division TP/FP/FN.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
from geff import GeffMetadata

from tracking_cellmot.io import DEFAULT_SCALE, list_datasets, open_dataset

from dataspec import DATASET_PATH

NID = td.DEFAULT_ATTR_KEYS.NODE_ID
SRC = td.DEFAULT_ATTR_KEYS.EDGE_SOURCE
TGT = td.DEFAULT_ATTR_KEYS.EDGE_TARGET


def _load_graph(geff_path: Path) -> td.graph.BaseGraph:
    result = td.graph.IndexedRXGraph.from_geff(geff_path)
    return result[0] if isinstance(result, tuple) else result


def _estimated_n_total(geff_path: Path) -> float:
    """``estimated_number_of_nodes`` from GEFF metadata (NaN if absent)."""
    try:
        meta = GeffMetadata.read(geff_path)
    except Exception:
        return float("nan")
    val = (meta.extra or {}).get("estimated_number_of_nodes")
    return float(val) if val is not None else float("nan")


def _shape_and_scale(
    data_dir: Path, name: str,
) -> tuple[tuple[int, ...] | None, tuple[float, float, float]]:
    """Return image ``(T, Z, Y, X)`` and voxel scale, reading zarr metadata only.

    Falls back to ``(None, DEFAULT_SCALE)`` when no ``.zarr`` is present.
    """
    try:
        ds = open_dataset(data_dir / name, load_image=False)
        return ds.image_shape, ds.scale
    except FileNotFoundError:
        return None, DEFAULT_SCALE


def _describe(arr: np.ndarray) -> dict[str, float]:
    """min / p50 / mean / p90 / max of a 1-D array (NaNs when empty)."""
    if arr.size == 0:
        return {k: float("nan") for k in ("min", "p50", "mean", "p90", "max")}
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def analyse_dataset(
    data_dir: Path, name: str,
) -> tuple[dict, np.ndarray]:
    """Compute EDA stats for one dataset; return (row, per-edge displacements µm)."""
    geff_path = data_dir / f"{name}.geff"
    graph = _load_graph(geff_path)

    n_nodes = graph.num_nodes()
    n_edges = graph.num_edges()
    shape, scale = _shape_and_scale(data_dir, name)
    est_total = _estimated_n_total(geff_path)

    row: dict = {
        "dataset": name,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "img_T": shape[0] if shape else None,
        "img_Z": shape[1] if shape else None,
        "img_Y": shape[2] if shape else None,
        "img_X": shape[3] if shape else None,
        "est_total_nodes": est_total,
        # ratio of *annotated* nodes to the coarse true-node estimate → sparsity
        "annotated_frac": (n_nodes / est_total) if est_total and est_total > 0 else float("nan"),
    }

    if n_nodes == 0:
        row.update(
            n_timepoints=0, t_min=None, t_max=None,
            nodes_per_t_p50=float("nan"), nodes_per_t_max=float("nan"),
            n_track_starts=0, n_track_ends=0,
            n_divisions=0, division_rate=float("nan"),
            n_outdeg_gt2=0, n_merges=0,
            edges_dt1=0, edges_dt_gap=0,
            disp_um_p50=float("nan"), disp_um_p90=float("nan"), disp_um_max=float("nan"),
        )
        return row, np.array([])

    node_ids = graph.node_ids()
    na = graph.node_attrs(attr_keys=[NID, "t", "z", "y", "x"])

    # Timepoints and per-frame density.
    t_col = na["t"]
    per_t = na.group_by("t").agg(pl.len().alias("n"))["n"].to_numpy()
    row["n_timepoints"] = int(na["t"].n_unique())
    row["t_min"] = int(t_col.min())
    row["t_max"] = int(t_col.max())
    row["nodes_per_t_p50"] = float(np.percentile(per_t, 50))
    row["nodes_per_t_max"] = float(np.max(per_t))

    # Degrees → track starts/ends, divisions, anomalies.
    out_deg = np.asarray(graph.out_degree(node_ids))
    in_deg = np.asarray(graph.in_degree(node_ids))
    n_div = int(np.count_nonzero(out_deg >= 2))
    row["n_track_starts"] = int(np.count_nonzero(in_deg == 0))
    row["n_track_ends"] = int(np.count_nonzero(out_deg == 0))
    row["n_divisions"] = n_div
    row["division_rate"] = n_div / n_nodes
    row["n_outdeg_gt2"] = int(np.count_nonzero(out_deg > 2))
    row["n_merges"] = int(np.count_nonzero(in_deg >= 2))

    # Edge temporal gap and physical displacement.
    disp = np.array([])
    if n_edges > 0:
        ea = graph.edge_attrs(attr_keys=[SRC, TGT])
        src_j = na.rename({NID: SRC, "t": "t_s", "z": "z_s", "y": "y_s", "x": "x_s"})
        tgt_j = na.rename({NID: TGT, "t": "t_t", "z": "z_t", "y": "y_t", "x": "x_t"})
        e = ea.join(src_j, on=SRC, how="left").join(tgt_j, on=TGT, how="left")
        sz, sy, sx = scale
        e = e.with_columns(
            (pl.col("t_t") - pl.col("t_s")).alias("dt"),
            (
                ((pl.col("z_t") - pl.col("z_s")) * sz) ** 2
                + ((pl.col("y_t") - pl.col("y_s")) * sy) ** 2
                + ((pl.col("x_t") - pl.col("x_s")) * sx) ** 2
            ).sqrt().alias("disp_um"),
        )
        dt = e["dt"].to_numpy()
        row["edges_dt1"] = int(np.count_nonzero(dt == 1))
        row["edges_dt_gap"] = int(np.count_nonzero(dt > 1))
        disp = e["disp_um"].drop_nulls().to_numpy()
        d = _describe(disp)
        row["disp_um_p50"] = d["p50"]
        row["disp_um_p90"] = d["p90"]
        row["disp_um_max"] = d["max"]
    else:
        row["edges_dt1"] = 0
        row["edges_dt_gap"] = 0
        row["disp_um_p50"] = float("nan")
        row["disp_um_p90"] = float("nan")
        row["disp_um_max"] = float("nan")

    return row, disp


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.3g}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATASET_PATH,
                        help="Directory of {name}.zarr + {name}.geff (default: dataspec.DATASET_PATH).")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional path to write the per-dataset table.")
    parser.add_argument("--max-datasets", type=int, default=None, help="Limit number of datasets (for a quick look).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    names = [p.name for p in list_datasets(data_dir, require_geff=True)]
    if not names:
        print(f"No {{name}}.zarr + {{name}}.geff pairs found in {data_dir}")
        print("Point --data-dir at the competition train/ mount, or set $CELLMOT_DATA_DIR.")
        return
    if args.max_datasets:
        names = names[: args.max_datasets]

    print(f"Analysing {len(names)} dataset(s) in {data_dir}\n")

    rows: list[dict] = []
    all_disp: list[np.ndarray] = []
    for name in names:
        try:
            row, disp = analyse_dataset(data_dir, name)
        except Exception as exc:  # unreadable/partial geff
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        all_disp.append(disp)
        print(
            f"  {row['dataset']:<20} nodes={row['n_nodes']:>6} edges={row['n_edges']:>6} "
            f"T={_fmt(row['img_T'])} frames_annot={row['n_timepoints']} "
            f"div={row['n_divisions']} (rate={_fmt(row['division_rate'])}) "
            f"annot_frac={_fmt(row['annotated_frac'])} "
            f"disp_um p50/p90/max={_fmt(row['disp_um_p50'])}/{_fmt(row['disp_um_p90'])}/{_fmt(row['disp_um_max'])}"
        )

    if not rows:
        print("\nNo datasets analysed.")
        return

    table = pl.DataFrame(rows)

    # Aggregate.
    tot_nodes = int(table["n_nodes"].sum())
    tot_edges = int(table["n_edges"].sum())
    tot_div = int(table["n_divisions"].sum())
    tot_starts = int(table["n_track_starts"].sum())
    tot_ends = int(table["n_track_ends"].sum())
    tot_gap = int(table["edges_dt_gap"].sum())
    tot_outdeg_gt2 = int(table["n_outdeg_gt2"].sum())
    tot_merges = int(table["n_merges"].sum())
    disp_all = np.concatenate([d for d in all_disp if d.size]) if any(d.size for d in all_disp) else np.array([])
    d = _describe(disp_all)
    annot = table["annotated_frac"].drop_nulls().drop_nans()

    print("\n=== Summary ===")
    print(f"datasets                 : {len(rows)}")
    print(f"total nodes / edges      : {tot_nodes} / {tot_edges}")
    print(f"total divisions          : {tot_div}  (overall rate {tot_div / tot_nodes:.4g} per node)")
    print(f"track starts / ends      : {tot_starts} / {tot_ends}")
    print(f"edges with temporal gap  : {tot_gap} of {tot_edges} ({tot_gap / max(tot_edges, 1):.2%}) span dt>1")
    print(f"structural anomalies     : out-degree>2 nodes={tot_outdeg_gt2}, in-degree>=2 (merges)={tot_merges}")
    if annot.len() > 0:
        print(f"annotated fraction       : p50={annot.median():.3g} min={annot.min():.3g} max={annot.max():.3g} "
              f"(annotated nodes / estimated true nodes)")
    else:
        print("annotated fraction       : n/a (no estimated_number_of_nodes metadata)")
    print(f"edge displacement (µm)   : p50={d['p50']:.3g} p90={d['p90']:.3g} max={d['max']:.3g} "
          f"(node-match radius is 7 µm)")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(args.out_csv)
        print(f"\nWrote per-dataset table ({table.height} rows) to {args.out_csv}")


if __name__ == "__main__":
    main()
