# Project journal

A running log of changes and experiments on top of the `tracking-cellmot`
baseline for the Biohub cell-tracking competition. Newest entries at the bottom.

---

## 2026-08 · Session 1 — EDA, Kaggle pipeline, PU detection loss

### Tooling / infrastructure (no model change yet)
- **EDA.** Added `scripts/eda.py` (+ README note). Characterises the ground-truth
  graphs: nodes/edges/frames, annotation sparsity (via the GEFF
  `estimated_number_of_nodes`), division rate, per-edge displacement in µm vs the
  7 µm match radius, and temporal gaps (`dt>1`). Reads graphs + zarr metadata
  only — no volumes loaded.
- **Kaggle notebook** (`notebookce6b79f8c0.ipynb`) rebuilt into a runnable
  end-to-end pipeline: setup+install → detect data → EDA → CV split → train →
  local validation → predict test → `submission.csv`.
- **Fixes made along the way:**
  - numpy/scipy install conflict → pin both to Kaggle's preinstalled versions via
    a constraints file so pip can't corrupt numpy under the running kernel.
  - `$TEST` was never exported → export TRAIN/TEST/CELLMOT_DATA_DIR for the
    `!python` cells.
  - `tracking_cellmot` not importable in-kernel (`pip install -e .` registers its
    hook only at interpreter startup) → put `repo/src` on `sys.path` + `PYTHONPATH`.
  - Notebook metadata: Internet + GPU enabled.
- **Validation harness.** `cv_splits.json` reproduces the trainer's seed-0 90/10
  split so training and `--evaluate` score the *same* held-out videos. Tune
  against the local CV number, not the leaderboard.

### Model change #1 — PU-aware detection loss (flag-gated)
- **What:** `scripts/train_unet_transformer.py::compute_detection_loss` now
  supports `loss_type="pu"`. Non-GT voxels are treated as *unlabeled*; their
  negative penalty is scaled by `(1 - p) ** pu_gate_power` (p = detached detection
  probability), so voxels that already look like cells — likely unannotated GT —
  are barely penalised. Reduces exactly to baseline BCE at `pu_gate_power=0`.
- **Why:** sparse GT makes detection a positive-unlabeled problem; the baseline
  penalises unannotated cells as background, suppressing `node_recall` (hence the
  0.99 threshold). Higher recall lifts edge TP, the dominant score term.
- **New flags:** `--detection-loss {bce,pu}` (default `bce`), `--pu-gate-power`
  (default 2.0). Training-time only — prediction/inference unchanged.
- **Status:** implemented + syntax-checked. NOT yet trained/validated (no local
  data or GPU in the dev environment).

### Open next steps
- Run the PU A/B on Kaggle: train `bce` vs `pu` (separate `--method` dirs),
  compare `node_recall` and final score on the CV fold; record numbers here.
- If PU lifts recall, re-sweep `--det-threshold` under `pu` (expect the optimum
  to drop below 0.99).
- Later levers (not started): higher-resolution detection (reduce `--downsample`),
  ILP linking + gap recovery, ensembling/TTA, richer 3D augmentations.

---

## 2026-08 · Session 2 — serious-model program kickoff

- **Delivery path.** Committed the session-1 work (PU loss, `eda.py`, README,
  this journal, notebook) to branch `serious-model-pu` so it can be pushed to a
  fork. Kaggle clones a repo, and our changes aren't on upstream royerlab —
  proof: a Kaggle run logged the *old* `Detection loss: weight=..., neg_weight=...`
  line, not our `type=..., pu_gate_power=...`. Notebook Cell 1 now takes a
  `REPO_URL` (default upstream) to switch to the fork.
- **Budget:** 30 free GPU-hours/week. Plan: cheap bounded A/Bs to pick direction
  (PU vs BCE, resolution), then spend the bulk on the winning config's
  convergence + a multi-fold ensemble.
- **Known gaps to build:** resumable checkpointing (training saves best-per-epoch
  but no optimizer/epoch state → can't resume across Kaggle session timeouts);
  a `results.csv` experiment logger (config → CV score) so stacked runs are
  comparable. The trainer's own "best" uses an internal proxy
  (`test_acc * test_recall`), not the competition metric — real CV score comes
  from `predict --evaluate`.
- **Program order:** (1) train to convergence = real baseline; (2) detection
  resolution `1,2,2` vs `1,4,4`; (3) lock detection recipe (threshold/pool/aug);
  (4) linking (ILP + gap recovery); (5) 5-fold ensemble + TTA. Divisions last.
