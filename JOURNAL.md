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

---

## 2026-09-09 · Session 3 — Kaggle CLI hookup + first convergence run launched

- **Kaggle API hookup working.** Drive Kaggle from the local machine via the CLI.
  Auth uses the new-style `KGAT_` token through the `KAGGLE_API_TOKEN` env var
  (read from `~/.kaggle/kaggle.json` on each call, never printed) — the legacy
  `{username,key}` kaggle.json path doesn't accept the KGAT token. Fixed the token
  file (was UTF-16 with a bare token, now clean UTF-8 JSON). Kernel:
  `michaelangel23/notebookce6b79f8c0` (GPU single T4, internet on, competition data
  attached). Loop = edit notebook locally -> `kaggle kernels push` (runs headless,
  persists output) -> poll status -> pull log/outputs.
- **Notebook cells 5-8 fixed for real runs:** `--method baseline` (upstream default
  is `unet_transformer`, which mismatched the `WEIGHTS`/predict/submission paths),
  a `cd /kaggle/working/repo &&` prefix on every `!python` cell (cwd was drifting),
  and an absolute `WEIGHTS` path.
- **Launched baseline convergence run** (upstream, no PU): `--method baseline
  --epochs 6 --max-iters 4000 --batch-size 1 --num-workers 4` = 24k iters (~1.4
  passes), ~5 h on the T4. Full pipeline in one commit: setup -> data -> cv_split
  -> train -> val CV (Cell 6) -> test predict -> submission.csv. Status: RUNNING.
- **Calibration from the shakedown:** ~1.5 it/s; per-epoch eval = 282 s (more than a
  300-iter train epoch) -> favor fewer epochs, larger `--max-iters`. Session storage
  is ephemeral; only committed runs persist `/kaggle/working`.
- **Next:** pull the log for per-epoch `test_recall` (convergence signal) and the
  Cell 6 CV score (edge Jaccard / node_recall = the real number); confirm
  `submission.csv` wrote. Then decide resolution `1,2,2` A/B.
- **Built (untested): count-calibrated detection.** `predict_unet_transformer.py`
  gains `--target-nodes N` and `--target-nodes-from-geff`. Instead of relying on a
  global probability threshold, it keeps the top-K most confident peaks per frame
  with `K = N/T`, so the video's total node count lands near N. This attacks the
  score's `(1 - 0.1*(N_pred - N_true)/N_true)` penalty *directly* rather than hoping
  a threshold happens to land there. Costs one inference pass (no retraining), so it
  A/Bs against the existing checkpoint almost free — highest expected value per
  GPU-hour on the board.
  **Open gap:** validate on val videos first (their `.geff` carries
  `estimated_number_of_nodes`); **test videos have no `.geff`**, so a per-video
  estimator of `N_true` is still required — fit it from `eda.csv` (N_true vs image
  dims / T / frame count), which arrives with the current run's output.

---

## 2026-09-09 · Session 3b — fork live + UPSTREAM METRIC CHANGED (important)

- **Fork live.** `Caleb-Kelly-25/kaggle-cell-tracking-competition`, `main` = our work
  rebased onto upstream. Notebook Cell 1 `REPO_URL` now points at the fork, so Kaggle
  runs get PU + count calibration. (Needs a *fresh* Kaggle session to re-clone.)
- **Upstream moved and it matters.** Rebased onto `075fc5f` ("patch weakly connected
  component exploit"). Our local base was stale; a force-push would have run the OLD
  metric locally while the leaderboard used the new one. Rebase was clean (we touch
  `scripts/`, they touched `src/tracking_cellmot/metrics*` + tests).

### Metric changes — three exploits patched (rewrites parts of our plan)
1. **Consecutive-frame edges only.** `_evaluate_matched_graph` now filters to
   `t_target - t_source == 1`; backward and gap-spanning edges are DISCARDED.
   => Skip-frame "gap recovery" edges score nothing. Gap recovery is only worth
   anything if it **inserts a node** at the missing timepoint.
2. **Merge collapsing.** Several predicted nodes matching one GT node can no longer
   each claim the same GT edge (lowest edge id kept). => Duplicate/over-detection
   near a cell buys no TP and still pays the node-count penalty: purely harmful.
   Tighten NMS (`--pool-kernel-um`) and rely on count calibration.
3. **Out-degree capped at 2**, keeping the two **lowest edge ids** (arbitrary).
   => Emitting >2 outgoing edges risks losing the *best* edge to an arbitrary
   tiebreak. **Action: prune out-degree to <=2 ourselves, keeping the two highest
   -probability edges.** Cheap, free win.
4. **Division metric is now local** (`grandparent -> parent -> children ->
   grandchildren`), no graph-wide reachability; fork must be a matched parent-side
   node or its immediate successor, with two distinct daughter branches.
   => Divisions are strictly harder now and still only 0.1 weight: deprioritize further.

- **Unchanged:** edge Jaccard and the `(1 - 0.1*(N_pred - N_true)/N_true)` node
  penalty. Count calibration remains the highest-ROI item.
- **Built + unit-tested: `prune_edges`** (`--max-out-degree`, default 2;
  `--max-in-degree`, default 1). Drops non-consecutive edges and enforces the
  degree caps **by predicted probability**, so our best links survive instead of
  the scorer's arbitrary lowest-edge-id tiebreak. Skipped under `--use-ilp` (the
  solver enforces its own flow constraints). Verified locally against the real
  source with 4 cases (non-consecutive drop, out-degree top-2, in-degree top-1,
  caps disabled) — the only change so far actually *executed*, not just
  syntax-checked. Inference-only, so it stacks with count calibration at no
  training cost.
