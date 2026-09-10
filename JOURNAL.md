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
- **Built + unit-tested: `prune_nodes_by_support`** (`--min-track-nodes`, default 2)
  and the **results logger** (`--log-csv`, `--run-name`).
  - Node pruning rests on a *provable* point: only **edges** score, so an isolated
    node can never produce a TP yet still inflates `N_pred` and pays the node-count
    penalty. Dropping it is strictly score-improving — edge TP/FP/FN untouched,
    penalty shrinks. Default 2 drops isolated nodes only; larger values also discard
    short fragments, trading a possible TP edge for a smaller node count (sweep,
    don't assume). Applied *after* the ILP, which can leave nodes unlinked.
  - Logger appends config -> CV score per run so the Tier-1 sweep is comparable.
  - Both verified locally against the real source (stub graph; real CSV round-trip).

### Priority (competition-first, publication secondary)
Rank by (expected gain per GPU-hour) x P(works), not novelty.
- **Tier 1 (inference-only, ~0 GPU):** count calibration, edge pruning, node
  pruning, `--det-tta` (already upstream, free points), threshold / pool-kernel /
  ILP-weight sweeps.
- **Tier 2:** train to convergence (running).
- **Tier 3 (gated on the Phase-0 diagnosis, 1-2 runs):** resolution `1,2,2`
  (dense-cell separation — note `(1,4,4)` is already *exactly isotropic* at
  1.625 um, so this is about separating neighbours, not the 7 um radius);
  PU loss A/B.
- **Tier 4:** multi-seed/fold ensemble + TTA (boring, reliable, reserve budget).
- **Deferred to post-deadline (the paper):** candidate-level nnPU (voxel-level
  degenerates at pi~1e-5; at candidate level pi is O(0.1-0.9) and
  `estimated_number_of_nodes` *gives* us the prior), track-consistency
  self-training, controlled annotation-sparsity sweep.

### Parallel runs launched (matched A/B) — CORRECTED: parallel GPU does NOT work
- **WRONG (retracted):** I claimed ">=2 concurrent GPU kernels confirmed" off a single
  `RUNNING` status string. Run 2 then went to **ERROR**. Its output has
  `weights/pu/split_0/config.json` but **no checkpoint**, i.e. training started and
  died before one epoch — the `torch.cuda.synchronize()` no-GPU failure mode again.
  Best inference (log came back 0 bytes, so unconfirmed): **a second concurrent job
  gets a session with NO GPU** while run 1 holds the allocation. CPU-only cells (EDA,
  cv_splits) completed fine; training died on first CUDA touch.
  **Operational rule: run GPU jobs sequentially. Verify with artifacts, not status.**
- `notebookce6b79f8c0` = **BCE control** (clones upstream).
- `cellmot-pu-run2` = **PU arm** (clones the fork), `--method pu --detection-loss pu
  --pu-gate-power 2.0`, otherwise identical.
- **Confound avoided:** run 2's Cell 6 explicitly passes `--min-track-nodes 1
  --max-out-degree 0 --max-in-degree 0`. Without that it would clone the fork's new
  pruning defaults and differ from the control by *two* variables (loss + inference),
  making the A/B meaningless. Cells 7/8 stubbed (no test predict/submission needed).
- Both log to `results.csv`. Cost ~10 GPU-h of the 30/week.
- **Never push a new version to a kernel that is running** — it supersedes the job.
  Parallel runs require a separate kernel id (but see above: no second GPU).

---

## 2026-09-09 · DATA CHARACTERISATION (eda.csv, all 199 videos) — strategy-changing

Recovered from the failed run 2. Hard numbers at last:

| quantity | value |
|---|---|
| videos | 199 |
| annotated nodes | 133,318 total; **median 659/video** (50–1950) |
| annotated edges | 128,883 |
| frames/video | median 100 (40–100) |
| **divisions** | **151 in the ENTIRE dataset** (87 videos have >=1) |
| **gap edges (dt>1)** | **0** |
| annotation fraction | **median 3.6%** (0.13%–20%) |
| implied `N_true` | **median ~17,900/video**; ~4.7M total |
| edge displacement | p50 **1.82 um**, p90 3.63 um (max 60.8 outlier) |

### What this changes
1. **Divisions are negligible — stop considering them.** 151 events dataset-wide
   (~0.76/video). A 19-video val fold has ~14 division events, so `division_jaccard`
   is statistical noise, at 0.1 weight, under a newly-stricter local metric.
   **Optimise edge Jaccard alone.**
2. **Gap recovery is definitively dead.** Zero dt>1 edges in GT *and* the scorer
   discards them. Two independent confirmations.
3. **The node-count penalty is MILD — I over-billed count calibration.** `N_true` is
   ~17.9k/video while we predict far fewer, so the multiplier
   `1 - 0.1*(N_pred - N_true)/N_true` is ~**1.07 when under-predicting** and only
   drops below 1 past ~17.9k nodes. It's a **±7% effect**; edge Jaccard J is the
   dominant term. Targeting `N_true` exactly is *not* optimal — the real objective is
   `max_N J(N) * (1 - 0.1*(N - N_true)/N_true)`, and J's slope dominates.
   => **We can afford to detect far more aggressively than the 0.99 threshold.**
4. **Linking is probably NOT the bottleneck.** Median inter-frame displacement is
   **1.82 um ~= 1 voxel** (grid is 1.625 um isotropic) against a 7 um match radius.
   Association is geometrically easy; **detection recall is the bottleneck**, which is
   exactly what PU + a lower threshold attack.

---

## 2026-09-10 · Session 4 — LINKING SWEEP: 0.6016 -> 0.6561 (+9.1%)

### Root cause found (code + literature agree)
`train:63` / `predict:615` use `softmax(dim=0)`: each t+1 child spreads exactly one
unit of probability over parents at t, with **no null slot**. Every child is forced
to take a parent, so a child whose real parent was never detected dumps its mass on
a neighbour -> a fork. In-degree<=1 is free; **out-degree was never regulated**.
Trackastra (ECCV 2024, won CTC ISBI 2024) fixes exactly this with a "parental
softmax" `exp(a)/(1 + sum exp(a))` — the `+1` is the no-parent slot. It also masks
attention beyond `d_max`. Our candidate generation was **all-pairs with no distance
gate**, so the softmax normalised over nodes 100+ um away.

### Round-1 sweep (inference-only, existing checkpoint, 5 val videos)

| run | score | edge_J | node_recall |
|---|---|---|---|
| **nofork_gate12** | **0.6561** | 0.6649 | 0.9368 |
| dual_gate12 | 0.6523 | 0.6621 | 0.9394 |
| dual_null_gate12 | 0.6433 | 0.6490 | 0.9040 |
| dual | 0.6426 | 0.6492 | 0.9155 |
| gate8 | 0.6237 | 0.6355 | 0.9354 |
| prune_nodes | 0.6214 | 0.6281 | 0.9228 |
| gate12 | 0.6150 | 0.6258 | 0.9401 |
| **control** | **0.6016** | 0.6149 | 0.9466 |
| thr0.7_gate12 | 0.5790 | 0.5789 | 0.8792 |

- **control reproduced run 1 exactly** (0.6016/0.6149/0.6007/0.9466) — the
  "defaults are bit-identical" guarantee held, so the harness is trustworthy.
- **Best = distance gate (12 um) + NO forks** (`--max-children-per-node 1`):
  **+0.0545 (+9.1%)**, zero retraining. With 7 real divisions against 234 FPs,
  never predicting a division is close to optimal.
- Raising `--threshold` to 0.7 was clearly harmful (0.579), consistent with the
  break-even-precision analysis: we were **under**-linking, not over-linking.

### Operational lessons (cost us 3 failed runs)
1. **`machine_shape` is REQUIRED in hand-written kernel-metadata.json.** Omitting it
   gets an accelerator whose compute capability the installed torch has no kernels
   for -> `CUDA error: no kernel image is available for execution on the device`.
   This — not my PU code — is the likely cause of BOTH failed PU runs too.
   Always set `"machine_shape": "NvidiaTeslaT4"`.
2. **The Kaggle API returns a 0-byte log.** Always `subprocess.run(capture_output=True)`
   and write stderr to a file under `/kaggle/working`, or failures are invisible.
3. `check=False` makes a kernel report COMPLETE while producing nothing. Verify by
   artifacts (results.csv), never by status.
4. `kernel_sources` did NOT expose another kernel's checkpoint; vendoring the 8.4MB
   checkpoint into the repo works and is simpler.
5. `kaggle kernels output` is slow — don't wrap it in a short `timeout` or it
   truncates before reaching alphabetically-late files.
