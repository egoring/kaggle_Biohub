"""
Biohub Cell Tracking - Trackastra fine-tuning on the competition's sparse GT.

Idea
----
The pretrained Trackastra "ctc" linker is kept as is (architecture, features, window=4, quiet_softmax).
We re-train its weights on OUR detections (same DoG pipeline as inference) using the sparse .geff GT:

  * detections are matched to GT nodes per frame (bipartite, <= 7 um, exactly like the metric)
  * a row i (frame t) is supervised only if detection i is matched to a GT node whose GT successor(s)
    at t+1 are ALSO matched to detections  -> target row = one-hot on the matched successor(s), 0 elsewhere
  * a column j (frame t+1) is supervised only if j is matched to a GT node whose GT predecessor is matched
    -> target column = one-hot on the matched predecessor
  * every other pair is masked out of the loss (we simply do not know the answer for unlabeled cells)

This gives exact supervision from sparse labels: within a supervised row, *all* other next-frame detections
are true negatives, because a GT cell has exactly one successor (two for a division).

Everything below is importable both in the notebook (inlined) and locally (tests on synthetic data).
Requires the objects from pipeline.py in the same namespace (CFG, SCALE_ZYX, open_zarr_array, detect_volume,
make_masks, choose_radius, read_geff, score_official, micro_average, prune_divisions).
"""
from __future__ import annotations

import gc
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ feature cache (one npz per clip)
def extract_clip_cache(zarr_path: Path, geff_path: Path, cfg, out_npz: Path, n_workers: int = 0, log=print):
    """Run detection + Trackastra feature extraction exactly like inference, match to GT, save npz."""
    import torch
    from trackastra.data import get_features
    from trackastra.data.wrfeat import WRFeatures
    from trackastra.utils import normalize
    from scipy.optimize import linear_sum_assignment

    t0 = time.time()
    name = zarr_path.name[:-5]
    arr = open_zarr_array(zarr_path)
    cfg = dict(cfg)
    if cfg.get("adaptive_radius"):
        cfg["cell_radius_um"] = choose_radius(arr, cfg, log=log)
    if cfg.get("mask_scale"):
        r_um = cfg["mask_scale"] * cfg["cell_radius_um"]
        cfg["mask_radius_vox"] = tuple(int(max(1, round(r_um / s_))) for s_ in SCALE_ZYX)
    dets, imgs = detect_volume(arr, cfg, log=lambda *a: None)
    masks = make_masks(dets, arr.shape[1:], cfg["mask_radius_vox"])
    imgs = normalize(imgs)                       # identical to Trackastra._predict(normalize_imgs=True)
    feats = get_features(masks, imgs, features="wrfeat", ndim=3, n_workers=n_workers,
                         progbar_class=lambda x, **k: x)
    del masks, imgs
    F = WRFeatures.concat(feats)
    keys = list(F.features.keys()); dims = [int(v.shape[1]) for v in F.features.values()]

    # align feature rows with detections: regionprops sorts labels ascending, label = det index + 1
    det_zyx = np.zeros((len(F), 3), np.int32)
    for t in range(len(dets)):
        sel = np.where(F.timepoints == t)[0]
        labs = F.labels[sel]
        assert len(sel) == len(dets[t]) and np.array_equal(np.sort(labs), np.arange(1, len(dets[t]) + 1)), \
            f"{name}: label/detection mismatch at t={t}"
        det_zyx[sel] = dets[t][labs - 1]

    # GT matching per frame (same rule as the metric: optimal assignment, <= 7 um)
    gt = read_geff(geff_path)
    gt_id = np.full(len(F), -1, np.int64)
    gt_by_t = {}
    for gi, gt_, gz, gy, gx in zip(gt["ids"], gt["t"], gt["z"], gt["y"], gt["x"]):
        gt_by_t.setdefault(int(gt_), []).append((int(gi), gz, gy, gx))
    n_hit = 0
    for t, gl in gt_by_t.items():
        sel = np.where(F.timepoints == t)[0]
        if len(sel) == 0:
            continue
        P_ = det_zyx[sel].astype(np.float32) * SCALE_ZYX
        G_ = np.array([g[1:] for g in gl], np.float32) * SCALE_ZYX
        D = np.linalg.norm(P_[:, None, :] - G_[None, :, :], axis=2)
        r, c = linear_sum_assignment(D)
        for i, j in zip(r, c):
            if D[i, j] <= 7.0:
                gt_id[sel[i]] = gl[j][0]; n_hit += 1
    np.savez_compressed(
        out_npz, name=name, coords=F.coords.astype(np.float32), labels=F.labels.astype(np.int32),
        timepoints=F.timepoints.astype(np.int32), feats=F.features_stacked.astype(np.float32),
        det_zyx=det_zyx, gt_id=gt_id, gt_ids=np.asarray(gt["ids"], np.int64), gt_t=np.asarray(gt["t"], np.int32),
        gt_edges=np.asarray(gt["edges"], np.int64).reshape(-1, 2), est=np.int64(gt.get("estimated_number_of_nodes") or 0),
        radius=np.float32(cfg["cell_radius_um"]), keys=np.array(keys), dims=np.array(dims, np.int32),
        n_frames=np.int32(arr.shape[0]))
    log(f"    [{name}] {len(F)} detections in {arr.shape[0]} frames, r={cfg['cell_radius_um']} um, "
        f"GT nodes matched {n_hit}/{len(gt['ids'])}, GT edges {len(gt['edges'])} ({time.time()-t0:.0f}s)")
    del F, feats, dets
    gc.collect()
    return out_npz


# ------------------------------------------------------------------ clip cache -> supervision
class ClipCache:
    """Per-clip detections + features + GT-derived supervision targets."""

    def __init__(self, npz_path: Path):
        d = np.load(npz_path, allow_pickle=False)
        self.name = str(d["name"]); self.coords = d["coords"]; self.labels = d["labels"]
        self.timepoints = d["timepoints"]; self.feats = d["feats"]; self.det_zyx = d["det_zyx"]
        self.gt_id = d["gt_id"]; self.gt_edges = d["gt_edges"]; self.est = int(d["est"]) or None
        self.keys = [str(k) for k in d["keys"]]; self.dims = [int(x) for x in d["dims"]]
        self.n_frames = int(d["n_frames"]); self.gt_ids = d["gt_ids"]; self.gt_t = d["gt_t"]
        self.radius = float(d["radius"])
        # frame -> row slice (rows are sorted by time then label)
        order = np.lexsort((self.labels, self.timepoints))
        assert np.array_equal(order, np.arange(len(order))), "cache rows not sorted by (t,label)"
        self.frame_start = np.searchsorted(self.timepoints, np.arange(self.n_frames + 1))
        self._build_targets()

    def rows(self, t):
        return np.arange(self.frame_start[t], self.frame_start[t + 1])

    def _build_targets(self):
        gt2row = {int(g): i for i, g in enumerate(self.gt_id) if g >= 0}
        succ, pred = {}, {}
        for s_, t_ in self.gt_edges.tolist():
            succ.setdefault(s_, []).append(t_); pred.setdefault(t_, []).append(s_)
        n = len(self.gt_id)
        # row target: list of successor rows (all successors matched) or None
        self.row_succ = [None] * n
        self.col_pred = np.full(n, -1, np.int64)
        for i, g in enumerate(self.gt_id):
            g = int(g)
            if g < 0:
                continue
            ch = succ.get(g)
            if ch and all(c in gt2row for c in ch):
                rows_ = [gt2row[c] for c in ch]
                if all(self.timepoints[r] == self.timepoints[i] + 1 for r in rows_):
                    self.row_succ[i] = rows_
            pa = pred.get(g)
            if pa and len(pa) == 1 and pa[0] in gt2row and self.timepoints[gt2row[pa[0]]] == self.timepoints[i] - 1:
                self.col_pred[i] = gt2row[pa[0]]
        self.n_sup_rows = sum(1 for r in self.row_succ if r is not None)
        self.n_sup_cols = int((self.col_pred >= 0).sum())
        # frames where a window starting at t1 has supervision
        sup_t = set()
        for i, r in enumerate(self.row_succ):
            if r is not None:
                sup_t.add(int(self.timepoints[i]))          # pair (t, t+1)
        for j in np.where(self.col_pred >= 0)[0]:
            sup_t.add(int(self.timepoints[j]) - 1)
        self.sup_pairs_t = sorted(sup_t)

    def window_starts(self, W):
        starts = set()
        for t in self.sup_pairs_t:                          # pair (t,t+1) is inside window [t1, t1+W) if t1<=t<t1+W-1
            for t1 in range(max(0, t - W + 2), min(t, self.n_frames - W) + 1):
                starts.add(t1)
        return sorted(starts)

    def wrfeatures(self, rows):
        from trackastra.data.wrfeat import WRFeatures
        fd, o = OrderedDict(), 0
        for k, dd in zip(self.keys, self.dims):
            fd[k] = self.feats[rows, o:o + dd]; o += dd
        return WRFeatures(coords=self.coords[rows].copy(), labels=self.labels[rows], timepoints=self.timepoints[rows], features=fd)

    def frame_wrfeatures(self):
        return [self.wrfeatures(self.rows(t)) for t in range(self.n_frames)]

    def make_window(self, t1, W, augmenter=None, max_tokens=None):
        """Returns dict(coords (n,4) float, features (n,F), timepoints (n,), A (n,n), M (n,n)) as torch tensors."""
        import torch
        rows = np.arange(self.frame_start[t1], self.frame_start[t1 + W])
        if max_tokens and len(rows) > max_tokens:      # drop whole frames from the end (like trackastra)
            while len(rows) > max_tokens and W > 2:
                W -= 1; rows = np.arange(self.frame_start[t1], self.frame_start[t1 + W])
        n = len(rows)
        A = np.zeros((n, n), np.float32); M = np.zeros((n, n), np.float32)
        loc = {int(r): k for k, r in enumerate(rows)}
        for k, r in enumerate(rows):
            t = self.timepoints[r]
            if self.row_succ[r] is not None and t + 1 < t1 + W:
                nxt = self.rows(t + 1)
                M[k, loc[int(nxt[0])]:loc[int(nxt[-1])] + 1] = 1.0
                for s in self.row_succ[r]:
                    A[k, loc[s]] = 1.0
            p = self.col_pred[r]
            if p >= 0 and t - 1 >= t1:
                prv = self.rows(t - 1)
                M[loc[int(prv[0])]:loc[int(prv[-1])] + 1, k] = 1.0
                A[loc[int(p)], k] = 1.0
        feat = self.wrfeatures(rows)
        if augmenter is not None:
            feat = augmenter(feat)
        coords = np.concatenate((feat.timepoints[:, None].astype(np.float32), feat.coords.astype(np.float32)), axis=1)
        return dict(coords=torch.from_numpy(coords), features=torch.from_numpy(feat.features_stacked.astype(np.float32)),
                    timepoints=torch.from_numpy(feat.timepoints.astype(np.int64)),
                    A=torch.from_numpy(A), M=torch.from_numpy(M), n=n)


def make_augmenter(level=2):
    from trackastra.data import wrfeat
    if level <= 0:
        return None
    augs = [wrfeat.WRRandomFlip(p=0.5),
            wrfeat.WRRandomAffine(p=0.8, degrees=180, scale=(0.8, 1.25), shear=(0.1, 0.1))]
    if level >= 2:
        augs += [wrfeat.WRRandomBrightness(p=0.8), wrfeat.WRRandomOffset(p=0.8, offset=(-1, 1))]
    return wrfeat.WRAugmentationPipeline(augs)


# ------------------------------------------------------------------ loss (mirrors trackastra scripts/train.py, plus our mask)
def window_loss(transformer, w, device, causal_norm="quiet_softmax", div_upweight=3.0, eps=1e-6, use_amp=False):
    """Forward in fp16 autocast (if use_amp); the loss itself is always computed in fp32 with autocast disabled,
    because torch refuses binary_cross_entropy under CUDA autocast."""
    import torch
    import torch.nn.functional as Fnn
    from trackastra.utils import blockwise_causal_norm, blockwise_sum

    coords = w["coords"][None].to(device); feats = w["features"][None].to(device)
    tp = w["timepoints"].to(device); A = w["A"].to(device); M = w["M"].to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
        A_pred = transformer(coords, feats, padding_mask=None)[0]
    with torch.autocast(device.type, enabled=False):
        A_pred = A_pred.float().clamp(-1e4, 1e4)
        loss = Fnn.binary_cross_entropy_with_logits(A_pred, A, reduction="none")
        if causal_norm != "none":
            A_soft = blockwise_causal_norm(A_pred, tp, mode=causal_norm).clamp(eps, 1 - eps)
            loss = 0.01 * loss + Fnn.binary_cross_entropy(A_soft, A, reduction="none")
    with torch.no_grad():
        bs = A * (blockwise_sum(A, tp, dim=-1) + blockwise_sum(A, tp, dim=-2))
        weight = 1 + 1.0 * (bs == 2).float() + div_upweight * (bs > 2).float()
        # accuracy on supervised rows: argmax over the next-frame block hits a target
        dt = tp[None, :] - tp[:, None]
        row_sup = (A * (dt == 1)).sum(1) > 0            # rows with a known successor
        acc_n = acc_hit = 0
        if row_sup.any() and causal_norm != "none":
            sc = A_soft.masked_fill(dt != 1, -1.0)
            j = sc[row_sup].argmax(1)
            acc_hit = int(A[row_sup][torch.arange(int(row_sup.sum()), device=device), j].sum().item())
            acc_n = int(row_sup.sum().item())
    loss = (loss * weight * M).sum() / (M.sum() + eps)
    return loss, dict(acc_hit=acc_hit, acc_n=acc_n, n_sup=float(M.sum().item()))


# ------------------------------------------------------------------ training loop
def finetune(transformer, train_caches, val_caches, out_dir: Path, *, W=4, steps=6000, lr=3e-5, warmup=300,
             accum=8, augment=2, max_tokens=3000, time_budget_sec=4 * 3600, eval_every=1000, eval_windows=150,
             causal_norm="quiet_softmax", div_upweight=3.0, seed=0, log=print):
    """Fine-tune `transformer` in place. Saves pretrained.pt / best_val.pt / last.pt to out_dir. Returns history.
    (The final choice between them is made with the official-style CV, see select_and_save.)"""
    import torch
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    device = next(transformer.parameters()).device
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.AdamW(transformer.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    aug = make_augmenter(augment)
    tr_starts = [(c, c.window_starts(W)) for c in train_caches]
    tr_starts = [(c, s) for c, s in tr_starts if s]
    va_items = []
    for c in val_caches:
        for t1 in c.window_starts(W):
            va_items.append((c, t1))
    rng.shuffle(va_items); va_items = va_items[:eval_windows]
    log(f"train clips with supervision: {len(tr_starts)}/{len(train_caches)} | "
        f"supervised rows {sum(c.n_sup_rows for c in train_caches)} cols {sum(c.n_sup_cols for c in train_caches)} | "
        f"val windows {len(va_items)} | steps {steps} x accum {accum} | device {device}")

    def evaluate():
        transformer.eval(); tot = n = hit = nn_ = 0.0
        with torch.no_grad():
            for c, t1 in va_items:
                w = c.make_window(t1, W, None, max_tokens)
                l, st = window_loss(transformer, w, device, causal_norm, div_upweight, use_amp=use_amp)
                tot += float(l); n += 1; hit += st["acc_hit"]; nn_ += st["acc_n"]
        transformer.train()
        return tot / max(1, n), hit / max(1, nn_)

    hist = []
    v0 = evaluate() if va_items else (float("nan"), float("nan"))
    log(f"  step 0 (pretrained): val loss {v0[0]:.4f} | val link acc {v0[1]:.4f}")
    hist.append(dict(step=0, val_loss=v0[0], val_acc=v0[1], train_loss=None))
    best = float("inf")                       # first evaluated checkpoint always becomes best_val.pt
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(transformer.state_dict(), out_dir / "pretrained.pt")
    transformer.train(); t0 = time.time(); run_loss, run_n, run_hit, run_sup = 0.0, 0, 0, 0
    step = 0
    while step < steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            c, starts = tr_starts[rng.randint(len(tr_starts))]      # clip-uniform sampling (dense & sparse embryos equal)
            t1 = starts[rng.randint(len(starts))]
            w = c.make_window(t1, W, aug, max_tokens)
            if w["M"].sum() == 0:
                continue
            l, st = window_loss(transformer, w, device, causal_norm, div_upweight, use_amp=use_amp)
            if not torch.isfinite(l):
                log("    non-finite loss, skipping window"); continue
            scaler.scale(l / accum).backward()
            run_loss += float(l); run_n += 1; run_hit += st["acc_hit"]; run_sup += st["acc_n"]
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        if step % 100 == 0:
            log(f"  step {step}/{steps} lr {sched.get_last_lr()[0]:.2e} train loss {run_loss/max(1,run_n):.4f} "
                f"train link acc {run_hit/max(1,run_sup):.4f} ({time.time()-t0:.0f}s)")
        if step % eval_every == 0 or step == steps or time.time() - t0 > time_budget_sec:
            vl, va = evaluate() if va_items else (float("nan"), float("nan"))
            hist.append(dict(step=step, val_loss=vl, val_acc=va, train_loss=run_loss / max(1, run_n)))
            log(f"  == step {step}: val loss {vl:.4f} | val link acc {va:.4f} | train loss {run_loss/max(1,run_n):.4f}")
            run_loss, run_n, run_hit, run_sup = 0.0, 0, 0, 0
            torch.save(transformer.state_dict(), out_dir / "last.pt")
            if not va_items or vl < best:
                best = vl; torch.save(transformer.state_dict(), out_dir / "best_val.pt")
                log(f"     new best val loss -> saved {out_dir/'best_val.pt'}")
            if time.time() - t0 > time_budget_sec:
                log(f"  time budget reached at step {step}"); break
    (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
    return hist


def save_model_folder(transformer, state_dict, src_model_dir: Path, out_dir: Path, note: str = ""):
    """Write a folder loadable by Trackastra.from_folder (config.yaml + train_config.yaml + model.pt)."""
    import torch, yaml
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(transformer.config, open(out_dir / "config.yaml", "w"))
    ta = yaml.safe_load(open(Path(src_model_dir) / "train_config.yaml"))
    ta["finetuned_on"] = "biohub-cell-tracking-during-development " + note
    yaml.safe_dump(ta, open(out_dir / "train_config.yaml", "w"))
    torch.save(state_dict, out_dir / "model.pt")


def select_and_save(transformer, work_dir: Path, holdout_caches, cfg, gt_dir: Path, src_model_dir: Path, out_dir: Path, log=print):
    """Score pretrained / best_val / last on the hold-out clips with the official-style metric, keep the winner."""
    import torch
    results = {}
    if not (work_dir / "pretrained.pt").exists():          # training never started -> at least evaluate/save the start weights
        work_dir.mkdir(parents=True, exist_ok=True)
        torch.save(transformer.state_dict(), work_dir / "pretrained.pt")
    for tag in ("pretrained", "best_val", "last"):
        p = work_dir / f"{tag}.pt"
        if not p.exists():
            continue
        transformer.load_state_dict(torch.load(p, map_location=next(transformer.parameters()).device, weights_only=True))
        _, ma = cv_from_cache(holdout_caches, transformer, cfg, gt_dir, label=tag, log=log)
        results[tag] = ma["score"]
    winner = max(results, key=results.get)
    sd = torch.load(work_dir / f"{winner}.pt", map_location="cpu", weights_only=True)
    save_model_folder(transformer, sd, src_model_dir, out_dir, note=f"selected={winner} holdout_cv={results}")
    (out_dir / "SELECTED.txt").write_text(json.dumps(dict(selected=winner, holdout_cv=results), indent=1))
    log(f"hold-out CV scores: {results} -> selected '{winner}' -> {out_dir}")
    if winner == "pretrained":
        log("!! fine-tuning did NOT beat the pretrained model on the hold-out clips; ctc_ft = pretrained weights")
    return winner, results


# ------------------------------------------------------------------ CV from cache (no re-detection needed)
def link_from_cache(cache: ClipCache, transformer, cfg, batch_size=4, log=print):
    """Reproduce link_trackastra() from cached features: predict_windows -> greedy -> division pruning."""
    import torch
    from trackastra.data import build_windows
    from trackastra.model.predict import predict_windows
    from trackastra.tracking import build_graph, track_greedy

    feats = cache.frame_wrfeatures()
    windows = build_windows(feats, window_size=transformer.config["window"], progbar_class=lambda x, **k: x, as_torch=True)
    transformer.eval()
    with torch.no_grad():
        pred = predict_windows(windows=windows, features=feats, model=transformer, edge_threshold=0.05,
                               spatial_dim=3, batch_size=batch_size, progbar_class=lambda x, **k: x)
    G = build_graph(nodes=pred["nodes"], weights=pred["weights"], use_distance=False,
                    max_distance=cfg["trackastra_max_distance"], max_neighbors=10, delta_t=1)
    graph = track_greedy(G) if cfg.get("trackastra_mode", "greedy") == "greedy" else track_greedy(G, allow_divisions=False)
    key2id, nodes = {}, []
    for r in range(len(cache.labels)):
        t, lab = int(cache.timepoints[r]), int(cache.labels[r])
        key2id[(t, lab)] = r + 1
        z, y, x = cache.det_zyx[r]
        nodes.append((t, int(z), int(y), int(x), r + 1))
    edges, weights = [], []
    for u, v, d in graph.edges(data=True):
        nu, nv = graph.nodes[u], graph.nodes[v]
        ku, kv = (int(nu["time"]), int(nu["label"])), (int(nv["time"]), int(nv["label"]))
        if ku in key2id and kv in key2id:
            edges.append((key2id[ku], key2id[kv])); weights.append(float(d.get("weight", 1.0)))
    edges = prune_divisions(nodes, edges, weights, cfg, log=lambda *a: None)
    return nodes, edges


def cv_from_cache(caches, transformer, cfg, gt_dir: Path, label="", log=print):
    rows = []
    for c in caches:
        gt = read_geff(gt_dir / (c.name + ".geff"))
        nodes, edges = link_from_cache(c, transformer, cfg)
        sc = score_official(nodes, edges, gt); sc.update(config=label, sample=c.name)
        rows.append(sc)
        log(f"    [{label}] {c.name}: score={sc['score']:.3f} (tp={sc['edge_tp']} fp={sc['edge_fp']} fn={sc['edge_fn']}) "
            f"node_recall={sc['node_recall']:.3f} n_pred/est={sc['n_pred']}/{sc['est']}")
    ma = micro_average(rows); ma["config"] = label
    log(f"  ==> [{label}] micro score={ma['score']:.4f} adjJ={ma['adj_edge_J']:.4f} divJ={ma['div_J']:.3f}")
    return rows, ma
