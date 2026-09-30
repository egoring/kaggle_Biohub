"""
Biohub — train the organisers' official baseline (TemporalUNet3D detector + SimpleNodeTransformer linker,
repo royerlab/kaggle-cell-tracking-competition) on Kaggle, with the things the original script lacks:

  * full-model resume (model + optimizer + epoch counter) so a 12 h Kaggle session can continue the previous one
  * a wall-clock budget: stop cleanly and save before Kaggle kills the kernel
  * last.pth + best.pth every epoch, written in the repo's own layout (weights/<method>/split_0/edge_predictor_best.pth
    + config.json) so the repo's predict script — and our own inference notebook — can load them
  * a fixed validation split = the same 12 hold-out clips used by all our earlier CVs
  * fallbacks so the code also runs on CPU / on data without precomputed quantiles (local tests on synthetic data)

Everything model/loss/data related is imported unchanged from the repo (scripts/train_unet_transformer.py).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_T = None  # the repo's train module, set by setup_repo()


# ------------------------------------------------------------------ repo import + small compatibility patches
def setup_repo(repo_dir: Path):
    """Put the repo on sys.path, import its train module, patch CPU/quantile fallbacks. Returns the module."""
    global _T
    import torch
    repo_dir = Path(repo_dir)
    for p in (repo_dir / "src", repo_dir / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import tracking_cellmot.io as cio
    import train_unet_transformer as T

    _orig_open = cio.open_dataset

    def open_dataset_with_quantiles(ds_path, *a, **k):
        ds = _orig_open(ds_path, *a, **k)
        if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
            import zarr
            arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
            T_ = arr.shape[0]
            idx = np.unique(np.linspace(0, T_ - 1, min(5, T_)).round().astype(int))
            sample = np.concatenate([np.asarray(arr[int(t)]).ravel()[::13] for t in idx]).astype(np.float32)
            lo, hi = np.quantile(sample, [0.001, 0.999])
            ds.quantiles = dict(ds.quantiles); ds.quantiles.update({"0.001": float(lo), "0.999": float(hi)})
        return ds

    cio.open_dataset = open_dataset_with_quantiles
    T.open_dataset = open_dataset_with_quantiles          # the script imported the name directly
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None       # train_epoch() calls it unconditionally
    from tqdm import tqdm as _tqdm
    T.tqdm = lambda *a, **k: _tqdm(*a, **{**k, "disable": True})   # the script forces disable=False -> silence per-iteration bars

    # Cap the number of detections per frame fed to the edge transformer. Early in training the detector fires
    # almost everywhere (thousands of peaks/frame) and the cross-attention matrix (n_src x n_tgt) blows the GPU
    # (Kaggle T4: "Tried to allocate 6 GiB" in softmax). Keeping only the top-K peaks by logit is safe: real cells
    # are < 700/frame and GT nodes are the strongest peaks. Implemented by lowering every voxel below the K-th peak
    # logit to (threshold - 1) on a clone, so the original function's peak logic is untouched; the detection loss
    # still uses the unmodified logits.
    import torch.nn.functional as F
    _orig_dm = T.detect_and_match

    def detect_and_match_capped(det_logits, gt_coords, mask, image_shape, det_threshold=0.3, pool_kernel_um=5.0,
                                max_match_distance=5.0, voxel_size=None, frame_index=0, window_size=None):
        cap = CAP["max_det_per_frame"]
        if cap:
            with torch.no_grad():
                if voxel_size is not None:
                    kern = tuple(max(1, k if k % 2 == 1 else k + 1) for k in (max(1, round(pool_kernel_um / s)) for s in voxel_size))
                else:
                    k = max(1, round(pool_kernel_um)); kern = (k if k % 2 == 1 else k + 1,) * 3
                pooled = F.max_pool3d(det_logits, kern, stride=1, padding=tuple(k // 2 for k in kern))
                is_peak = (det_logits == pooled) & (det_logits > det_threshold)
                counts = is_peak.flatten(1).sum(1)
                if bool((counts > cap).any()):
                    det_logits = det_logits.clone()
                    for b in torch.nonzero(counts > cap).flatten().tolist():
                        thr_b = torch.topk(det_logits[b][is_peak[b]], cap).values.min()
                        det_logits[b] = torch.where(det_logits[b] >= thr_b, det_logits[b],
                                                    torch.full_like(det_logits[b], det_threshold - 1.0))
                    CAP["n_capped"] += 1
        return _orig_dm(det_logits, gt_coords, mask, image_shape, det_threshold, pool_kernel_um, max_match_distance,
                        voxel_size, frame_index, window_size)

    T.detect_and_match = detect_and_match_capped
    _T = T
    return T


CAP = {"max_det_per_frame": 1200, "n_capped": 0}   # set CAP["max_det_per_frame"] = None to disable


# ------------------------------------------------------------------ splits (identical rule to our CV / v11 hold-out)
def make_splits(train_dir: Path, n_holdout: int = 12) -> dict:
    stems = sorted(p.name[:-5] for p in Path(train_dir).iterdir()
                   if p.name.endswith(".zarr") and (Path(train_dir) / (p.name[:-5] + ".geff")).exists())
    step = max(1, len(stems) // max(1, n_holdout))
    hold = stems[1::step][:n_holdout]
    hs = set(hold)
    return {"split": 0, "train": [s for s in stems if s not in hs], "test": hold}


# ------------------------------------------------------------------ data
def build_loaders(data_dir: Path, train_stems, test_stems, *, downsample=(1, 4, 4), window_size=2, batch_size=16,
                  num_workers=4, max_frames=None, augment=True, log=print):
    import torch
    from torch.utils.data import DataLoader
    T = _T

    def _load(stems, desc):
        data = []
        t0 = time.time()
        for i, s in enumerate(stems):
            vm, windows = T.load_dataset_windows(Path(data_dir) / s, window_size=window_size,
                                                 max_frames=max_frames, downsample=downsample)
            data.append((vm, windows))
            if (i + 1) % 25 == 0 or i + 1 == len(stems):
                log(f"    {desc}: {i+1}/{len(stems)} clips, {sum(len(w) for _, w in data)} windows ({time.time()-t0:.0f}s)")
        return data

    tr = _load(train_stems, "train"); te = _load(test_stems, "val")
    all_windows = [w for _, ws in tr + te for w in ws]
    max_nodes = max(max(w.node_counts) for w in all_windows)
    log(f"    max GT nodes per frame = {max_nodes}")
    train_ds = T.FrameWindowDataset(tr, max_nodes=max_nodes, augmentations=T.DEFAULT_AUGMENTATIONS if augment else None)
    val_ds = T.FrameWindowDataset(te, max_nodes=max_nodes)
    kw = dict(num_workers=num_workers, prefetch_factor=2 if num_workers > 0 else None,
              persistent_workers=num_workers > 0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, max_nodes


# ------------------------------------------------------------------ model / checkpoints
def build_model(cfg: dict, device, data_parallel=True, log=print):
    import torch
    import torch.nn as nn
    T = _T
    unet = T.TemporalUNet3D(in_channels=1, out_channels=cfg["unet_out_channels"], layers=cfg["unet_layers"])
    model = T.UNetNodeTransformer(unet=unet, unet_out_channels=cfg["unet_out_channels"],
                                  pos_feat_dim=4 * T._POS_EMBED_DIM).to(device)
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    if data_parallel and n_gpu > 1:
        model.unet = nn.DataParallel(model.unet)
        log(f"    DataParallel: UNet across {n_gpu} GPUs")
    log(f"    parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def _plain_state(model):
    """state_dict with the DataParallel prefix removed (the repo's checkpoint format)."""
    return {k.replace("unet.module.", "unet.", 1): v.detach().cpu() for k, v in model.state_dict().items()}


def _load_plain_state(model, state):
    import torch.nn as nn
    if isinstance(model.unet, nn.DataParallel):
        state = {(k.replace("unet.", "unet.module.", 1) if k.startswith("unet.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def find_resume(dirs, log=print):
    """Newest last.pth among candidate weight dirs (attached outputs of a previous run)."""
    cands = []
    for d in dirs:
        for p in Path(d).rglob("last.pth"):
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime)
    log(f"    resume checkpoint: {cands[-1]}")
    return cands[-1]


# ------------------------------------------------------------------ training loop with budget + resume
def train_loop(model, train_loader, val_loader, *, out_dir: Path, cfg: dict, epochs: int, lr: float = 1e-4,
               det_loss_weight=1.0, det_neg_weight=1e-2, time_budget_sec=10 * 3600, resume: Path | None = None,
               max_iters=None, lr_min_frac=0.1, log=print):
    """Trains until `epochs` total (counting resumed epochs) or the time budget is exhausted.
    Saves last.pth (full resume state), edge_predictor_best.pth / edge_predictor_last.pth (repo format), config.json."""
    import torch
    T = _T
    device = next(model.parameters()).device
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {k: cfg[k] for k in ("unet_out_channels", "unet_layers", "downsample", "window_size", "pool_kernel_um")}, indent=2))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    start_epoch, best, hist, best_model_state = 0, 0.0, [], None
    if resume is not None and Path(resume).exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        mi, un = _load_plain_state(model, ck["model"])
        try:
            opt.load_state_dict(ck["optimizer"])
        except Exception as e:  # noqa
            log(f"    optimizer state not restored ({type(e).__name__}) - fresh optimizer")
        start_epoch, best, hist = int(ck["epoch"]), float(ck.get("best", 0.0)), list(ck.get("history", []))
        best_model_state = ck.get("best_model")
        if best_model_state is not None:
            torch.save(best_model_state, out_dir / "edge_predictor_best.pth")   # carry the best weights into this run's output
        torch.save(ck, out_dir / "last.pth"); torch.save(ck["model"], out_dir / "edge_predictor_last.pth")   # output is complete even if no epoch fits
        (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
        log(f"    resumed at epoch {start_epoch} (best acc*recall so far {best:.4f}; missing={len(mi)} unexpected={len(un)})")

    def lr_at(ep):  # cosine over the planned total, floor at lr_min_frac
        return lr * (lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, ep / max(1, epochs)))))

    t_start = time.time(); ep_times = []
    for ep in range(start_epoch, epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep)
        t0 = time.time()
        edge_loss, det_loss = T.train_epoch(model, train_loader, opt, device, det_loss_weight, det_neg_weight,
                                            max_iters=max_iters, pool_kernel_um=cfg["pool_kernel_um"])
        t_train = time.time() - t0; t0 = time.time()
        with torch.no_grad():
            val_loss, val_acc, val_recall = T.evaluate(model, val_loader, device, pool_kernel_um=cfg["pool_kernel_um"])
        t_val = time.time() - t0
        score = val_acc * val_recall
        is_best = score >= best
        state = _plain_state(model)
        if is_best:
            best = score; best_model_state = state
            torch.save(state, out_dir / "edge_predictor_best.pth")
        torch.save(state, out_dir / "edge_predictor_last.pth")
        row = dict(epoch=ep + 1, lr=lr_at(ep), edge_loss=edge_loss, det_loss=det_loss, val_loss=val_loss,
                   val_acc=val_acc, val_recall=val_recall, score=score, best=best, t_train=t_train, t_val=t_val)
        hist.append(row)
        torch.save(dict(model=state, optimizer=opt.state_dict(), epoch=ep + 1, best=best, history=hist,
                        best_model=best_model_state, cfg=cfg), out_dir / "last.pth")
        (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
        log(f"  epoch {ep+1:3d}/{epochs} lr {lr_at(ep):.1e} | edge {edge_loss:.4f} det {det_loss:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} recall {val_recall:.4f} -> {score:.4f} {'*' if is_best else ''} "
            f"(best {best:.4f}) | train {t_train:.0f}s val {t_val:.0f}s | det-cap hits {CAP['n_capped']}")
        CAP["n_capped"] = 0
        ep_times.append(t_train + t_val)
        elapsed = time.time() - t_start
        if ep + 1 < epochs and elapsed + 1.15 * np.mean(ep_times) > time_budget_sec:
            log(f"  time budget: {elapsed/3600:.2f} h used, next epoch (~{np.mean(ep_times)/60:.0f} min) would not fit -> stopping at epoch {ep+1}")
            break
    return hist, best
