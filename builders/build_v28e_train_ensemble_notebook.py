"""v28c = TRAIN-ONLY: loads divnet_ours/divnet_dataset.npz produced by v28b (attach the v28b notebook Output as Input)
and trains the mitosis classifier with the 4 frames as input CHANNELS (in_channels=4) -> fixes the 6-D tensor crash.
Output: /kaggle/working/divnet_ours/best_overall.pt {"model_state", "meta"} + meta.json.  ~10 min on T4.
"""
import json, sys
from pathlib import Path
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "biohub_v28e_divnet_train_ensemble.ipynb")

CELL = r'''
import os, json, time, random, hashlib, glob
import numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

TRAIN_EPOCHS, TRAIN_BATCH, VAL_FRACTION, SEED = 60, 32, 0.2, 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device:", device)

# ---- dataset from the v28b Output (attached as Input)
cands = sorted(glob.glob("/kaggle/input/notebooks/*/*/divnet_ours/divnet_dataset.npz") + glob.glob("/kaggle/input/*/divnet_ours/divnet_dataset.npz") + glob.glob("/kaggle/input/datasets/*/*/divnet_ours/divnet_dataset.npz"))
assert cands, "attach the v28b notebook Output (contains divnet_ours/divnet_dataset.npz)"
# v28e: merge EVERY attached dataset (v28b: 60 clips + GT crops, v28d: up to 80 new clips, candidates only)
_parts = []
for _p in cands:
    _d = np.load(_p, allow_pickle=False)
    _parts.append((_d["X"], _d["Y"].astype(np.float32), _d["STEMS"], _d["SOURCE"], _d["DC"]))
    print("loaded", _p, "| X", _d["X"].shape, "| positives", int(_d["Y"].sum()), "/", len(_d["Y"]), "| clips", len(set(_d["STEMS"].tolist())))
X = np.concatenate([p[0] for p in _parts]); Y = np.concatenate([p[1] for p in _parts]); STEMS = np.concatenate([p[2] for p in _parts])
SOURCE = np.concatenate([p[3] for p in _parts]); DC = np.concatenate([p[4] for p in _parts])
print("MERGED | X", X.shape, X.dtype, "| positives", int(Y.sum()), "/", len(Y), "| clips", len(set(STEMS.tolist())), "| datasets", len(_parts))
print("sources:", {s: int((SOURCE == s).sum()) for s in np.unique(SOURCE)}, "| candidate positives:", int(Y[SOURCE == "cand"].sum()))
assert X.ndim == 5 and X.shape[1] == 4, X.shape          # N, 4 frames, 16, 32, 32

# ---- model: DivNet architecture, but the 4 frames are input channels
class _DivNetConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_c, out_c, kernel_size=3, padding=1), torch.nn.BatchNorm3d(out_c), torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_c, out_c, kernel_size=3, padding=1), torch.nn.BatchNorm3d(out_c), torch.nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class DivNetMitosisClassifier(torch.nn.Module):
    def __init__(self, in_c=4):
        super().__init__()
        self.b1 = _DivNetConvBlock(in_c, 16); self.p1 = torch.nn.MaxPool3d((1, 2, 2))
        self.b2 = _DivNetConvBlock(16, 32); self.p2 = torch.nn.MaxPool3d((2, 2, 2))
        self.b3 = _DivNetConvBlock(32, 64); self.gap = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(inplace=True), torch.nn.Linear(32, 1))
    def forward(self, x):
        x = self.p1(self.b1(x)); x = self.p2(self.b2(x)); x = self.gap(self.b3(x)); return self.fc(torch.flatten(x, 1))

# ---- split by clip
stems_unique = sorted(set(STEMS.tolist()))
val_clips = {s for s in stems_unique if int(hashlib.md5(s.encode()).hexdigest(), 16) % 100 < VAL_FRACTION * 100}
is_val = np.array([s in val_clips for s in STEMS]); tr_idx = np.where(~is_val)[0]; va_idx = np.where(is_val)[0]
print(f"train clips {len(stems_unique) - len(val_clips)} / val clips {len(val_clips)} | train {len(tr_idx)} (pos {int(Y[tr_idx].sum())}) | val {len(va_idx)} (pos {int(Y[va_idx].sum())})")
assert Y[va_idx].sum() >= 5 and (1 - Y[va_idx]).sum() >= 5

Xt = torch.from_numpy(X.astype(np.float32)); Yt = torch.from_numpy(Y)     # N,4,16,32,32

def augment(xb):
    if random.random() < 0.5: xb = xb.flip(-1)
    if random.random() < 0.5: xb = xb.flip(-2)
    k = random.randint(0, 3)
    if k: xb = torch.rot90(xb, k, dims=(-2, -1))
    if random.random() < 0.5: xb = xb.flip(-3)
    return xb * (1.0 + 0.1 * (torch.rand(1, device=xb.device) - 0.5)) + 0.1 * (torch.rand(1, device=xb.device) - 0.5)

def run_training(seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = DivNetMitosisClassifier(4).to(device)
    pos_w = torch.tensor([(1 - Y[tr_idx]).sum() / max(1.0, Y[tr_idx].sum())], device=device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_EPOCHS)
    def predict(idx):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx), 128):
                out.append(torch.sigmoid(model(Xt[idx[i:i + 128]].to(device))).squeeze(1).cpu())
        return torch.cat(out).numpy()
    best = {"auc": -1.0}; t0 = time.time()
    for ep in range(1, TRAIN_EPOCHS + 1):
        model.train(); perm = np.random.permutation(tr_idx); tot = 0.0
        for i in range(0, len(perm), TRAIN_BATCH):
            b = perm[i:i + TRAIN_BATCH]; xb = augment(Xt[b].to(device)); yb = Yt[b].to(device)
            loss = crit(model(xb).squeeze(1), yb); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        sched.step()
        pv = predict(va_idx); auc = roc_auc_score(Y[va_idx], pv)
        p05 = pv >= 0.5; tp = int((p05 & (Y[va_idx] == 1)).sum()); fp = int((p05 & (Y[va_idx] == 0)).sum()); fn = int((~p05 & (Y[va_idx] == 1)).sum())
        if ep % 5 == 0 or ep == 1: print(f"  seed {seed} epoch {ep:2d} loss {tot / len(perm):.4f} | val AUC {auc:.4f} | @0.5 tp/fp/fn {tp}/{fp}/{fn} | {time.time() - t0:.0f}s", flush=True)
        if auc > best["auc"]:
            best = {"auc": auc, "epoch": ep, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "pv": pv}
    return best

results = [run_training(s) for s in (SEED, SEED + 1, SEED + 2)]      # 3 seeds
best = max(results, key=lambda r: r["auc"])
print("seed AUCs:", [round(r["auc"], 4) for r in results])
# v28e: ENSEMBLE = mean of the 3 seeds' best-epoch probabilities (same val split for every seed -> fair comparison)
pv_ens = np.mean([r["pv"] for r in results], axis=0)
auc_ens = roc_auc_score(Y[va_idx], pv_ens)
print(f"ENSEMBLE(3 seeds) val AUC {auc_ens:.4f}  vs best single {best['auc']:.4f}")
USE_ENSEMBLE = auc_ens >= best["auc"] - 0.002       # keep the ensemble unless it is clearly worse
if USE_ENSEMBLE:
    best = {"auc": auc_ens, "epoch": -1, "state": best["state"], "pv": pv_ens}
    print("-> shipping the 3-seed ensemble")
else:
    print("-> shipping the best single seed")

pv = best["pv"]; yv = Y[va_idx]; src_v = SOURCE[va_idx]; mask = src_v == "cand"
def f1_thr(p, y):
    bt, bf = 0.5, -1.0
    for thr in np.linspace(0.1, 0.9, 33):
        pr = p >= thr; tp = float((pr & (y == 1)).sum()); fp = float((pr & (y == 0)).sum()); fn = float((~pr & (y == 1)).sum())
        f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
        if f1 > bf: bf, bt = f1, float(thr)
    return bt, bf
thr_all, f1_all = f1_thr(pv, yv)
enough_cand = mask.sum() >= 10 and len(set(yv[mask].tolist())) == 2
thr_cand, f1_cand = f1_thr(pv[mask], yv[mask]) if enough_cand else (float("nan"), float("nan"))
ours_auc_cand = roc_auc_score(yv[mask], pv[mask]) if enough_cand else float("nan")
dc_v = DC[va_idx]; dcm = mask & np.isfinite(dc_v)
dc_auc = roc_auc_score(yv[dcm], dc_v[dcm]) if dcm.sum() >= 10 and len(set(yv[dcm].tolist())) == 2 else float("nan")
rec_thr = thr_cand if enough_cand and np.isfinite(thr_cand) else thr_all
print("=" * 78)
print(f"BEST epoch {best['epoch']}  val AUC(all) {best['auc']:.4f}")
print(f"on pipeline candidates only (n={int(mask.sum())}, pos={int(yv[mask].sum())}): OURS AUC {ours_auc_cand:.4f}  vs  DeepCenter score AUC {dc_auc:.4f}")
print(f"F1-optimal threshold: all-samples {thr_all:.2f} (F1 {f1_all:.3f}) | candidates {thr_cand:.2f} (F1 {f1_cand:.3f})")
print(f"-> recommended BIOHUB_DIV_MIN_PROB = {rec_thr:.2f}")
print("=" * 78)
out_dir = Path("/kaggle/working/divnet_ours"); out_dir.mkdir(parents=True, exist_ok=True)
meta = {"in_channels": 4, "val_auc_all": float(best["auc"]), "val_auc_cand": float(ours_auc_cand), "deepcenter_auc_cand": float(dc_auc),
        "best_epoch": int(best["epoch"]), "recommended_threshold": float(rec_thr), "threshold_all": thr_all, "threshold_cand": thr_cand,
        "n_samples": int(len(Y)), "n_pos": int(Y.sum()), "n_cand": int((SOURCE == "cand").sum()), "val_clips": sorted(val_clips),
        "input_spec": {"frames": 4, "z_pad": 8, "xy_pad": 16, "norm": "per-crop mean/std", "layout": "N,4,Z,Y,X (frames = channels)"}}
meta["ensemble"] = bool(USE_ENSEMBLE); meta["seed_aucs"] = [float(r["auc"]) for r in results]; meta["n_clips"] = int(len(set(STEMS.tolist())))
_ckpt = {"model_state": best["state"], "meta": meta}
if USE_ENSEMBLE:
    _ckpt["ensemble_states"] = [r["state"] for r in results]      # v25d averages sigmoid over these
torch.save(_ckpt, out_dir / "best_overall.pt")
(out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
print("saved", out_dir / "best_overall.pt", "| keys:", len(best["state"]), "| b1 in_channels:", tuple(best["state"]["b1.conv.0.weight"].shape), "| ensemble:", USE_ENSEMBLE)
'''
nb = {"cells": [
    {"cell_type": "markdown", "metadata": {}, "source": "# Biohub · v28e — train on v28b + v28d datasets, ship a 3-seed ensemble\n\nInput: the **v28b AND v28d notebook Outputs** (each has divnet_ours/divnet_dataset.npz). GPU. ~15 min. Output: divnet_ours/best_overall.pt with `ensemble_states` (v25d averages them)."},
    {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": CELL.strip("\n")}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(OUT, "w"), indent=1); print("wrote", OUT)
