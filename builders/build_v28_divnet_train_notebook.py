"""v28 = train OUR OWN mitosis classifier on the pipeline's real division candidates.

Base: public harmonic-fusion notebook (same cells as v24) run in the Q1 configuration (det 0.96, edge/DeepCenter
TTA, tight 5.5) with the hold-out validator enabled on MANY train clips (VALIDATOR_N_PER_TYPE, default 30 per embryo
type = 60 clips).  The validator predicts those train clips with the real detector, and the post-process runs the real
safe-division candidate generator on them.  We hook that generator: every geometric candidate (parent, existing
child, candidate sister) is dumped with its DeepCenter score.  Then:
  * label each dumped candidate against the sparse GT (7 um bipartite matching, same code as the validator)
        parent matched to a GT node with 2 GT children and the candidate matches one of them -> 1
        parent matched to a GT node with 1 GT child                                          -> 0
        candidate matches neither child of a 2-child GT parent                                -> 0
        parent unmatched / GT parent has no children (track end)                              -> unknown, dropped
  * add extra positives at GT division parents and a capped number of extra negatives at 1-child GT nodes
  * crop 4 frames x 16 x 32 x 32 around the parent (exactly DivNet's input spec, so the checkpoint drops into the
    v25 gate unchanged) and train the DivNet architecture from scratch, split by clip, pick best val AUC
  * report val AUC of DeepCenter's own score on the same candidates for comparison, and the F1-optimal threshold
Output: /kaggle/working/divnet_ours/best_overall.pt  {"model_state": ..., "meta": {...}}  + meta.json + dataset npz
"""
import json, sys
from pathlib import Path

SRC = Path("public_notebooks/7c6cee3f-biohub-harmonic-fusion.ipynb")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "biohub_v28_divnet_train.ipynb")

nb = json.load(open(SRC))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for c in cells:
    c["outputs"] = []; c["execution_count"] = None; c["source"] = "".join(c["source"])

CONFIG_CELL = r'''
# ---------------------------------------------------------------- OURS: v28 configuration
os.environ["BIOHUB_DET_THRESHOLD"] = "0.96"                 # Q1 detector setting
os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"         # Q1 post-process setting
os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"
os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "30"   # FORCE (public config cell sets 4 before us): 30 per embryo type = 60 train clips
GT_CROPS_ALL_TRAIN = True                                # GT-based positives/negatives from ALL train clips (no prediction needed)
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"         # veto stays ON for the base scoring; the dump happens BEFORE the veto
DIVNET_DUMP: list = []                                      # filled by the hook in the post-process cell
EXTRA_NEG_PER_POS = 3                                       # extra GT 1-child negatives per positive
TRAIN_EPOCHS = 40
TRAIN_BATCH = 32
VAL_FRACTION = 0.2
SEED = 1234
print("v28: validator clips per type =", os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"])
'''

# --- guard raise -> print
g = cells[1]["source"]
old = 'if _drift:\n    raise RuntimeError(\n        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)\n    )'
assert g.count(old) == 1; cells[1]["source"] = g.replace(old, 'if _drift:\n    print("NOTE (ours):", _guard_json.dumps(_drift, sort_keys=True))')

# --- TTA env -> setdefault (unchanged behaviour, kept for parity with v24)
p = cells[4]["source"]
for a, b in {
    "os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'": "os.environ.setdefault('BIOHUB_EDGE_FEATURE_TTA', '1')",
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "1")',
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "0.75")',
}.items():
    assert p.count(a) == 1, a; p = p.replace(a, b)
cells[4]["source"] = p

# --- hook in the safe-division candidate loop (post-process cell = cells[5])
q = cells[5]["source"]
anchor = '''                stats["safe_division_geometric_candidates"] += 1
                if DEEPCENTER_SAFE_DIV_VETO and not deepcenter_accept_repair_point('''
assert q.count(anchor) == 1, "anchor not unique"
hook = '''                # ---- OURS (v28): dump every geometric candidate BEFORE the DeepCenter veto, with DeepCenter's score
                if globals().get("DIVNET_DUMP") is not None:
                    _dc_score = deepcenter_score_point(dataset, int(candidate["t"]), node_point(candidate),
                                                       deepcenter_bundle, frame_cache, deepcenter_cache)
                    DIVNET_DUMP.append({
                        "dataset": dataset, "t": int(source["t"]),
                        "parent": node_point(source), "child": node_point(existing_child), "cand": node_point(candidate),
                        "parent_dist": float(parent_dist), "sister_dist": float(sister_dist), "child_dist": float(child_dist),
                        "dc": None if _dc_score is None else float(_dc_score),
                    })
'''
cells[5]["source"] = q.replace(anchor, hook + anchor)

LABEL_CELL = r'''
# ================================================================ OURS (v28): label dumped candidates against GT, build crops
import numpy as np, json, random, hashlib, time
from collections import defaultdict
random.seed(SEED); np.random.seed(SEED)
assert VALIDATOR_ENABLE and val_stems, "validator must be on"
print("dumped geometric candidates:", len(DIVNET_DUMP), "over", len({d["dataset"] for d in DIVNET_DUMP}), "clips")

Z_PAD, XY_PAD, N_FRAMES = 8, 16, 4          # == DivNet input spec used by divnet_score_division in v25

def _zarr_T(stem):
    meta = json.loads((TRAIN_DIR / f"{stem}.zarr" / "0" / "zarr.json").read_text()); return int(meta["shape"][0])

def make_crop(stem, t, point, frame_cache, T):
    cz, cy, cx = (int(round(float(v))) for v in point)
    frames = [read_test_frame(stem, min(max(0, t + dt), T - 1), frame_cache) for dt in range(N_FRAMES)]
    vol = np.stack(frames, axis=0)
    Z, Y, X = vol.shape[1:]
    z0, z1 = max(0, cz - Z_PAD), min(Z, cz + Z_PAD); y0, y1 = max(0, cy - XY_PAD), min(Y, cy + XY_PAD); x0, x1 = max(0, cx - XY_PAD), min(X, cx + XY_PAD)
    padded = np.zeros((N_FRAMES, 2 * Z_PAD, 2 * XY_PAD, 2 * XY_PAD), dtype=np.float32)
    dz0, dy0, dx0 = Z_PAD - (cz - z0), XY_PAD - (cy - y0), XY_PAD - (cx - x0)
    padded[:, dz0:dz0 + (z1 - z0), dy0:dy0 + (y1 - y0), dx0:dx0 + (x1 - x0)] = vol[:, z0:z1, y0:y1, x0:x1]
    padded = (padded - float(padded.mean())) / (float(padded.std()) + 1e-6)
    return padded.astype(np.float16)

samples = []      # dicts: x (crop), y (label), stem, source ('cand'|'gt_pos'|'gt_neg'), dc (DeepCenter score or nan)
label_stats = defaultdict(int)
_real_test_dir = TEST_DIR
globals()["TEST_DIR"] = TRAIN_DIR      # read_test_frame reads from TRAIN_DIR while we crop
try:
    by_stem = defaultdict(list)
    for d in DIVNET_DUMP:
        by_stem[d["dataset"]].append(d)
    for stem in val_stems:
        gt_nodes_plain, gt_edges_plain, _ = VAL_GT[stem]
        gt_out = defaultdict(set)
        for s_, t_ in gt_edges_plain:
            gt_out[s_].add(t_)
        # pseudo pred nodes = every dumped point (dedup by (t, z, y, x)); match them to GT with the validator's matcher
        pts = {}
        def key(t, p): return (int(t), round(p[0], 2), round(p[1], 2), round(p[2], 2))
        for d in by_stem.get(stem, []):
            for name, tt in (("parent", d["t"]), ("child", d["t"] + 1), ("cand", d["t"] + 1)):
                k = key(tt, d[name])
                if k not in pts:
                    pts[k] = (len(pts) + 1, (int(tt), float(d[name][0]), float(d[name][1]), float(d[name][2])))
        pred_nodes_plain = {pid: node for pid, node in pts.values()}
        pred_to_gt, gt_to_pred = match_nodes_bipartite(pred_nodes_plain, gt_nodes_plain, VALIDATOR_MATCH_RADIUS_UM)
        pid_of = {k: v[0] for k, v in pts.items()}
        T = _zarr_T(stem); frame_cache = {}
        n_pos_stem = 0
        gt_in = {t_: s_ for s_, t_ in gt_edges_plain}          # GT child -> GT parent
        for d in by_stem.get(stem, []):
            gp = pred_to_gt.get(pid_of[key(d["t"], d["parent"])])
            gc = pred_to_gt.get(pid_of[key(d["t"] + 1, d["cand"])])
            if gp is None:
                # parent not in the sparse GT.  If the candidate IS a GT node whose GT parent exists, then linking our parent
                # to it is a wrong edge (the metric counts it as FP) -> certain negative.  Otherwise unknown.
                if gc is not None and gc in gt_in:
                    y = 0; label_stats["neg_cand_other_parent"] += 1
                    samples.append({"x": make_crop(stem, d["t"], d["parent"], frame_cache, T), "y": 0, "stem": stem, "source": "cand",
                                    "dc": np.nan if d["dc"] is None else d["dc"]})
                else:
                    label_stats["unknown_parent_unmatched"] += 1
                continue
            children = gt_out.get(gp, set())
            if len(children) >= 2:
                y = 1 if (gc is not None and gc in children) else 0
                label_stats["pos_cand" if y else "neg_wrong_partner"] += 1
            elif len(children) == 1:
                y = 0; label_stats["neg_single_child"] += 1
            else:
                label_stats["unknown_track_end"] += 1; continue
            samples.append({"x": make_crop(stem, d["t"], d["parent"], frame_cache, T), "y": y, "stem": stem, "source": "cand",
                            "dc": np.nan if d["dc"] is None else d["dc"]})
            n_pos_stem += y
        frame_cache.clear()
    print("candidate-labelled samples:", len(samples), "| positives:", sum(s_["y"] for s_ in samples))

    # GT-based positives (division parents) and negatives (1-child nodes) -- from ALL train clips, no prediction needed
    gt_stems = sorted(p_.name[:-5] for p_ in TRAIN_DIR.iterdir() if p_.name.endswith(".zarr")) if GT_CROPS_ALL_TRAIN else list(val_stems)
    gt_stems = [s_ for s_ in gt_stems if s_ not in set(test_stems)]
    t_gt0 = time.time()
    for k_, stem in enumerate(gt_stems):
        gt_path = TRAIN_DIR / f"{stem}.geff"
        if not gt_path.exists():
            continue
        try:
            gt_nodes_plain, gt_edges_plain = graph_to_plain(graph_from_geff(gt_path))
        except Exception as exc:
            print("  GT read failed", stem, exc); continue
        gt_out = defaultdict(set)
        for s_, t_ in gt_edges_plain:
            gt_out[s_].add(t_)
        T = _zarr_T(stem); frame_cache = {}
        gt_pos = [s_ for s_, ch in gt_out.items() if len(ch) >= 2]
        gt_neg = [s_ for s_, ch in gt_out.items() if len(ch) == 1]
        random.shuffle(gt_neg)
        for s_ in gt_pos:
            t_, z_, y_, x_ = gt_nodes_plain[s_]
            samples.append({"x": make_crop(stem, int(t_), (z_, y_, x_), frame_cache, T), "y": 1, "stem": stem, "source": "gt_pos", "dc": np.nan})
            label_stats["pos_gt"] += 1
        for s_ in gt_neg[: max(3, EXTRA_NEG_PER_POS * len(gt_pos))]:
            t_, z_, y_, x_ = gt_nodes_plain[s_]
            samples.append({"x": make_crop(stem, int(t_), (z_, y_, x_), frame_cache, T), "y": 0, "stem": stem, "source": "gt_neg", "dc": np.nan})
            label_stats["neg_gt"] += 1
        frame_cache.clear()
        if (k_ + 1) % 20 == 0:
            print(f"  GT crops: {k_ + 1}/{len(gt_stems)} clips, {len(samples)} samples, {time.time() - t_gt0:.0f}s", flush=True)
finally:
    globals()["TEST_DIR"] = _real_test_dir

print("label stats:", dict(label_stats))
ys = np.array([s["y"] for s in samples]); print(f"samples: {len(samples)}  positives: {int(ys.sum())}  negatives: {int((1 - ys).sum())}")
assert ys.sum() >= 60, "too few positives to train anything meaningful"
X = np.stack([s["x"] for s in samples]); Y = ys.astype(np.float32)
STEMS = np.array([s["stem"] for s in samples]); SOURCE = np.array([s["source"] for s in samples]); DC = np.array([s["dc"] for s in samples], dtype=np.float32)
out_dir = WORKING_DIR / "divnet_ours"; out_dir.mkdir(exist_ok=True)
np.savez_compressed(out_dir / "divnet_dataset.npz", X=X, Y=Y, STEMS=STEMS, SOURCE=SOURCE, DC=DC)
print("dataset saved:", X.shape, X.dtype)
'''

TRAIN_CELL = r'''
# ================================================================ OURS (v28): train the DivNet-architecture classifier from scratch
import torch, math, time
from sklearn.metrics import roc_auc_score
device = torch.device("cuda")
torch.manual_seed(SEED)

class _DivNetConvBlock(torch.nn.Module):            # == DivNet architecture in the v25 gate (state_dict compatible)
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_c, out_c, kernel_size=3, padding=1), torch.nn.BatchNorm3d(out_c), torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_c, out_c, kernel_size=3, padding=1), torch.nn.BatchNorm3d(out_c), torch.nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class DivNetMitosisClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = _DivNetConvBlock(4, 16); self.p1 = torch.nn.MaxPool3d((1, 2, 2))   # 4 frames = input channels
        self.b2 = _DivNetConvBlock(16, 32); self.p2 = torch.nn.MaxPool3d((2, 2, 2))
        self.b3 = _DivNetConvBlock(32, 64); self.gap = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(inplace=True), torch.nn.Linear(32, 1))
    def forward(self, x):
        x = self.p1(self.b1(x)); x = self.p2(self.b2(x)); x = self.gap(self.b3(x)); return self.fc(torch.flatten(x, 1))

# split by clip (deterministic hash) so val clips are never seen in training
stems_unique = sorted(set(STEMS.tolist()))
val_clips = {s for s in stems_unique if int(hashlib.md5(s.encode()).hexdigest(), 16) % 100 < VAL_FRACTION * 100}
is_val = np.array([s in val_clips for s in STEMS])
print(f"train clips {len(stems_unique) - len(val_clips)} / val clips {len(val_clips)} | train samples {int((~is_val).sum())} (pos {int(Y[~is_val].sum())}) | val samples {int(is_val.sum())} (pos {int(Y[is_val].sum())})")
assert Y[is_val].sum() >= 5 and (1 - Y[is_val]).sum() >= 5, "val split too small; lower VAL_FRACTION or raise N_PER_TYPE"

Xt = torch.from_numpy(X.astype(np.float32))      # N,4,16,32,32  (frames = channels)
Yt = torch.from_numpy(Y)
tr_idx = np.where(~is_val)[0]; va_idx = np.where(is_val)[0]

def augment(xb):
    # D4 in yx + random z flip + intensity jitter (crops are already z-normalised)
    if random.random() < 0.5: xb = xb.flip(-1)
    if random.random() < 0.5: xb = xb.flip(-2)
    k = random.randint(0, 3)
    if k: xb = torch.rot90(xb, k, dims=(-2, -1))
    if random.random() < 0.5: xb = xb.flip(-3)
    return xb * (1.0 + 0.1 * (torch.rand(1, device=xb.device) - 0.5)) + 0.1 * (torch.rand(1, device=xb.device) - 0.5)

model = DivNetMitosisClassifier().to(device)
pos_w = torch.tensor([(1 - Y[tr_idx]).sum() / max(1.0, Y[tr_idx].sum())], device=device)
crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_EPOCHS)

def predict(idx):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 128):
            xb = Xt[idx[i:i + 128]].to(device); out.append(torch.sigmoid(model(xb)).squeeze(1).cpu())
    return torch.cat(out).numpy()

best = {"auc": -1.0}; t0 = time.time()
for ep in range(1, TRAIN_EPOCHS + 1):
    model.train(); perm = np.random.permutation(tr_idx); tot = 0.0
    for i in range(0, len(perm), TRAIN_BATCH):
        b = perm[i:i + TRAIN_BATCH]; xb = augment(Xt[b].to(device)); yb = Yt[b].to(device)
        loss = crit(model(xb).squeeze(1), yb); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
    sched.step()
    pv = predict(va_idx); auc = roc_auc_score(Y[va_idx], pv)
    pred05 = pv >= 0.5; tp = int((pred05 & (Y[va_idx] == 1)).sum()); fp = int((pred05 & (Y[va_idx] == 0)).sum()); fn = int((~pred05 & (Y[va_idx] == 1)).sum())
    print(f"epoch {ep:2d} loss {tot / len(perm):.4f} | val AUC {auc:.4f} | @0.5 tp/fp/fn {tp}/{fp}/{fn} | {time.time() - t0:.0f}s", flush=True)
    if auc > best["auc"]:
        best = {"auc": auc, "epoch": ep, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "pv": pv}

# threshold: maximise F1 on val (candidate-source samples only, since that is what the gate sees)
pv = best["pv"]; yv = Y[va_idx]; src_v = SOURCE[va_idx]; mask = src_v == "cand"
best_thr, best_f1 = 0.5, -1.0
for thr in np.linspace(0.1, 0.9, 33):
    pr = pv[mask] >= thr; tp = float((pr & (yv[mask] == 1)).sum()); fp = float((pr & (yv[mask] == 0)).sum()); fn = float((~pr & (yv[mask] == 1)).sum())
    f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
    if f1 > best_f1: best_f1, best_thr = f1, float(thr)
dc_v = DC[va_idx]; dc_mask = mask & np.isfinite(dc_v)
dc_auc = roc_auc_score(yv[dc_mask], dc_v[dc_mask]) if dc_mask.sum() > 5 and len(set(yv[dc_mask].tolist())) == 2 else float("nan")
ours_auc_cand = roc_auc_score(yv[mask], pv[mask]) if mask.sum() > 5 and len(set(yv[mask].tolist())) == 2 else float("nan")
print("=" * 78)
print(f"BEST epoch {best['epoch']}  val AUC(all) {best['auc']:.4f}  | on pipeline candidates only: OURS AUC {ours_auc_cand:.4f}  vs  DeepCenter score AUC {dc_auc:.4f}")
print(f"F1-optimal threshold on candidates: {best_thr:.3f} (F1 {best_f1:.3f})  -> use BIOHUB_DIV_MIN_PROB={best_thr:.2f} in v25 R5/R6")
print("=" * 78)
meta = {"in_channels": 4, "val_auc_all": float(best["auc"]), "val_auc_cand": float(ours_auc_cand), "deepcenter_auc_cand": float(dc_auc), "best_epoch": int(best["epoch"]),
        "recommended_threshold": best_thr, "n_samples": int(len(Y)), "n_pos": int(Y.sum()), "val_clips": sorted(val_clips), "label_stats": dict(label_stats),
        "input_spec": {"frames": N_FRAMES, "z_pad": Z_PAD, "xy_pad": XY_PAD, "norm": "per-crop mean/std"}}
torch.save({"model_state": best["state"], "meta": meta}, out_dir / "best_overall.pt")
(out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
print("saved", out_dir / "best_overall.pt", "| keys:", len(best["state"]))
'''

def code(src): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src.strip("\n")}
# keep: config(0) + our config, guard(1), parse(2), pkg(3), predict(4), postprocess(5), hash(6), validator setup(7), matching(8), cache+base(9)
keep = cells[:10]
new = [{"cell_type": "markdown", "metadata": {}, "source": "# Biohub · v28 — train our own mitosis classifier on the pipeline's real division candidates\n\nGPU T4×2. Inputs: competition + pilkwang ×3. Output: `divnet_ours/best_overall.pt` (drop-in for the v25 DivNet gate)."},
       keep[0], code(CONFIG_CELL)] + keep[1:] + [code(LABEL_CELL), code(TRAIN_CELL)]
json.dump({"cells": new, "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}, open(OUT, "w"), indent=1)
print("wrote", OUT, "cells:", len(new))
