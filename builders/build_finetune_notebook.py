"""Builds biohub_v11_finetune.ipynb — Trackastra fine-tuning on Kaggle (GPU, Internet OFF is fine)."""
import sys
from pathlib import Path

import nbformat as nbf
from nb_common import LOCATE_CELL, PIP_CELL

pipeline_src = Path("pipeline.py").read_text()
finetune_src = Path("finetune.py").read_text()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Biohub · v11 — fine-tune the Trackastra linker on the competition GT

**What this notebook produces** (in its Output): `models/ctc_ft/` — Trackastra weights re-trained on *our own
DoG detections* with the sparse `.geff` ground truth, plus `feat_cache/` (re-usable features, one `.npz` per clip).
Attach this notebook's Output to the inference notebook (v12+); it prefers `ctc_ft` over the pretrained `ctc`.

**Inputs to attach**: competition data + `biohub-prep-bundle` Output (wheels + pretrained model).
Optionally the Output of a *previous run of this notebook* → its `feat_cache` is re-used and feature extraction is skipped.

**Settings**: Accelerator GPU T4 x2, Internet off. Runtime ≈ features 3–4 h (first run) + training ≈ 3–4 h + CV 20 min.

How the sparse GT is used: a detection matched to a GT cell (≤ 7 µm, like the metric) whose GT successor is also
detected gives one supervised *row* (the successor is the positive, every other next-frame detection is a true
negative). Unlabeled cells are masked out of the loss entirely. Hold-out clips are the same 12 clips as the v9 CV.
""")

code('''# ------------------------------------------------------------------ 0. settings
import os, sys, glob, time, json, subprocess, importlib, shutil
os.environ["TQDM_DISABLE"] = "1"   # quiet progress bars in the log
from pathlib import Path

DETECT_CONFIG = {"adaptive_radius": {"a": 26.5, "rmin": 3.0, "rmax": 6.0}, "merge_min_dist_um": 6.0, "thresh_k": 0.5}
                                  # = FINAL_CONFIG "adapt_a26_merge" of v8 (+ the calibrated thresh_k). Must match inference.
N_HOLDOUT         = 12            # hold-out clips = same rule as the v9 CV (cv_zarrs[1::step][:12]) -> honest comparison
MAX_TRAIN_CLIPS   = None          # None = all remaining train clips (order interleaves embryos)
FEATURE_BUDGET_SEC= 3.0 * 3600    # stop extracting features after this; training uses whatever is cached
FEATURE_WORKERS   = 4             # joblib workers for regionprops feature extraction
TRAIN_STEPS       = 3000          # optimizer steps (each = ACCUM windows)
ACCUM             = 8
LR                = 5e-5
WARMUP_STEPS      = 200
AUGMENT_LEVEL     = 2             # 0 none | 1 flip+affine | 2 + brightness + jitter
MAX_TOKENS        = 3000          # windows with more detections drop trailing frames (dense clips: ~600/frame x 4)
TRAIN_BUDGET_SEC  = 5.5 * 3600    # features took ~1.9 h on Kaggle for all 199 clips
EVAL_EVERY        = 250
MODEL_PREFER      = "ctc"         # start fine-tuning from the pretrained weights (use "ctc_ft" to continue a previous fine-tune)
SEED              = 0

KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
OUT_DIR      = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
T_START = time.time()
print("input root:", KAGGLE_INPUT, "| output:", OUT_DIR)
''')

code(LOCATE_CELL)
code(PIP_CELL)
code("# ------------------------------------------------------------------ 3. pipeline code (identical to the inference notebook)\n" + pipeline_src)
code("# ------------------------------------------------------------------ 3b. fine-tuning code\n" + finetune_src)

code('''# ------------------------------------------------------------------ 4. load pretrained model, choose clips
from trackastra.model import Trackastra
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = Trackastra.from_folder(Path(MODEL_DIR), device=dev)
transformer = model.transformer
print("start weights:", MODEL_DIR, "| pretrained folder:", PRETRAINED_DIR, "| device", dev,
      "| causal_norm", transformer.config["causal_norm"], "| window", transformer.config["window"])
CFG.update(DETECT_CONFIG)

train_dir = COMP_DIR / "train"
all_zarrs = sorted(p for p in train_dir.iterdir() if p.name.endswith(".zarr") and (train_dir / (p.name[:-5] + ".geff")).exists())
step = max(1, len(all_zarrs) // max(1, N_HOLDOUT))
HOLDOUT = all_zarrs[1::step][:N_HOLDOUT]                    # identical rule to the inference notebook's RUN_CV
rest = [p for p in all_zarrs if p not in set(HOLDOUT)]
# interleave so that every embryo appears early (matters if the feature budget stops extraction early)
by_embryo = {}
for p in rest:
    by_embryo.setdefault(p.name.split("_")[0], []).append(p)
TRAIN = []
while any(by_embryo.values()):
    for k in list(by_embryo):
        if by_embryo[k]:
            TRAIN.append(by_embryo[k].pop(0))
if MAX_TRAIN_CLIPS:
    TRAIN = TRAIN[:MAX_TRAIN_CLIPS]
print(f"{len(all_zarrs)} train clips -> hold-out {len(HOLDOUT)}: {[p.name[:-5] for p in HOLDOUT]}")
print(f"fine-tune on up to {len(TRAIN)} clips, first: {[p.name[:-5] for p in TRAIN[:6]]} ...")
''')

code('''# ------------------------------------------------------------------ 5. feature cache (re-use attached caches, extract the rest)
CACHE_OUT = OUT_DIR / "feat_cache"; CACHE_OUT.mkdir(exist_ok=True)
existing = {}
for d in CACHE_DIRS + [CACHE_OUT]:
    for p in sorted(d.glob("*.npz")):
        existing.setdefault(p.name[:-4], p)
print(f"{len(existing)} cached clips found in attached inputs")

def _cache_path(zp):
    name = zp.name[:-5]
    if name in existing:
        return existing[name]
    if time.time() - T_START > FEATURE_BUDGET_SEC:
        return None
    out = CACHE_OUT / (name + ".npz")
    extract_clip_cache(zp, train_dir / (name + ".geff"), CFG, out, n_workers=FEATURE_WORKERS)
    existing[name] = out
    return out

t0 = time.time(); n_new = 0
hold_npz = []
for zp in HOLDOUT:                                  # hold-out first: they are needed for the final selection
    p = _cache_path(zp); hold_npz.append(p)
train_npz = []
for k, zp in enumerate(TRAIN):
    p = _cache_path(zp)
    if p is None:
        print(f"!! feature budget reached: {k}/{len(TRAIN)} train clips cached"); break
    train_npz.append(p)
    if k % 10 == 9:
        print(f"    ... {k+1}/{len(TRAIN)} train clips, elapsed {time.time()-T_START:.0f}s")
hold_npz = [p for p in hold_npz if p is not None]
print(f"features ready: {len(hold_npz)} hold-out + {len(train_npz)} train clips ({time.time()-t0:.0f}s)")

hold_caches = [ClipCache(p) for p in hold_npz]
train_caches = [ClipCache(p) for p in train_npz]
import pandas as pd
tab = pd.DataFrame([dict(clip=c.name, frames=c.n_frames, detections=len(c.labels), radius=c.radius,
                         gt_nodes=len(c.gt_ids), gt_matched=int((c.gt_id >= 0).sum()), gt_edges=len(c.gt_edges),
                         sup_rows=c.n_sup_rows, sup_cols=c.n_sup_cols, split="holdout" if c in hold_caches else "train")
                   for c in hold_caches + train_caches])
display(tab.groupby("split")[["detections", "gt_nodes", "gt_matched", "gt_edges", "sup_rows", "sup_cols"]].sum())
display(tab.head(15))
tab.to_csv(OUT_DIR / "clips_table.csv", index=False)
''')

code('''# ------------------------------------------------------------------ 6. fine-tune
WORK = OUT_DIR / "ft_work"
remaining = max(600, min(TRAIN_BUDGET_SEC, 11.0 * 3600 - (time.time() - T_START) - 40 * 60))   # leave ~40 min for CV + save
print(f"training budget: {remaining/3600:.2f} h")
import traceback
try:
    hist = finetune(transformer, train_caches, hold_caches, WORK, W=transformer.config["window"], steps=TRAIN_STEPS, lr=LR,
                    warmup=WARMUP_STEPS, accum=ACCUM, augment=AUGMENT_LEVEL, max_tokens=MAX_TOKENS, time_budget_sec=remaining,
                    eval_every=EVAL_EVERY, eval_windows=150, causal_norm=transformer.config["causal_norm"], seed=SEED)
    display(pd.DataFrame(hist))
except Exception:
    # never let a training crash lose the 2 h feature cache: the notebook must finish so the Output is kept
    print("!! TRAINING FAILED — the feat_cache is still written to the Output; attach it next time to skip extraction")
    traceback.print_exc()
''')

code('''# ------------------------------------------------------------------ 7. select by official-style CV on the hold-out clips, save models/ctc_ft
FT_DIR = OUT_DIR / "models" / "ctc_ft"
winner, cv_scores = select_and_save(transformer, WORK, hold_caches, CFG, train_dir, PRETRAINED_DIR, FT_DIR)
print("\\nhold-out micro scores:", cv_scores)
print("selected:", winner, "->", sorted(p.name for p in FT_DIR.iterdir()))
for p in ("pretrained.pt", "last.pt", "best_val.pt"):          # keep output small (each is 110 MB)
    (WORK / p).unlink(missing_ok=True)
print(f"total {time.time()-T_START:.0f}s. Next: Save Version -> Save & Run All; then attach this notebook's Output to the "
      f"inference notebook (it prefers models/ctc_ft). Re-running this notebook with its own Output attached skips feature extraction.")
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out_name = sys.argv[1] if len(sys.argv) > 1 else "biohub_v11_finetune.ipynb"
Path(out_name).write_text(nbf.writes(nb))
print("written", out_name, len(cells), "cells")
