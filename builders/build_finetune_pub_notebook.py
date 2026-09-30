"""Builds biohub_v22_finetune_pub.ipynb — continue training the PUBLIC 400-epoch weights (pilkwang support pack) on (almost) all train clips with a low LR -> a third seed for the public dual-seed pipeline."""
import sys
from pathlib import Path

import nbformat as nbf
from nb_common import LOCATE_CELL, PIP_CELL

train_src = Path("cellmot_train.py").read_text()
infer_src = Path("cellmot_infer.py").read_text()
pipeline_src = Path("pipeline.py").read_text()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Biohub · v22 — fine-tune the public 400-epoch weights on all train clips (third seed)

Trains the organisers' end-to-end baseline (`royerlab/kaggle-cell-tracking-competition`, code shipped in the
**prep-bundle v2** Output) on the competition train set. Output: `weights/unet_transformer/split_0/`
(`edge_predictor_best.pth`, `edge_predictor_last.pth`, `last.pth` = full resume state, `config.json`, `history.json`).

**Inputs**: competition data + `biohub-prep-bundle-v2` Output + the public dataset `pilkwang/biohub-tracking-support-pack-50ep-v1`
(start weights, chosen by `INIT_WEIGHTS_HINT`). Do NOT attach a v13 Output (its `last.pth` would be resumed instead).
The public weights saw only split 0 (80% of train); here we train on all but 4 clips with a low LR for a few epochs.

**Settings**: GPU T4 x2, Internet off. One session ≈ 11 h of training; the loop stops by itself before the 12 h limit.
Validation = the same 12 hold-out clips as all our earlier CVs (never trained on).
""")

code('''# ------------------------------------------------------------------ 0. settings
import os, sys, glob, time, json, subprocess, importlib, shutil
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"   # less CUDA fragmentation
from pathlib import Path

EPOCHS_TOTAL      = 3           # fine-tuning: a few epochs from the converged public weights
LR                = 2e-5        # low peak lr (repo trained at 1e-4); cosine to 0.5*LR
LR_MIN_FRAC       = 0.5
INIT_WEIGHTS_HINT = "support-pack"   # substring of the checkpoint path to start from (public 400-epoch weights)
BATCH_SIZE        = 8           # 16 (repo default) needs ~15 GB on GPU0 -> CUDA OOM on a T4; 8 fits
NUM_WORKERS       = 4
N_HOLDOUT         = 4           # only a sanity-check validation; the public model has seen most clips anyway
RESUME            = False       # start from INIT_WEIGHTS, not from a last.pth
DET_LOSS_WEIGHT   = 1.0
DET_NEG_WEIGHT    = 1e-2
DOWNSAMPLE        = (1, 4, 4)
WINDOW_SIZE       = 2
POOL_KERNEL_UM    = 5.0
AUGMENT           = True
MAX_DET_PER_FRAME = 1200        # top-K detections/frame fed to the edge transformer (memory guard; real cells < 700/frame)
TIME_BUDGET_SEC   = 10.0 * 3600 # total wall clock (leave GPU quota for the inference A/B runs)
QUICK_TEST        = False       # True: 3 iterations/epoch, 2 epochs, 10 frames per clip -> checks the plumbing in minutes

KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
OUT_DIR      = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
T_START = time.time()
print("input root:", KAGGLE_INPUT, "| output:", OUT_DIR)
''')
code(LOCATE_CELL)
code(PIP_CELL)
import base64
def _b64(src): return base64.b64encode(src.encode()).decode()
code('''# ------------------------------------------------------------------ 3. our wrapper code (written next to the output, imported as modules)
import base64
CODE_DIR = OUT_DIR / "code"; CODE_DIR.mkdir(exist_ok=True)
for _name, _b in [("pipeline.py", "''' + _b64(pipeline_src) + '''"),
                  ("cellmot_train.py", "''' + _b64(train_src) + '''"),
                  ("cellmot_infer.py", "''' + _b64(infer_src) + '''")]:
    (CODE_DIR / _name).write_bytes(base64.b64decode(_b))
sys.path.insert(0, str(CODE_DIR))
import cellmot_train as CT
assert REPO_DIR is not None, "official baseline repo not found: attach the prep-bundle v2 Output (it contains repo/)"
T = CT.setup_repo(REPO_DIR)
print("repo:", REPO_DIR, "| commit:", (Path(REPO_DIR) / "COMMIT.txt").read_text().strip() if (Path(REPO_DIR) / "COMMIT.txt").exists() else "?")
''')
code('''# ------------------------------------------------------------------ 4. splits + data loaders (metadata only; frames are read from zarr per batch)
train_dir = COMP_DIR / "train"
splits = CT.make_splits(train_dir, N_HOLDOUT)
(OUT_DIR / "dataset_splits.json").write_text(json.dumps([splits], indent=1))
print(f"{len(splits['train'])} train clips | {len(splits['test'])} hold-out: {splits['test']}")
max_frames = 10 if QUICK_TEST else None
train_loader, val_loader, max_nodes = CT.build_loaders(
    train_dir, splits["train"], splits["test"], downsample=DOWNSAMPLE, window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, max_frames=max_frames, augment=AUGMENT)
print(f"train batches/epoch: {len(train_loader)} | val batches: {len(val_loader)} | setup took {time.time()-T_START:.0f}s")
''')
code('''# ------------------------------------------------------------------ 5. model, resume, train
cfg = dict(unet_out_channels=32, unet_layers=[32, 64, 128], downsample=list(DOWNSAMPLE), window_size=WINDOW_SIZE,
           pool_kernel_um=POOL_KERNEL_UM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CT.CAP["max_det_per_frame"] = MAX_DET_PER_FRAME
model = CT.build_model(cfg, device, data_parallel=True)
resume = CT.find_resume(CKPT_DIRS) if (RESUME and CKPT_DIRS) else None
inits = [d / "edge_predictor_best.pth" for d in CKPT_DIRS if INIT_WEIGHTS_HINT in str(d) and (d / "edge_predictor_best.pth").exists()]
assert inits, f"no checkpoint path contains '{INIT_WEIGHTS_HINT}': attach pilkwang/biohub-tracking-support-pack-50ep-v1 (found {[str(d) for d in CKPT_DIRS]})"
INIT_WEIGHTS = inits[-1]
_state = torch.load(INIT_WEIGHTS, map_location="cpu", weights_only=True)
_mi, _un = CT._load_plain_state(model, _state)
assert not _un and len(_mi) == 0, f"init weights do not match the model: missing={_mi[:5]} unexpected={_un[:5]}"
import hashlib; print(f"init weights: {INIT_WEIGHTS} sha256={hashlib.sha256(INIT_WEIGHTS.read_bytes()).hexdigest()[:16]} | keys loaded: {len(_state)}")
with torch.no_grad():
    _vl, _va, _vr = T.evaluate(model, val_loader, device, pool_kernel_um=POOL_KERNEL_UM)
print(f"BEFORE fine-tuning: val_loss={_vl:.4f} val_acc={_va:.4f} val_recall={_vr:.4f} score={_va*_vr:.4f}  (the epoch rows below must stay >= this)")
WEIGHTS_OUT = OUT_DIR / "weights" / "unet_transformer" / "split_0"
budget = max(600, TIME_BUDGET_SEC - (time.time() - T_START) - 10 * 60)
print(f"training budget {budget/3600:.2f} h")
hist, best = CT.train_loop(model, train_loader, val_loader, out_dir=WEIGHTS_OUT, cfg=cfg,
                           epochs=2 if QUICK_TEST else EPOCHS_TOTAL, lr=LR, det_loss_weight=DET_LOSS_WEIGHT,
                           det_neg_weight=DET_NEG_WEIGHT, time_budget_sec=budget, resume=resume,
                           max_iters=3 if QUICK_TEST else None, lr_min_frac=LR_MIN_FRAC)
''')
code('''# ------------------------------------------------------------------ 6. summary
import pandas as pd
h = pd.DataFrame(hist)
display(h[["epoch", "lr", "edge_loss", "det_loss", "val_loss", "val_acc", "val_recall", "score", "t_train", "t_val"]].round(4))
print("best val acc*recall:", round(best, 4), "at epoch", int(h.loc[h.score.idxmax(), "epoch"]))
print("files:", sorted(p.name for p in WEIGHTS_OUT.iterdir()))
print(f"total {(time.time()-T_START)/3600:.2f} h. Next: attach this Output to the v23 public-pipeline notebook as the third seed.")
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out_name = sys.argv[1] if len(sys.argv) > 1 else "biohub_v22_finetune_pub.ipynb"
Path(out_name).write_text(nbf.writes(nb))
print("written", out_name, len(cells), "cells")
