"""Builds biohub_v13_train_cellmot.ipynb — train the organisers' official baseline on Kaggle (GPU, Internet OFF)."""
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

md("""# Biohub · v13 — train the official baseline (TemporalUNet3D + node transformer) ourselves

Trains the organisers' end-to-end baseline (`royerlab/kaggle-cell-tracking-competition`, code shipped in the
**prep-bundle v2** Output) on the competition train set. Output: `weights/unet_transformer/split_0/`
(`edge_predictor_best.pth`, `edge_predictor_last.pth`, `last.pth` = full resume state, `config.json`, `history.json`).

**Inputs**: competition data + `biohub-prep-bundle-v2` Output. To **continue training**, additionally attach the
Output of the previous run of this notebook — `last.pth` is picked up automatically (model + optimizer + epoch).

**Settings**: GPU T4 x2, Internet off. One session ≈ 11 h of training; the loop stops by itself before the 12 h limit.
Validation = the same 12 hold-out clips as all our earlier CVs (never trained on).
""")

code('''# ------------------------------------------------------------------ 0. settings
import os, sys, glob, time, json, subprocess, importlib, shutil
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"   # less CUDA fragmentation
from pathlib import Path

EPOCHS_TOTAL      = 60          # target number of epochs *in total* (resumed epochs count); the time budget may stop earlier
LR                = 1e-4        # peak lr; cosine decay to 0.1*LR at EPOCHS_TOTAL (repo default constant 1e-4)
BATCH_SIZE        = 8           # 16 (repo default) needs ~15 GB on GPU0 -> CUDA OOM on a T4; 8 fits
NUM_WORKERS       = 4
N_HOLDOUT         = 12          # validation clips = all[1::len//12][:12] (identical to our CV / v11 hold-out)
RESUME            = True        # pick up last.pth from an attached previous Output
DET_LOSS_WEIGHT   = 1.0
DET_NEG_WEIGHT    = 1e-2
DOWNSAMPLE        = (1, 4, 4)
WINDOW_SIZE       = 2
POOL_KERNEL_UM    = 5.0
AUGMENT           = True
MAX_DET_PER_FRAME = 1200        # top-K detections/frame fed to the edge transformer (memory guard; real cells < 700/frame)
TIME_BUDGET_SEC   = 11.0 * 3600 # total wall clock for this notebook (Kaggle limit 12 h)
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
if resume is None: print("    no previous checkpoint attached -> training from scratch")
WEIGHTS_OUT = OUT_DIR / "weights" / "unet_transformer" / "split_0"
budget = max(600, TIME_BUDGET_SEC - (time.time() - T_START) - 10 * 60)
print(f"training budget {budget/3600:.2f} h")
hist, best = CT.train_loop(model, train_loader, val_loader, out_dir=WEIGHTS_OUT, cfg=cfg,
                           epochs=2 if QUICK_TEST else EPOCHS_TOTAL, lr=LR, det_loss_weight=DET_LOSS_WEIGHT,
                           det_neg_weight=DET_NEG_WEIGHT, time_budget_sec=budget, resume=resume,
                           max_iters=3 if QUICK_TEST else None)
''')
code('''# ------------------------------------------------------------------ 6. summary
import pandas as pd
h = pd.DataFrame(hist)
display(h[["epoch", "lr", "edge_loss", "det_loss", "val_loss", "val_acc", "val_recall", "score", "t_train", "t_val"]].round(4))
print("best val acc*recall:", round(best, 4), "at epoch", int(h.loc[h.score.idxmax(), "epoch"]))
print("files:", sorted(p.name for p in WEIGHTS_OUT.iterdir()))
print(f"total {(time.time()-T_START)/3600:.2f} h. Next: (a) attach this Output to the v14 inference notebook, or "
      f"(b) attach it to THIS notebook and run again to continue from epoch {int(h.epoch.max())}.")
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out_name = sys.argv[1] if len(sys.argv) > 1 else "biohub_v13_train_cellmot.ipynb"
Path(out_name).write_text(nbf.writes(nb))
print("written", out_name, len(cells), "cells")
