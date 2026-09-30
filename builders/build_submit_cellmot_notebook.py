"""Builds biohub_v16_cellmot_submit.ipynb — inference/submission with OUR trained official-baseline weights (+ v15 tuned post-processing)."""
import base64
import sys
from pathlib import Path

import nbformat as nbf
from nb_common import LOCATE_CELL, PIP_CELL

def _b64(p): return base64.b64encode(Path(p).read_bytes()).decode()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Biohub · v16 — submission with our self-trained baseline (UNet detector + node transformer + ILP), post-processing from v15

**Inputs**: competition data + `biohub-prep-bundle-v2` Output (wheels + repo) + the **v13 training Output**
(`weights/unet_transformer/split_0/edge_predictor_best.pth`) + optionally the **v15 Output** (`best_config.json`
= the best CV setting of the post-processing sweep; used as FINAL_CONFIG when `USE_BEST_CONFIG=True`). GPU T4 x2, Internet off.

1. `RUN_CV=True`: scores a few post-processing settings on the 12 hold-out clips (never trained on) with our
   official-style scorer — the same 12 clips / scorer as every earlier CV, so numbers are comparable (v12 = 0.8175).
2. `FINAL_CONFIG` is applied to the hidden test set → `submission.csv`.
""")

code('''# ------------------------------------------------------------------ 0. settings
import os, sys, glob, time, json, subprocess, importlib, shutil
os.environ["TQDM_DISABLE"] = "1"
from pathlib import Path

RUN_CV            = True
N_CV_SAMPLES      = 12
MAX_TEST_SAMPLES  = None            # None = all test clips; 1 = quick debug; 0 = CV only, no submission (e.g. CPU run)
DET_TTA           = True            # 4-flip test-time augmentation for detection (4x U-Net cost; set False on CPU)
WEIGHTS_PREFER    = "edge_predictor_best.pth"   # or "edge_predictor_last.pth"
CV_CONFIGS = {   # overrides: det_threshold, pool_kernel_um, use_ilp, ilp_* weights, edge_threshold, min_track_len
    "ilp_t95":         dict(det_threshold=0.95, pool_kernel_um=3.0, use_ilp=True, ilp_disappearance_weight=1.4, ilp_division_weight=1.0),  # v14 reference
    "ilp_t99_p5_L5":   dict(det_threshold=0.99, pool_kernel_um=5.0, use_ilp=True, ilp_disappearance_weight=1.4, ilp_division_weight=1.0, min_track_len=5),
}
FINAL_CONFIG      = "ilp_t95"
USE_BEST_CONFIG   = True            # if a v15 Output with best_config.json is attached, it becomes CV config "v15_best" and FINAL_CONFIG
TIME_BUDGET_SEC   = 10.5 * 3600
MODEL_PREFER      = "ctc"           # (Trackastra folder preference; unused here, kept for the shared locate cell)

KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
OUT_DIR      = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
T_START = time.time()
print("input root:", KAGGLE_INPUT, "| output:", OUT_DIR)
''')
code(LOCATE_CELL)
code(PIP_CELL)
code('''# ------------------------------------------------------------------ 3. code + model
import base64
CODE_DIR = OUT_DIR / "code"; CODE_DIR.mkdir(exist_ok=True)
for _name, _b in [("pipeline.py", "''' + _b64("pipeline.py") + '''"),
                  ("cellmot_train.py", "''' + _b64("cellmot_train.py") + '''"),
                  ("cellmot_infer.py", "''' + _b64("cellmot_infer.py") + '''")]:
    (CODE_DIR / _name).write_bytes(base64.b64decode(_b))
sys.path.insert(0, str(CODE_DIR))
from pipeline import read_geff, score_official, micro_average, rows_for_sample, write_submission
import cellmot_infer as CI
assert REPO_DIR is not None, "official baseline repo not found: attach the prep-bundle v2 Output"
P = CI.setup_predict(REPO_DIR)

cands = [d / WEIGHTS_PREFER for d in CKPT_DIRS if (d / WEIGHTS_PREFER).exists()]
assert cands, f"no {WEIGHTS_PREFER} found in inputs: attach the v13 training Output"
cands.sort(key=lambda p: p.stat().st_mtime)
WEIGHTS = cands[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 4); print("CPU mode, threads:", torch.get_num_threads(), "| det_tta:", DET_TTA)
model, WINDOW, DOWNSAMPLE = CI.load_model(WEIGHTS, device)
hist_p = WEIGHTS.parent / "history.json"
if hist_p.exists():
    _h = json.loads(hist_p.read_text()); print(f"weights: {WEIGHTS} | trained epochs: {len(_h)} | best val acc*recall {max(r['score'] for r in _h):.4f}")
else:
    print("weights:", WEIGHTS)
print("window", WINDOW, "downsample", DOWNSAMPLE, "device", device)

BEST_JSON = [Path(dp) / "best_config.json" for dp, fns in _walk_pruned(KAGGLE_INPUT) if "best_config.json" in fns]
if USE_BEST_CONFIG and BEST_JSON:
    _b = json.loads(BEST_JSON[-1].read_text())
    CV_CONFIGS["v15_best"] = {**_b["det"], **_b["post"], "min_track_len": _b["min_track_len"]}
    FINAL_CONFIG = "v15_best"
    print(f"v15 best_config.json found ({BEST_JSON[-1]}; CV {_b.get('cv_score')}) -> FINAL_CONFIG = v15_best:", CV_CONFIGS["v15_best"])
else:
    print("no v15 best_config.json attached -> FINAL_CONFIG =", FINAL_CONFIG)

def run_clip(zp, over, log=print):
    cfg = CI.make_cfg(**{"det_tta": DET_TTA, **over})
    nodes, edges, w = CI.predict_clip(model, zp, device, cfg, WINDOW, DOWNSAMPLE, log=log)
    return nodes, edges
''')
code('''# ------------------------------------------------------------------ 4. CV on the 12 hold-out clips (official-style scorer)
if RUN_CV and (COMP_DIR / "train").is_dir():
    cv_zarrs = sorted(p for p in (COMP_DIR / "train").iterdir() if p.name.endswith(".zarr"))
    cv_zarrs = [p for p in cv_zarrs if (p.parent / (p.name[:-5] + ".geff")).exists()]
    step = max(1, len(cv_zarrs) // max(1, N_CV_SAMPLES))
    cv_zarrs = cv_zarrs[1::step][:N_CV_SAMPLES]
    print("CV clips:", [p.name[:-5] for p in cv_zarrs])
    gts = {zp: read_geff(zp.parent / (zp.name[:-5] + ".geff")) for zp in cv_zarrs}
    cv_rows, cv_summary = [], []
    for name, over in CV_CONFIGS.items():
        rows = []
        for zp in cv_zarrs:
            if time.time() - T_START > TIME_BUDGET_SEC:
                print("!! time budget exhausted during CV; remaining clips skipped"); break
            t_clip = time.time()
            nodes, edges = run_clip(zp, over, log=lambda *a: None)
            sc = score_official(nodes, edges, gts[zp]); sc.update(config=name, sample=zp.name[:-5]); rows.append(sc)
            print(f"    [{name}] {zp.name[:-5]}: score={sc['score']:.3f} (tp={sc['edge_tp']} fp={sc['edge_fp']} fn={sc['edge_fn']}) "
                  f"node_recall={sc['node_recall']:.3f} divJ={sc['div_J']:.2f} n_pred/est={sc['n_pred']}/{sc['est']} ({time.time()-t_clip:.0f}s)")
        if not rows: break
        ma = micro_average(rows); ma["config"] = name; cv_summary.append(ma); cv_rows += rows
        print(f"  ==> [{name}] micro score={ma['score']:.4f} adjJ={ma['adj_edge_J']:.4f} divJ={ma['div_J']:.3f}  ({time.time()-T_START:.0f}s)")
    import pandas as pd
    display(pd.DataFrame(cv_summary).set_index("config"))
    display(pd.DataFrame(cv_rows).pivot_table(index="sample", columns="config", values="score").round(3))
    pd.DataFrame(cv_rows).to_csv(OUT_DIR / "cv_rows.csv", index=False)
print("FINAL_CONFIG:", FINAL_CONFIG, "->", CV_CONFIGS[FINAL_CONFIG])
if MAX_TEST_SAMPLES == 0:
    print("MAX_TEST_SAMPLES = 0 -> CV only, no submission.csv (skip the next two cells)")
''')
code('''# ------------------------------------------------------------------ 5. inference on test -> submission.csv
all_rows, summary = [], []
zarrs = TEST_ZARRS if MAX_TEST_SAMPLES is None else TEST_ZARRS[:MAX_TEST_SAMPLES]
sub_path = OUT_DIR / "submission.csv"
for k, zp in enumerate(zarrs):
    name = zp.name[:-5]
    if time.time() - T_START > TIME_BUDGET_SEC:
        print(f"!! time budget exhausted after {k} samples; remaining samples get a placeholder node")
        all_rows.append((name, "node", 1, 0, 0, 0, 0, -1, -1)); continue
    print(f"[{name}]")
    nodes, edges = run_clip(zp, CV_CONFIGS[FINAL_CONFIG])
    rows = rows_for_sample(name, nodes, edges) or [(name, "node", 1, 0, 0, 0, 0, -1, -1)]
    all_rows += rows; summary.append(dict(name=name, n_nodes=len(nodes), n_edges=len(edges)))
    print(f"    [{k+1}/{len(zarrs)}] done, elapsed {time.time()-T_START:.0f}s")
if zarrs:
    df = write_submission(all_rows, sub_path)
    print(df.row_type.value_counts().to_dict(), "| written:", sub_path, f"{sub_path.stat().st_size/1e6:.1f} MB")
else:
    print("no test inference requested (MAX_TEST_SAMPLES = 0)")
''')
code('''# ------------------------------------------------------------------ 6. format self-check
import pandas as pd
if not zarrs:
    print("CV-only run: nothing to check")
else:
    sample = pd.read_csv(COMP_DIR / "sample_submission.csv")
    sub = pd.read_csv(sub_path)
    assert list(sub.columns) == list(sample.columns), (list(sub.columns), list(sample.columns))
    assert sub["id"].is_unique and (sub["id"] == range(len(sub))).all()
    nodes_ = sub[sub.row_type == "node"]; edges_ = sub[sub.row_type == "edge"]
    assert not nodes_.duplicated(["dataset", "node_id"]).any(), "duplicate node ids within a dataset"
    nid = set(zip(nodes_.dataset, nodes_.node_id))
    bad = [(d, s, t) for d, s, t in zip(edges_.dataset, edges_.source_id, edges_.target_id) if (d, s) not in nid or (d, t) not in nid]
    assert not bad, f"{len(bad)} edges reference unknown nodes, e.g. {bad[:3]}"
    assert set(p.name[:-5] for p in zarrs) <= set(sub.dataset.unique()), "some test datasets have no rows"
    for c in ["node_id", "t", "z", "y", "x", "source_id", "target_id"]:
        assert np.issubdtype(sub[c].dtype, np.integer), c
    print("format OK ✓ |", len(nodes_), "nodes,", len(edges_), "edges over", sub.dataset.nunique(), "datasets |", f"total {time.time()-T_START:.0f}s")
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out_name = sys.argv[1] if len(sys.argv) > 1 else "biohub_v16_cellmot_submit.ipynb"
Path(out_name).write_text(nbf.writes(nb))
print("written", out_name, len(cells), "cells")
