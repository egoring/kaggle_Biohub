"""Builds biohub_v15_cellmot_cv_tune.ipynb — CPU-friendly post-processing sweep on the 12 hold-out clips.

Stage 1 (network, expensive, ~4-8 min/clip on CPU) runs ONCE per detection config and is cached as .npz in
OUT_DIR/raw_cache/<det_cfg>/<clip>.npz. Attach this notebook's Output to a later run and the cache is reused, so
new ILP / pruning settings can be scored in minutes.
"""
import base64
import sys
from pathlib import Path

import nbformat as nbf
from nb_common import LOCATE_CELL, PIP_CELL

def _b64(p): return base64.b64encode(Path(p).read_bytes()).decode()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Biohub · v15 — post-processing sweep for our self-trained baseline (CPU OK, no submission)

**Inputs**: competition data + `biohub-prep-bundle-v2` Output + the **v13 training Output** (+ optionally a previous
v15 Output: its `raw_cache/` is reused so the network is not re-run).

Why: the v14 CPU CV (0.8196) showed node recall 0.92–1.00 but **1.2–2.3× more nodes than the estimate** → the metric's
node-count penalty costs ≈0.04 (adjJ 0.8196 vs 0.8586 without penalty). This notebook sweeps the cheap knobs:
detection threshold / peak-pooling radius (network stage, cached) × ILP weights / edge threshold / short-track pruning
(post stage, seconds per clip). Output: `cv_rows.csv`, `cv_summary.csv`, `best_config.json`.
""")

code('''# ------------------------------------------------------------------ 0. settings
import os, sys, glob, time, json, subprocess, importlib, shutil
os.environ["TQDM_DISABLE"] = "1"
from pathlib import Path

N_CV_SAMPLES      = 12
DET_TTA           = False           # 4-flip TTA = 4x network cost; keep False on CPU, True on GPU
WEIGHTS_PREFER    = "edge_predictor_best.pth"
SAVE_RAW          = True            # cache stage-1 output to OUT_DIR/raw_cache (reused when this Output is attached later)
# stage 1 (network) configs: det_threshold = min peak probability, pool_kernel_um = min distance between two peaks
DET_CONFIGS = {
    "t95_p3": dict(det_threshold=0.95, pool_kernel_um=3.0),   # = v14 setting (reference: 0.8196 on CPU w/o TTA)
    "t99_p3": dict(det_threshold=0.99, pool_kernel_um=3.0),   # repo default threshold
    "t95_p5": dict(det_threshold=0.95, pool_kernel_um=5.0),   # = training pool radius; fewer double peaks per nucleus
    "t99_p5": dict(det_threshold=0.99, pool_kernel_um=5.0),
}
# stage 2 (linking) configs: ILP weights (edge reward = -prob; a track costs appearance + disappearance), edge_threshold
POST_CONFIGS = {
    "ilp_d14":        dict(use_ilp=True,  ilp_appearance_weight=0.0, ilp_disappearance_weight=1.4, ilp_division_weight=1.0),  # = v14
    "ilp_d14_a05":    dict(use_ilp=True,  ilp_appearance_weight=0.5, ilp_disappearance_weight=1.4, ilp_division_weight=1.0),
    "ilp_d20_a10":    dict(use_ilp=True,  ilp_appearance_weight=1.0, ilp_disappearance_weight=2.0, ilp_division_weight=1.0),
    "ilp_d14_e50":    dict(use_ilp=True,  ilp_appearance_weight=0.0, ilp_disappearance_weight=1.4, ilp_division_weight=1.0, edge_threshold=0.5),
    "ilp_d14_div05":  dict(use_ilp=True,  ilp_appearance_weight=0.0, ilp_disappearance_weight=1.4, ilp_division_weight=0.5),
    "greedy":         dict(use_ilp=False),
}
MIN_TRACK_LENS    = [1, 3, 5, 10]   # after linking, drop tracks (connected components) with fewer nodes; 1 = keep all
TIME_BUDGET_SEC   = 11.0 * 3600
MODEL_PREFER      = "ctc"

KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
OUT_DIR      = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
T_START = time.time()
print("input root:", KAGGLE_INPUT, "| output:", OUT_DIR)
''')
code(LOCATE_CELL)
code(PIP_CELL)
code('''# ------------------------------------------------------------------ 3. code + model + raw cache lookup
import base64
CODE_DIR = OUT_DIR / "code"; CODE_DIR.mkdir(exist_ok=True)
for _name, _b in [("pipeline.py", "''' + _b64("pipeline.py") + '''"),
                  ("cellmot_train.py", "''' + _b64("cellmot_train.py") + '''"),
                  ("cellmot_infer.py", "''' + _b64("cellmot_infer.py") + '''")]:
    (CODE_DIR / _name).write_bytes(base64.b64decode(_b))
sys.path.insert(0, str(CODE_DIR))
from pipeline import read_geff, score_official, micro_average
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

# previous v15 Output attached? -> raw_cache/<det_cfg>/<clip>.npz can be reused (only if made with the same weights + TTA)
RAW_IN = [Path(dp) for dp, fns in _walk_pruned(KAGGLE_INPUT) if Path(dp).name == "raw_cache"]
RAW_OUT = OUT_DIR / "raw_cache"
print("raw caches attached:", [str(p) for p in RAW_IN])
import hashlib
CACHE_TAG = f"{hashlib.md5(WEIGHTS.read_bytes()).hexdigest()[:12]}|tta{int(DET_TTA)}"   # cache valid only for these exact weights

def find_raw(det_name, clip):
    for r in RAW_IN + [RAW_OUT]:
        p = r / det_name / f"{clip}.npz"
        if p.exists():
            tag = (r / det_name / "TAG.txt")
            if tag.exists() and tag.read_text().strip() != CACHE_TAG:
                continue                       # cache from other weights / TTA setting -> ignore
            return p
    return None
''')
code('''# ------------------------------------------------------------------ 4. sweep on the 12 hold-out clips
import pandas as pd
cv_zarrs = sorted(p for p in (COMP_DIR / "train").iterdir() if p.name.endswith(".zarr"))
cv_zarrs = [p for p in cv_zarrs if (p.parent / (p.name[:-5] + ".geff")).exists()]
step = max(1, len(cv_zarrs) // max(1, N_CV_SAMPLES))
cv_zarrs = cv_zarrs[1::step][:N_CV_SAMPLES]
print("CV clips:", [p.name[:-5] for p in cv_zarrs])
gts = {zp.name[:-5]: read_geff(zp.parent / (zp.name[:-5] + ".geff")) for zp in cv_zarrs}

rows, stopped = [], False
for det_name, det_over in DET_CONFIGS.items():
    if stopped: break
    for zp in cv_zarrs:
        clip = zp.name[:-5]
        cached = find_raw(det_name, clip)
        if cached is None:
            if time.time() - T_START > TIME_BUDGET_SEC - 900:
                print(f"!! time budget: stopping before [{det_name}] {clip}; later configs skipped"); stopped = True; break
            t0 = time.time()
            cfg = CI.make_cfg(det_tta=DET_TTA, **det_over)
            coords, raw_edges = CI.predict_raw(model, zp, device, cfg, WINDOW, DOWNSAMPLE)
            t_net = time.time() - t0
            if SAVE_RAW:
                CI.save_raw(RAW_OUT / det_name / f"{clip}.npz", coords, raw_edges)
                (RAW_OUT / det_name / "TAG.txt").write_text(CACHE_TAG)
            src = f"net {t_net:.0f}s"
        else:
            coords, raw_edges = CI.load_raw(cached); src = "cache"
        line = [f"[{det_name}] {clip}: {len(coords)} det, {len(raw_edges)} cand. edges ({src})"]
        for post_name, post_over in POST_CONFIGS.items():
            t1 = time.time()
            cfg = CI.make_cfg(det_tta=DET_TTA, **det_over, **post_over)
            nodes, edges, w = CI.postprocess(coords, raw_edges, cfg, log=lambda *a: None)
            t_post = time.time() - t1
            lrows = []
            for L in MIN_TRACK_LENS:
                n2, e2, _ = CI.prune_short_tracks(nodes, edges, w, L)
                sc = score_official(n2, e2, gts[clip])
                sc.update(det=det_name, post=post_name, min_len=L, config=f"{det_name}/{post_name}/L{L}", sample=clip, t_post=round(t_post, 1))
                lrows.append(sc)
            rows += lrows
            best = max(lrows, key=lambda r: r["score"]); l1 = min(lrows, key=lambda r: r["min_len"])
            line.append(f"    {post_name:<13} {t_post:5.0f}s  L{l1['min_len']} score={l1['score']:.3f} n_pred/est={l1['n_pred']}/{l1['est']}"
                        f"  best L{best['min_len']} score={best['score']:.3f} (tp={best['edge_tp']} fp={best['edge_fp']} fn={best['edge_fn']} n_pred={best['n_pred']} rec={best['node_recall']:.3f})")
        print("\\n".join(line), f"| elapsed {time.time()-T_START:.0f}s", flush=True)
    if not stopped:
        done = [r for r in rows if r["det"] == det_name]
        for (post_name, L), grp in pd.DataFrame(done).groupby(["post", "min_len"]):
            if len(grp) == len(cv_zarrs):
                ma = micro_average(grp.to_dict("records"))
                print(f"  ==> [{det_name}/{post_name}/L{L}] micro score={ma['score']:.4f} adjJ={ma['adj_edge_J']:.4f} divJ={ma['div_J']:.3f}")

df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "cv_rows.csv", index=False)
summ = []
for cfg_name, grp in df.groupby("config"):
    if len(grp) == len(cv_zarrs):                  # only complete configs are comparable
        ma = micro_average(grp.to_dict("records")); ma.update(config=cfg_name, n_pred=int(grp.n_pred.sum()), est=int(grp.est.sum()),
                                                            node_recall=round(grp.node_recall.mean(), 4)); summ.append(ma)
summ = pd.DataFrame(summ).sort_values("score", ascending=False).set_index("config")
summ.to_csv(OUT_DIR / "cv_summary.csv")
display(summ.head(25))
if len(summ):
    best_cfg = summ.index[0]; d, p, L = best_cfg.split("/")
    best = dict(det=DET_CONFIGS[d], post=POST_CONFIGS[p], min_track_len=int(L[1:]), det_tta=DET_TTA, cv_score=float(summ.iloc[0]["score"]))
    (OUT_DIR / "best_config.json").write_text(json.dumps(best, indent=1))
    print("BEST:", best_cfg, "->", best)
    display(df[df.config == best_cfg].set_index("sample")[["score", "edge_tp", "edge_fp", "edge_fn", "node_recall", "n_pred", "est"]])
    ref = "t95_p3/ilp_d14/L1"
    if ref in summ.index: print(f"reference (v14 setting) {ref}: {summ.loc[ref, 'score']:.4f}")
print(f"total {(time.time()-T_START)/3600:.2f} h | raw cache saved to {RAW_OUT} (attach this Output next time to skip the network)")
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out_name = sys.argv[1] if len(sys.argv) > 1 else "biohub_v15_cellmot_cv_tune.ipynb"
Path(out_name).write_text(nbf.writes(nb))
print("written", out_name, len(cells), "cells")
