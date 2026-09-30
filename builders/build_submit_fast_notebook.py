"""Builds biohub_v17_cellmot_submit_fast.ipynb — submission-only run: GPU net + ILP post-processing in parallel worker processes."""
import base64
import sys
from pathlib import Path

import nbformat as nbf
from nb_common import LOCATE_CELL, PIP_CELL

def _b64(p): return base64.b64encode(Path(p).read_bytes()).decode()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Biohub · v17 — submission (fast): our self-trained baseline, ILP post-processing in parallel workers

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

RUN_CV            = False           # submission run: skip CV (v16 already measured it: t99/L5 = 0.8549 with TTA)
N_CV_SAMPLES      = 12
MAX_TEST_SAMPLES  = None            # None = all test clips; 1 = quick debug; 0 = CV only, no submission (e.g. CPU run)
DET_TTA           = True            # 4-flip test-time augmentation for detection (4x U-Net cost; set False on CPU)
WEIGHTS_PREFER    = "edge_predictor_best.pth"   # or "edge_predictor_last.pth"
CV_CONFIGS = {   # overrides: det_threshold, pool_kernel_um, use_ilp, ilp_* weights, edge_threshold, min_track_len
    "ilp_t99_L5":      dict(det_threshold=0.99, pool_kernel_um=3.0, use_ilp=True, ilp_disappearance_weight=1.4, ilp_division_weight=1.0, min_track_len=5),   # v16 CV (TTA) 0.8549
    "ilp_t99_L10":     dict(det_threshold=0.99, pool_kernel_um=3.0, use_ilp=True, ilp_disappearance_weight=1.4, ilp_division_weight=1.0, min_track_len=10),  # v16 CV (TTA) 0.8508
}
FINAL_CONFIG      = "ilp_t99_L5"
USE_BEST_CONFIG   = False           # v15's best_config.json was tuned WITHOUT TTA; with TTA L5 beat L10 (v16 log) -> fixed choice above
POST_WORKERS      = 3               # ILP (single-threaded, up to ~5 min on dense clips) runs in worker processes while the GPU does the next clip
TIME_BUDGET_SEC   = 11.0 * 3600   # hard wall for the whole notebook; clips not finished by then get a placeholder node
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
code('''# ------------------------------------------------------------------ 5. inference on test -> submission.csv (net on GPU, ILP in worker processes)
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
import multiprocessing as mp
all_rows, summary = [], []
zarrs = TEST_ZARRS if MAX_TEST_SAMPLES is None else TEST_ZARRS[:MAX_TEST_SAMPLES]
sub_path = OUT_DIR / "submission.csv"
over = CV_CONFIGS[FINAL_CONFIG]
placeholder = lambda name: (name, "node", 1, 0, 0, 0, 0, -1, -1)
remaining = lambda: TIME_BUDGET_SEC - (time.time() - T_START)
ex = ProcessPoolExecutor(POST_WORKERS, mp_context=mp.get_context("spawn"), initializer=CI.init_worker, initargs=(str(CODE_DIR), str(REPO_DIR)))
futs = {}                      # name -> future (submitted in order)
t_net_sum = 0.0
try:
    for k, zp in enumerate(zarrs):
        name = zp.name[:-5]
        if remaining() < 20 * 60:             # keep 20 min to drain the workers + write the csv
            print(f"!! time budget: stopping the network after {k} clips; the rest get a placeholder node"); break
        t0 = time.time()
        cfg = CI.make_cfg(det_tta=DET_TTA, **over)
        coords, raw_edges = CI.predict_raw(model, zp, device, cfg, WINDOW, DOWNSAMPLE)
        t_net = time.time() - t0; t_net_sum += t_net
        futs[name] = ex.submit(CI.post_task, (coords, raw_edges, over, DET_TTA))
        n_done = sum(f.done() for f in futs.values())
        print(f"[{k+1}/{len(zarrs)}] {name}: net {t_net:.0f}s ({len(coords)} det, {len(raw_edges)} cand. edges) | post done {n_done}/{len(futs)} | elapsed {time.time()-T_START:.0f}s", flush=True)
    for name in [zp.name[:-5] for zp in zarrs]:
        f = futs.get(name)
        rows = None
        if f is not None:
            try:
                nodes, edges, t_post = f.result(timeout=max(1.0, remaining() - 5 * 60))
                rows = rows_for_sample(name, nodes, edges) or None
                summary.append(dict(name=name, n_nodes=len(nodes), n_edges=len(edges), t_post=round(t_post)))
            except FutTimeout:
                print(f"!! {name}: post-processing not finished within the budget -> placeholder")
            except Exception as e:
                print(f"!! {name}: post-processing failed ({type(e).__name__}: {e}) -> placeholder")
        all_rows += rows or [placeholder(name)]
finally:
    ex.shutdown(wait=False, cancel_futures=True)
if zarrs:
    df = write_submission(all_rows, sub_path)
    import pandas as pd
    sm = pd.DataFrame(summary)
    if len(sm): print(f"post-processing: {len(sm)} clips, mean {sm.t_post.mean():.0f}s, max {sm.t_post.max()}s | net mean {t_net_sum/max(1,len(futs)):.0f}s")
    print(df.row_type.value_counts().to_dict(), "| written:", sub_path, f"{sub_path.stat().st_size/1e6:.1f} MB | total {time.time()-T_START:.0f}s")
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
out_name = sys.argv[1] if len(sys.argv) > 1 else "biohub_v17_cellmot_submit_fast.ipynb"
Path(out_name).write_text(nbf.writes(nb))
print("written", out_name, len(cells), "cells")
