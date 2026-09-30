"""Builds biohub_v18_public50ep_cv_submit.ipynb — plug the public 50-epoch official-baseline weights into OUR pipeline: CV (TTA, several post settings, divisions reported) -> best -> submission."""
import base64
import sys
from pathlib import Path

import nbformat as nbf
from nb_common import LOCATE_CELL, PIP_CELL

def _b64(p): return base64.b64encode(Path(p).read_bytes()).decode()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Biohub · v18 — public 50-epoch baseline weights in our pipeline: CV → best post-processing → submission

**Inputs**: competition data + `biohub-prep-bundle-v2` Output (wheels + repo) + the public dataset
**`pilkwang/biohub-tracking-support-pack-50ep-v1`** (official baseline trained 50 epochs; `WEIGHTS_DIR_HINT` picks it) —
the v13 Output may stay attached (it is ignored when the hint matches). Also saves `raw_cache/` (TTA, det 0.99) of the 12 CV clips for CPU follow-ups
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

RUN_CV            = True            # CV first (12 hold-out clips, TTA): picks the best post-processing automatically
WEIGHTS_DIR_HINT  = "support-pack"  # substring of the checkpoint folder to use (public 50-epoch weights); "" = latest attached
SAVE_RAW          = True            # cache the CV clips' network output in OUT_DIR/raw_cache (CPU tuning later)
N_CV_SAMPLES      = 12
MAX_TEST_SAMPLES  = None            # None = all test clips; 1 = quick debug; 0 = CV only, no submission (e.g. CPU run)
DET_TTA           = True            # 4-flip test-time augmentation for detection (4x U-Net cost; set False on CPU)
WEIGHTS_PREFER    = "edge_predictor_best.pth"   # or "edge_predictor_last.pth"
DET_CONFIG = dict(det_threshold=0.99, pool_kernel_um=3.0)          # network stage (run once per CV clip, cached)
POST_CONFIGS = {                                                    # linking stage (ILP re-solved per config)
    "ilp_div10": dict(use_ilp=True, ilp_appearance_weight=0.0, ilp_disappearance_weight=1.4, ilp_division_weight=1.0),
    "ilp_div05": dict(use_ilp=True, ilp_appearance_weight=0.0, ilp_disappearance_weight=1.4, ilp_division_weight=0.5),
    "ilp_div00": dict(use_ilp=True, ilp_appearance_weight=0.0, ilp_disappearance_weight=1.4, ilp_division_weight=0.0),
}
MIN_TRACK_LENS    = [1, 3, 5, 10]                                    # short-track pruning (free, applied to each ILP solution)
FALLBACK_CONFIG   = {**DET_CONFIG, **POST_CONFIGS["ilp_div10"], "min_track_len": 5}   # used if CV is off / incomplete
CV_CONFIGS        = {}              # filled by the CV cell; FINAL_CONFIG = best CV entry
FINAL_CONFIG      = "fallback"
USE_BEST_CONFIG   = False
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
assert cands, f"no {WEIGHTS_PREFER} found in inputs: attach the public support-pack dataset (or the v13 Output)"
print("checkpoints found:", [str(c) for c in cands])
hinted = [c for c in cands if WEIGHTS_DIR_HINT and WEIGHTS_DIR_HINT in str(c)]
if WEIGHTS_DIR_HINT and not hinted:
    print(f"!! no checkpoint path contains '{WEIGHTS_DIR_HINT}' -> falling back to the most recent one")
cands = hinted or sorted(cands, key=lambda p: p.stat().st_mtime)
WEIGHTS = cands[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 4); print("CPU mode, threads:", torch.get_num_threads(), "| det_tta:", DET_TTA)
try:
    model, WINDOW, DOWNSAMPLE = CI.load_model(WEIGHTS, device)
except RuntimeError as e:                       # architecture mismatch between the public weights and our repo copy?
    msg = str(e); print("!! load_state_dict failed:", msg[:1500])
    raise SystemExit("weights/repo mismatch — send this log")
print("config.json next to weights:", (WEIGHTS.parent / "config.json").read_text() if (WEIGHTS.parent / "config.json").exists() else "(none -> defaults)")
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
code('''# ------------------------------------------------------------------ 4. CV on the 12 hold-out clips: network once per clip, several post settings
import pandas as pd
if RUN_CV and (COMP_DIR / "train").is_dir():
    cv_zarrs = sorted(p for p in (COMP_DIR / "train").iterdir() if p.name.endswith(".zarr"))
    cv_zarrs = [p for p in cv_zarrs if (p.parent / (p.name[:-5] + ".geff")).exists()]
    step = max(1, len(cv_zarrs) // max(1, N_CV_SAMPLES))
    cv_zarrs = cv_zarrs[1::step][:N_CV_SAMPLES]
    print("CV clips:", [p.name[:-5] for p in cv_zarrs])
    gts = {zp.name[:-5]: read_geff(zp.parent / (zp.name[:-5] + ".geff")) for zp in cv_zarrs}
    print("GT divisions per clip:", {k: sum(1 for s_, ch in __import__("collections").Counter(gts[k]["edges"][:, 0].tolist()).items() if ch >= 2) for k in gts})
    RAW_OUT = OUT_DIR / "raw_cache" / "t99_tta"
    rows = []
    for zp in cv_zarrs:
        clip = zp.name[:-5]
        if time.time() - T_START > TIME_BUDGET_SEC * 0.35:
            print("!! CV time budget exhausted; remaining clips skipped"); break
        t0 = time.time()
        coords, raw_edges = CI.predict_raw(model, zp, device, CI.make_cfg(det_tta=DET_TTA, **DET_CONFIG), WINDOW, DOWNSAMPLE)
        t_net = time.time() - t0
        if SAVE_RAW: CI.save_raw(RAW_OUT / f"{clip}.npz", coords, raw_edges)
        line = [f"[{clip}] {len(coords)} det, {len(raw_edges)} cand. edges (net {t_net:.0f}s)"]
        for pname, pover in POST_CONFIGS.items():
            t1 = time.time()
            nodes, edges, w = CI.postprocess(coords, raw_edges, CI.make_cfg(det_tta=DET_TTA, **DET_CONFIG, **pover), log=lambda *a: None)
            t_post = time.time() - t1
            for L in MIN_TRACK_LENS:
                n2, e2, _ = CI.prune_short_tracks(nodes, edges, w, L)
                sc = score_official(n2, e2, gts[clip]); sc.update(config=f"{pname}/L{L}", post=pname, min_len=L, sample=clip); rows.append(sc)
                if L == 5:
                    line.append(f"    {pname}/L5 ({t_post:.0f}s): score={sc['score']:.3f} adjJ={sc['adj_edge_J']:.3f} (tp={sc['edge_tp']} fp={sc['edge_fp']} fn={sc['edge_fn']}) "
                                f"rec={sc['node_recall']:.3f} n_pred/est={sc['n_pred']}/{sc['est']} | div tp/fp/fn={sc['div_tp']}/{sc['div_fp']}/{sc['div_fn']} gt_div={sc['gt_div']}")
        print("\\n".join(line), f"| elapsed {time.time()-T_START:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "cv_rows.csv", index=False)
    n_clips = df["sample"].nunique()
    summ = []
    for cfg_name, grp in df.groupby("config"):
        if len(grp) == n_clips:
            ma = micro_average(grp.to_dict("records")); ma.update(config=cfg_name, div_tp=int(grp.div_tp.sum()), div_fp=int(grp.div_fp.sum()), div_fn=int(grp.div_fn.sum()),
                                                            n_pred=int(grp.n_pred.sum()), est=int(grp.est.sum())); summ.append(ma)
    summ = pd.DataFrame(summ).sort_values("score", ascending=False).set_index("config"); summ.to_csv(OUT_DIR / "cv_summary.csv")
    display(summ)
    best = summ.index[0]; pname, L = best.split("/")
    CV_CONFIGS[best] = {**DET_CONFIG, **POST_CONFIGS[pname], "min_track_len": int(L[1:])}
    FINAL_CONFIG = best
    (OUT_DIR / "best_config.json").write_text(json.dumps(dict(config=best, over=CV_CONFIGS[best], cv_score=float(summ.iloc[0]["score"]), n_clips=int(n_clips)), indent=1))
    print(f"==> best: {best} micro score={summ.iloc[0]['score']:.4f} adjJ={summ.iloc[0]['adj_edge_J']:.4f} divJ={summ.iloc[0]['div_J']:.3f} on {n_clips} clips")
if FINAL_CONFIG == "fallback":
    CV_CONFIGS["fallback"] = FALLBACK_CONFIG
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
out_name = sys.argv[1] if len(sys.argv) > 1 else "biohub_v18_public50ep_cv_submit.ipynb"
Path(out_name).write_text(nbf.writes(nb))
print("written", out_name, len(cells), "cells")
