import json
from pathlib import Path

import nbformat as nbf
import sys
from nb_common import LOCATE_CELL, PIP_CELL

pipeline_src = Path("pipeline.py").read_text()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Biohub - Cell Tracking During Development · Trackastra inference notebook

**Pipeline** (offline, GPU or CPU, < 12 h):

1. locate competition data + attached Kaggle Models (`trackastra` artifact with wheels & `models/ctc/model.pt`)
2. offline `pip install --no-index` from the bundled wheels (never touches the preinstalled torch/numpy)
3. per-frame 3D blob detection (anisotropic DoG + local maxima, voxel scale z 1.625 / y,x 0.40625 µm)
4. spherical pseudo-masks → **Trackastra** (pretrained CTC 3D transformer, greedy linking with divisions)
   → fallback: physical-distance linear-assignment linker if Trackastra is unavailable
5. optional sanity-check on a few `train/` samples against sparse `.geff` GT
6. write `submission.csv`

Attach as inputs: the competition dataset, Model *biohubcelltrack-trackastra-artifact-and-packages*
(and optionally *buddy-scikitlearn-biohub* – not used for inference). Internet **off**.
""")

code('''# ------------------------------------------------------------------ 0. settings
import os, sys, glob, time, json, subprocess, importlib
from pathlib import Path
os.environ["TQDM_DISABLE"] = "1"   # quiet progress bars in the log

VALIDATE_ON_TRAIN = True      # run sanity-check on a few train samples (costs time; set False for final run)
N_VAL_SAMPLES     = 2
MAX_TEST_SAMPLES  = None      # None = all; small int for a quick debug run
AUTO_CALIBRATE    = True      # sweep detection threshold on N_CAL train clips to match estimated cell counts
N_CAL_SAMPLES     = 3
RUN_CV            = False     # official-style metric on N_CV train clips for each config in CV_CONFIGS (no submission needed)
N_CV_SAMPLES      = 12
CV_CONFIGS = {                # round 4: 12 clips; dense-clip improvements on top of adaptive radius + merge
    "adapt_a26_merge":        {"adaptive_radius": {"a": 26.5, "rmin": 3.0, "rmax": 6.0}, "merge_min_dist_um": 6.0},
    "adapt_a24_merge":        {"adaptive_radius": {"a": 24.0, "rmin": 2.75, "rmax": 6.0}, "merge_min_dist_um": 6.0},
    "adapt_a26_merge_mask06": {"adaptive_radius": {"a": 26.5, "rmin": 3.0, "rmax": 6.0}, "merge_min_dist_um": 6.0, "mask_scale": 0.6},
    "adapt_a26_merge_k1":     {"adaptive_radius": {"a": 26.5, "rmin": 3.0, "rmax": 6.0}, "merge_min_dist_um": 6.0, "thresh_k": 1.0},
}
FINAL_CONFIG      = "adapt_a26_merge"    # which CV_CONFIGS entry to use for the test run (after you looked at the CV table)
TIME_BUDGET_SEC   = 10.5 * 3600   # hard stop; whatever is done gets written
MODEL_PREFER      = "ctc_ft"      # Trackastra weights folder to prefer when several are attached: "ctc_ft" (fine-tuned) or "ctc" (pretrained)

KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
OUT_DIR      = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
T_START = time.time()
print("input root:", KAGGLE_INPUT, "| output:", OUT_DIR)
''')

code(LOCATE_CELL)

code(PIP_CELL)

code("# ------------------------------------------------------------------ 3. pipeline code\n" + pipeline_src)

code('''# ------------------------------------------------------------------ 4. load Trackastra model
model = None
if TRACKASTRA_OK and MODEL_DIR is not None:
    try:
        from trackastra.model import Trackastra
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = Trackastra.from_folder(Path(MODEL_DIR), device=dev)
        print("Trackastra loaded on", dev, "| features:", model.train_args.get("features"),
              "| window:", model.train_args.get("window"), "| weights folder:", Path(MODEL_DIR).name,
              "| fine-tuned:", model.train_args.get("finetuned_on", "no (pretrained ctc)"))
        if (Path(MODEL_DIR) / "SELECTED.txt").exists():
            print("fine-tune selection:", (Path(MODEL_DIR) / "SELECTED.txt").read_text())
    except Exception as e:
        print("!! could not load Trackastra:", type(e).__name__, e)
        model = None
if model is None:
    print("-> will use the nearest-neighbour fallback linker")
ILP_OK = _importable("motile") and _importable("ilpy")
print("ILP solver available:", ILP_OK, "| requested mode:", CFG["trackastra_mode"])
if CFG["trackastra_mode"] == "ilp" and not ILP_OK:
    CFG["trackastra_mode"] = "greedy"; print("-> motile/ilpy missing, using greedy")
''')

code('''# ------------------------------------------------------------------ 4b. (optional) calibrate detection threshold on train
if AUTO_CALIBRATE and (COMP_DIR / "train").is_dir():
    cal_zarrs = sorted(p for p in (COMP_DIR / "train").iterdir() if p.name.endswith(".zarr"))
    cal_zarrs = [p for p in cal_zarrs if (p.parent / (p.name[:-5] + ".geff")).exists()]
    # spread picks across the list so several embryos are represented
    step = max(1, len(cal_zarrs) // max(1, N_CAL_SAMPLES))
    cal_zarrs = cal_zarrs[::step][:N_CAL_SAMPLES]
    print("calibrating on:", [p.name for p in cal_zarrs])
    best_k, cal_rows = calibrate_thresh_k(cal_zarrs, CFG)
    import pandas as pd
    display(pd.DataFrame(cal_rows).pivot_table(index="thresh_k", columns="sample", values=["ratio", "node_recall"]).round(3))
    if best_k is not None:
        CFG["thresh_k"] = best_k
    print("CFG thresh_k ->", CFG["thresh_k"], f"| calibration took {time.time()-T_START:.0f}s so far")
''')

code('''# ------------------------------------------------------------------ 4c. (optional) local CV with the official-style metric
if RUN_CV and (COMP_DIR / "train").is_dir():
    cv_zarrs = sorted(p for p in (COMP_DIR / "train").iterdir() if p.name.endswith(".zarr"))
    cv_zarrs = [p for p in cv_zarrs if (p.parent / (p.name[:-5] + ".geff")).exists()]
    step = max(1, len(cv_zarrs) // max(1, N_CV_SAMPLES))
    cv_zarrs = cv_zarrs[1::step][:N_CV_SAMPLES]          # offset 1 -> different clips than calibration
    print("CV clips:", [p.name for p in cv_zarrs])
    _configs = {}
    for k_, v_ in CV_CONFIGS.items():
        _configs[k_] = {kk: vv for kk, vv in v_.items() if not kk.startswith("_")}
    cv_rows, cv_summary = [], []
    for name, over in _configs.items():
        use_model = None if CV_CONFIGS[name].get("_no_model") else model
        r_, s_ = run_cv(cv_zarrs, use_model, {name: over})
        cv_rows += r_; cv_summary += s_
    import pandas as pd
    display(pd.DataFrame(cv_summary).set_index("config"))
    display(pd.DataFrame(cv_rows).pivot_table(index="sample", columns="config", values="score").round(3))
    print(f"CV took {time.time()-T_START:.0f}s so far")

# apply the chosen config for the test run
_final = {k: v for k, v in CV_CONFIGS.get(FINAL_CONFIG, {}).items() if not k.startswith("_")}
CFG.update(_final)
if CV_CONFIGS.get(FINAL_CONFIG, {}).get("_no_model"):
    model = None
print("FINAL_CONFIG:", FINAL_CONFIG, "->", _final, "| CFG:", {k: CFG[k] for k in ("thresh_k", "cell_radius_um", "adaptive_radius", "merge_min_dist_um", "trackastra_mode", "div_min_weight")})
''')

code('''# ------------------------------------------------------------------ 5. (optional) sanity check on train samples
if VALIDATE_ON_TRAIN and (COMP_DIR / "train").is_dir():
    train_zarrs = sorted(p for p in (COMP_DIR / "train").iterdir() if p.name.endswith(".zarr"))
    train_zarrs = [p for p in train_zarrs if (p.parent / (p.name[:-5] + ".geff")).exists()][:N_VAL_SAMPLES]
    val_report = []
    for zp in train_zarrs:
        rows, info = process_sample(zp, model, CFG)
        gt = read_geff(zp.parent / (zp.name[:-5] + ".geff"))
        nodes = [(r[3], r[4], r[5], r[6], r[2]) for r in rows if r[1] == "node"]
        edges = [(r[7], r[8]) for r in rows if r[1] == "edge"]
        ev = evaluate_against_geff(nodes, edges, gt)
        ev["sample"] = info["name"]
        val_report.append(ev)
        print("   ", json.dumps(ev, default=str))
    import pandas as pd
    if val_report:
        display(pd.DataFrame(val_report).set_index("sample").T)
    print(f"validation done in {time.time()-T_START:.0f}s")
''')

code('''# ------------------------------------------------------------------ 6. inference on test -> submission.csv
all_rows, summary = [], []
zarrs = TEST_ZARRS if MAX_TEST_SAMPLES is None else TEST_ZARRS[:MAX_TEST_SAMPLES]
for k, zp in enumerate(zarrs):
    elapsed = time.time() - T_START
    if elapsed > TIME_BUDGET_SEC:
        print(f"!! time budget exhausted after {k} samples; remaining samples get empty predictions")
        all_rows.append((zp.name[:-5], "node", 1, 0, 0, 0, 0, -1, -1))
        continue
    rows, info = process_sample(zp, model, CFG)
    if not rows:  # never leave a dataset without rows
        rows = [(info["name"], "node", 1, 0, 0, 0, 0, -1, -1)]
    all_rows += rows
    summary.append(info)
    print(f"    [{k+1}/{len(zarrs)}] done, elapsed {time.time()-T_START:.0f}s")

sub_path = OUT_DIR / "submission.csv"
df = write_submission(all_rows, sub_path)
print(df.head())
print(df.row_type.value_counts().to_dict())
print("written:", sub_path, f"{sub_path.stat().st_size/1e6:.1f} MB, {len(df)} rows")
''')

code('''# ------------------------------------------------------------------ 7. format self-check
import pandas as pd
sample = pd.read_csv(COMP_DIR / "sample_submission.csv")
sub = pd.read_csv(sub_path)
assert list(sub.columns) == list(sample.columns), (list(sub.columns), list(sample.columns))
assert sub["id"].is_unique and (sub["id"] == range(len(sub))).all()
assert set(sub.row_type.unique()) <= {"node", "edge"}
nodes = sub[sub.row_type == "node"]; edges = sub[sub.row_type == "edge"]
assert not nodes.duplicated(["dataset", "node_id"]).any(), "duplicate node ids within a dataset"
nid = set(zip(nodes.dataset, nodes.node_id))
bad = [(d, s, t) for d, s, t in zip(edges.dataset, edges.source_id, edges.target_id) if (d, s) not in nid or (d, t) not in nid]
assert not bad, f"{len(bad)} edges reference unknown nodes, e.g. {bad[:3]}"
tested = set(p.name[:-5] for p in zarrs)
assert tested <= set(sub.dataset.unique()), "some test datasets have no rows"
for c in ["node_id", "t", "z", "y", "x", "source_id", "target_id"]:
    assert np.issubdtype(sub[c].dtype, np.integer), c
print("format OK ✓ |", len(nodes), "nodes,", len(edges), "edges over", sub.dataset.nunique(), "datasets |",
      f"total {time.time()-T_START:.0f}s")
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out_name = sys.argv[1] if len(sys.argv) > 1 else "biohub_trackastra_inference.ipynb"
Path(out_name).write_text(nbf.writes(nb))
print("notebook written, cells:", len(cells))
