"""v24 = public 'harmonic-fusion' notebook (0.939 base + edge-feature TTA + DeepCenter TTA + hold-out post-process sweep)
with our knobs layered on top via a VARIANT cell.  Source notebook cells kept verbatim except:
  * config-drift guard  raise -> print   (we change the detection threshold on purpose)
  * hard-coded TTA env assignments in the patch cell  os.environ[...] = ...  ->  os.environ.setdefault(...)
    so the VARIANT cell can switch each TTA piece off.
"""
import json, sys, copy
from pathlib import Path

SRC = Path("public_notebooks/7c6cee3f-biohub-harmonic-fusion.ipynb")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "biohub_v24_pub_tta_variants.ipynb")

nb = json.load(open(SRC))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for c in cells:                       # strip stored outputs / execution counts
    c["outputs"] = []; c["execution_count"] = None
    c["source"] = "".join(c["source"])

VARIANT_CELL = r'''
# ---------------------------------------------------------------- OURS: variant switch (edit VARIANT only)
# Q0 = the public notebook exactly (det 0.965, edge-feature TTA on, DeepCenter TTA on, hold-out sweep on)
# Q1 = Q0 + our detection threshold 0.96 (the +0.001 knob from v21 P0)        <- main candidate
# Q2 = Q1 but hold-out sweep OFF (fixed published post-process; ~20 min faster, no train-leak selection)
# Q3 = Q2 but DeepCenter TTA OFF (isolates the edge-feature TTA effect)
# Q4 = P0 (0.942) + MOTION_RELINK_TIGHT_UM 5.5 only, all TTA OFF, sweep OFF  (pure knob test of the sweep winner)
VARIANT = "Q1"
VARIANTS = {
    "Q0": {},
    "Q1": {"BIOHUB_DET_THRESHOLD": "0.96"},
    "Q2": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_VALIDATOR_ENABLE": "0"},
    "Q3": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_VALIDATOR_ENABLE": "0", "BIOHUB_DEEPCENTER_TTA": "0"},
    "Q4": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_VALIDATOR_ENABLE": "0", "BIOHUB_DEEPCENTER_TTA": "0",
           "BIOHUB_EDGE_FEATURE_TTA": "0", "BIOHUB_SECONDARY_EDGE_FEATURE_TTA": "0",
           "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5"},
    # Q5/Q6: ILP division weight (never probed in the P-series; the 'dynamic synthesis' public notebook uses 0.6).
    #        Lower = the linker itself proposes more divisions.  Sweep OFF so the effect is isolated (compare with Q2).
    "Q5": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_VALIDATOR_ENABLE": "0", "BIOHUB_ILP_DIVISION_WEIGHT": "0.6"},
    "Q6": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_VALIDATOR_ENABLE": "0", "BIOHUB_ILP_DIVISION_WEIGHT": "0.9"},
    # Q7: Q2 with the secondary-seed edge-feature TTA at full weight 1.0 (the 'dctta020-sectta1' public variant; 0.75 is the default).
    "Q7": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_VALIDATOR_ENABLE": "0", "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT": "1.0"},
    # QB: Q1 (LB 0.947) reproduced without the 20-min hold-out sweep: the sweep selected tight55, so pin it.  Base for combos.
    "QB": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_VALIDATOR_ENABLE": "0", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5"},
}
for _k, _v in VARIANTS[VARIANT].items():
    os.environ[_k] = _v
print("VARIANT:", VARIANT, VARIANTS[VARIANT])
print("  det_threshold      :", os.environ.get("BIOHUB_DET_THRESHOLD"))
print("  validator/sweep    :", os.environ.get("BIOHUB_VALIDATOR_ENABLE", "1"))
print("  deepcenter TTA     :", os.environ.get("BIOHUB_DEEPCENTER_TTA"))
print("  edge-feature TTA   :", os.environ.get("BIOHUB_EDGE_FEATURE_TTA", "(set later: 1)"),
      "| secondary:", os.environ.get("BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "(set later: 1)"))
print("  motion relink tight:", os.environ.get("BIOHUB_MOTION_RELINK_TIGHT_UM", "6.0 (default)"))
print("  ILP division weight:", os.environ.get("BIOHUB_ILP_DIVISION_WEIGHT"))
print("  secondary TTA weight:", os.environ.get("BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "(set later: 0.75)"))
'''

# --- 1. guard cell: raise -> print
g = cells[1]["source"]
old = 'if _drift:\n    raise RuntimeError(\n        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)\n    )'
assert g.count(old) == 1, "guard block not found"
cells[1]["source"] = g.replace(old, 'if _drift:\n    print("NOTE (ours): configuration differs from the published 0.939 line on purpose:", _guard_json.dumps(_drift, sort_keys=True))')

# --- 2. patch cell: hard-coded TTA env -> setdefault
p = cells[4]["source"]
repl = {
    "os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'": "os.environ.setdefault('BIOHUB_EDGE_FEATURE_TTA', '1')",
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "1")',
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "0.75")',
}
for a, b in repl.items():
    assert p.count(a) == 1, a
    p = p.replace(a, b)
cells[4]["source"] = p

# --- 3. insert VARIANT cell after the config cell (cells[0]); leave everything else verbatim
new_cells = [cells[0], {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": VARIANT_CELL.strip("\n")}] + cells[1:]
md = {"cell_type": "markdown", "metadata": {}, "source": "# Biohub · v24 — public harmonic-fusion (edge-feature TTA + DeepCenter TTA + hold-out sweep) with our det-threshold knob\n\nEdit `VARIANT` in cell 2 only. Inputs: competition data + the three pilkwang datasets (support pack, seed 314159, DeepCenter)."}
nb_out = {"cells": [md] + new_cells, "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}
for c in nb_out["cells"]:
    c["source"] = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
json.dump(nb_out, open(OUT, "w"), indent=1)
print("wrote", OUT, "cells:", len(nb_out["cells"]))
