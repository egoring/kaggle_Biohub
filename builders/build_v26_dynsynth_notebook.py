"""v26 = public 'multi-paradigm dynamic synthesis' notebook (harmonic-fusion + det 0.96 + ILP division weight 0.6 +
safe-div sister min/permanence + DeepCenter-score-weighted candidate ranking + direction-consistency bonus in motion
relink + wider hold-out sweep) with our VARIANT switch.  Source cells verbatim except guard raise->print and
TTA env assignments -> setdefault.
"""
import json, sys
from pathlib import Path

SRC = Path("public_notebooks/7c27b875-biohub-cell-tracking-lineage-submission.ipynb")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "biohub_v26_dynsynth_variants.ipynb")

nb = json.load(open(SRC))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for c in cells:
    c["outputs"] = []; c["execution_count"] = None; c["source"] = "".join(c["source"])

VARIANT_CELL = r'''
# ---------------------------------------------------------------- OURS: variant switch (edit VARIANT only)
# S0 = the public notebook exactly (det 0.96, ILP div 0.6, code changes, hold-out sweep ON -> picks a relink knob)
# S1 = S0 with the hold-out sweep OFF (fixed config; isolates the code changes + ILP div 0.6)   <- compare with Q2
# S2 = S1 with ILP division weight back to 1.2 (isolates the code changes alone)
# S3 = S1 + MOTION_RELINK_RELAXED_UM 8.0 (the knob its own sweep selected), no sweep
VARIANT = "S1"
VARIANTS = {
    "S0": {},
    "S1": {"BIOHUB_VALIDATOR_ENABLE": "0"},
    "S2": {"BIOHUB_VALIDATOR_ENABLE": "0", "BIOHUB_ILP_DIVISION_WEIGHT": "1.2"},
    "S3": {"BIOHUB_VALIDATOR_ENABLE": "0", "BIOHUB_MOTION_RELINK_RELAXED_UM": "8.0"},
}
for _k, _v in VARIANTS[VARIANT].items():
    os.environ[_k] = _v
print("VARIANT:", VARIANT, VARIANTS[VARIANT])
print("  det_threshold      :", os.environ.get("BIOHUB_DET_THRESHOLD"))
print("  ILP division weight:", os.environ.get("BIOHUB_ILP_DIVISION_WEIGHT"))
print("  validator/sweep    :", os.environ.get("BIOHUB_VALIDATOR_ENABLE", "1"))
print("  relink relaxed um  :", os.environ.get("BIOHUB_MOTION_RELINK_RELAXED_UM", "10.0 (default)"))
print("  streamline vel w   :", os.environ.get("BIOHUB_STREAMLINE_VELOCITY_WEIGHT"), "| permanence:", os.environ.get("BIOHUB_SAFE_DIV_REQUIRE_PERMANENCE"))
'''

g = cells[1]["source"]
old = 'if _drift:\n    raise RuntimeError(\n        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)\n    )'
assert g.count(old) == 1, "guard block not found"
cells[1]["source"] = g.replace(old, 'if _drift:\n    print("NOTE (ours): configuration differs from the guard table on purpose:", _guard_json.dumps(_drift, sort_keys=True))')

p = cells[4]["source"]
for a, b in {
    "os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'": "os.environ.setdefault('BIOHUB_EDGE_FEATURE_TTA', '1')",
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "1")',
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "0.75")',
}.items():
    assert p.count(a) == 1, a; p = p.replace(a, b)
cells[4]["source"] = p

md = {"cell_type": "markdown", "metadata": {}, "source": "# Biohub · v26 — public 'dynamic synthesis' line (det 0.96, ILP div 0.6, direction bonus, permanence, DC-weighted ranking)\n\nEdit `VARIANT` in cell 2 only. Inputs: competition + pilkwang ×3."}
new = [md, cells[0], {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": VARIANT_CELL.strip("\n")}] + cells[1:]
json.dump({"cells": new, "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}, open(OUT, "w"), indent=1)
print("wrote", OUT, "cells:", len(new))
