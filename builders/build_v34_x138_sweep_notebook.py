"""v33 = public 'biohub-x138' notebook (harmonic base + flow-prior motion relink + re-admitted low-score detections + gap fill
from sub-threshold peaks + ILP timeout / runtime guards) with our DivNet safe-division gate layered on via a VARIANT cell.
X0 = x138 exactly (to learn its LB).  X1 = x138 + our classifier gate (R14 config).  Inputs: competition + pilkwang x3 (+ v28e Output for X1)."""
import json, sys
from pathlib import Path
SC = Path("../src")
SRC = Path("public_notebooks/ab30ed16-biohub-x138.ipynb")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "biohub_v34_x138_sweep60.ipynb")
nb = json.load(open(SRC))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for c in cells:
    c["outputs"] = []; c["execution_count"] = None; c["source"] = "".join(c["source"])

VARIANT_CELL = r'''
# ---------------------------------------------------------------- OURS: v34 = x138 + our v28e gate (X1 config) + 60-clip hold-out sweep of the x138 knobs
for _k, _v in {"BIOHUB_DIVNET_SAFE_DIV_GATE": "1", "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80",
               "BIOHUB_DIVNET_REQUIRE_OURS": "1", "BIOHUB_DIVNET_CKPT_HINT": "v28e",
               "BIOHUB_VALIDATOR_ENABLE": "1", "BIOHUB_VALIDATOR_N_PER_TYPE": "30",     # 60 held-out train clips (v28b/v30/v31 set)
               "BIOHUB_REPAIR_DEADLINE_S": "42000",                                       # x138's 7.5h guard would degrade the final rewrite
               "BIOHUB_PPSWEEP_SELECT_MARGIN": "0.001", "BIOHUB_PPSWEEP_MAX_ADJ_LOSS": "0.0005"}.items():
    os.environ[_k] = _v
print("v34: X1 gate config + validator 30/type + x138-knob sweep | hint:", os.environ["BIOHUB_DIVNET_CKPT_HINT"])
'''

g = cells[1]["source"]
old = 'if _drift:\n    raise RuntimeError(\n        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)\n    )'
assert g.count(old) == 1
cells[1]["source"] = g.replace(old, 'if _drift:\n    print("NOTE (ours):", _guard_json.dumps(_drift, sort_keys=True))')
p = cells[4]["source"]
for a, b in {
    "os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'": "os.environ.setdefault('BIOHUB_EDGE_FEATURE_TTA', '1')",
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "1")',
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "0.75")',
}.items():
    assert p.count(a) == 1, a; p = p.replace(a, b)
# OURS: the V1284 coordinate-refinement head lives in a 4th public dataset ('biohub-v1284-head-s075').
# If it is not attached, fall back to V1284_MODE='zero' (= x138 without coordinate refinement) instead of killing the kernel.
_h_old = """if len(_myhead) != 1:
    raise RuntimeError(('my V1284 head mount mismatch', [str(p) for p in _myhead]))"""
_h_new = """if len(_myhead) != 1:
    print("!!!!!!!! OURS: V1284 head dataset 'biohub-v1284-head-s075' NOT attached ->", [str(p) for p in _myhead])
    print("!!!!!!!! OURS: running WITHOUT coordinate refinement (V1284_MODE=zero). Attach the dataset for the full x138 result.")
    _myhead = [Path('/kaggle/working/_no_v1284_head.pt')]
    os.environ['V1284_NO_HEAD'] = '1'"""
assert p.count(_h_old) == 1
p = p.replace(_h_old, _h_new)
_m_old = "os.environ['V1284_MODE']='candidate'"
assert p.count(_m_old) == 1
p = p.replace(_m_old, "os.environ['V1284_MODE']='zero' if os.environ.get('V1284_NO_HEAD') == '1' else 'candidate'\nprint('V1284_MODE =', os.environ['V1284_MODE'])")
cells[4]["source"] = p

q = cells[5]["source"]
blk1 = (SC / "divnet_blk1.py").read_text(); blk2 = (SC / "divnet_blk2.py").read_text()
DIVNET_GLOBALS = r'''
# ---- OURS: DivNet gate switches (ported from v25e / v31)
DIVNET_VERIFY = True
DIVNET_MIN_PROB = float(os.environ.get("BIOHUB_DIV_MIN_PROB", "0.80"))
DIVNET_SAFE_DIV_GATE = os.environ.get("BIOHUB_DIVNET_SAFE_DIV_GATE", "0") != "0"
DIVNET_TTA = os.environ.get("BIOHUB_DIVNET_TTA", "0") != "0"
DIVNET_RANK_PROPOSALS = False
'''
anchor_dc = "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()"
assert q.count(anchor_dc) == 1
q = q.replace(anchor_dc, anchor_dc + "\n" + DIVNET_GLOBALS + "\n" + blk1 + "\n\n" + blk2 + "\n")
anchor = '''                stats["safe_division_geometric_candidates"] += 1
                if DEEPCENTER_SAFE_DIV_VETO and not deepcenter_accept_repair_point(
                    dataset,
                    int(candidate["t"]),
                    node_point(candidate),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "safe_div",
                    DEEPCENTER_SAFE_DIV_THRESHOLD,
                ):
                    continue
'''
assert q.count(anchor) == 1
q = q.replace(anchor, anchor + '''                # ---- OURS: DivNet mitosis gate on the (parent, existing child, candidate) triple
                _dn_prob = None
                if DIVNET_SAFE_DIV_GATE:
                    _dn_key = (dataset, int(source["t"]), int(round(float(source["z"]))), int(round(float(source["y"]))), int(round(float(source["x"]))))
                    _dn_memo = globals().setdefault("_DIVNET_MEMO", {})
                    if _dn_key in _dn_memo:
                        _dn_prob = _dn_memo[_dn_key]
                    else:
                        _dn_prob = divnet_score_division(dataset, int(source["t"]), source, existing_child, candidate, globals().get("DIVNET_BUNDLE"), frame_cache)
                        _dn_memo[_dn_key] = _dn_prob
                    if _dn_prob is None:
                        stats["divnet_safe_div_unscored"] = stats.get("divnet_safe_div_unscored", 0) + 1
                    elif _dn_prob < DIVNET_MIN_PROB:
                        stats["divnet_safe_div_rejected"] = stats.get("divnet_safe_div_rejected", 0) + 1
                        continue
                    else:
                        stats["divnet_safe_div_accepted"] = stats.get("divnet_safe_div_accepted", 0) + 1
''')
_log = "    _geo_cands = stats['safe_division_geometric_candidates']"
assert q.count(_log) == 1
q = q.replace(_log, "    print(f\"  [{dataset}] divnet safe-div gate: rejected={stats.get('divnet_safe_div_rejected', 0)} accepted={stats.get('divnet_safe_div_accepted', 0)} unscored={stats.get('divnet_safe_div_unscored', 0)}\")\n" + _log)
cells[5]["source"] = q

# ---- OURS: extend the sweepable keys with the x138 knobs and replace the candidate grid
_k_old = '    "GAP_CLOSE_REUSE_UM", "OUTPUT_EDGE_MAX_UM",\n]'
_k_hits = [i for i, c in enumerate(cells) if c["source"].count(_k_old) == 1]
assert len(_k_hits) == 1, "PP_SWEEP_KEYS anchor"
cells[_k_hits[0]]["source"] = cells[_k_hits[0]]["source"].replace(_k_old, '    "GAP_CLOSE_REUSE_UM", "OUTPUT_EDGE_MAX_UM",\n'
                               '    "READMIT_RADIUS_UM", "READMIT_MIN_SCORE", "GAPFILL_MIN_SCORE", "GAPFILL_MAX_GAP",\n'
                               '    "MOTION_RELINK_FLOW_TIGHT_UM", "MOTION_RELINK_FLOW_K",\n]')
print("sweep keys patched in cell", _k_hits[0])
for _ci, _c in enumerate(cells):
    if 'PP_CANDIDATES: dict[str, dict] = {' in _c["source"]:
        _s = _c["source"]
        _i = _s.find('PP_CANDIDATES: dict[str, dict] = {'); _j = _s.find('}\n', _i) + 2
        _s = _s[:_i] + """PP_CANDIDATES: dict[str, dict] = {
    # x138's own knobs were never scored on a hold-out (its validator is off); edge-side only (the classifier threshold
    # is NOT swept here: the 60 validator clips are part of the v28e training set, so that comparison would be biased).
    "readmit_s095": {"READMIT_MIN_SCORE": 0.95},
    "readmit_s098": {"READMIT_MIN_SCORE": 0.98},
    "readmit_r3": {"READMIT_RADIUS_UM": 3.0},
    "readmit_r5": {"READMIT_RADIUS_UM": 5.0},
    "readmit_off": {"READMIT_RADIUS_UM": 0.0},
    "gapfill_s04": {"GAPFILL_MIN_SCORE": 0.4},
    "gapfill_s06": {"GAPFILL_MIN_SCORE": 0.6},
    "gapfill_g2": {"GAPFILL_MAX_GAP": 2},
    "gapfill_g4": {"GAPFILL_MAX_GAP": 4},
    "gapfill_off": {"GAPFILL_MAX_GAP": 0},
    "flow_t6": {"MOTION_RELINK_FLOW_TIGHT_UM": 6.0},
    "flow_t8": {"MOTION_RELINK_FLOW_TIGHT_UM": 8.0},
    "flow_k8": {"MOTION_RELINK_FLOW_K": 8},
    "flow_k16": {"MOTION_RELINK_FLOW_K": 16},
    "tight50": {"MOTION_RELINK_TIGHT_UM": 5.0},
    "tight60": {"MOTION_RELINK_TIGHT_UM": 6.0},
}
""" + _s[_j:]
        cells[_ci]["source"] = _s
        print("sweep grid patched in cell", _ci)
        break
else:
    raise RuntimeError("PP_CANDIDATES cell not found")

def code(src): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src.strip("\n")}
new = [{"cell_type": "markdown", "metadata": {}, "source": "# Biohub · v34 — x138 + our v28e gate (X1) + 60-clip hold-out sweep of the x138 knobs (readmit / gapfill / flow / tight)\n\nInputs: competition + pilkwang ×3 + biohub-v1284-head-s075 + **v28e** notebook Output. Writes submission.csv with the hold-out-selected configuration; ppsweep_results.csv / ppsweep_selected.json in the output."},
       cells[0], code(VARIANT_CELL)] + cells[1:]
json.dump({"cells": new, "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}, open(OUT, "w"), indent=1)
print("wrote", OUT, "cells:", len(new))
