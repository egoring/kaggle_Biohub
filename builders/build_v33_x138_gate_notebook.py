"""v33 = public 'biohub-x138' notebook (harmonic base + flow-prior motion relink + re-admitted low-score detections + gap fill
from sub-threshold peaks + ILP timeout / runtime guards) with our DivNet safe-division gate layered on via a VARIANT cell.
X0 = x138 exactly (to learn its LB).  X1 = x138 + our classifier gate (R14 config).  Inputs: competition + pilkwang x3 (+ v28e Output for X1)."""
import json, sys
from pathlib import Path
SC = Path("../src")
SRC = Path("public_notebooks/ab30ed16-biohub-x138.ipynb")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "biohub_v33_x138_divnet_variants.ipynb")
nb = json.load(open(SRC))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for c in cells:
    c["outputs"] = []; c["execution_count"] = None; c["source"] = "".join(c["source"])

VARIANT_CELL = r'''
# ---------------------------------------------------------------- OURS: variant switch (edit VARIANT only)
# X0 = the public x138 notebook exactly (flow relink + readmit + gapfill; DeepCenter safe-div veto; det 0.965)
# X1 = X0 + OUR division classifier replaces the DeepCenter safe-div veto (the R14 recipe: gate 0.80, v28e checkpoint)
VARIANT = "X1"
VARIANTS = {
    "X0": {},
    "X1": {"BIOHUB_DIVNET_SAFE_DIV_GATE": "1", "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80",
           "BIOHUB_DIVNET_REQUIRE_OURS": "1", "BIOHUB_DIVNET_CKPT_HINT": "v28e"},
}
# X2 = X1 + DeepCenter veto kept ON as well (both gates);  X3 / X4 = X1 with classifier threshold 0.75 / 0.85
VARIANTS["X2"] = dict(VARIANTS["X1"], **{"BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "1"})
VARIANTS["X3"] = dict(VARIANTS["X1"], **{"BIOHUB_DIV_MIN_PROB": "0.75"})
VARIANTS["X4"] = dict(VARIANTS["X1"], **{"BIOHUB_DIV_MIN_PROB": "0.85"})
VARIANTS["X5"] = dict(VARIANTS["X1"], **{"BIOHUB_DIV_MIN_PROB": "0.70"})
# X6 = X1 but with the v28f 5-seed ensemble at a threshold matched to its softer calibration (R17 tested it only at 0.80: 0.945)
VARIANTS["X6"] = dict(VARIANTS["X1"], **{"BIOHUB_DIVNET_CKPT_HINT": "v28f", "BIOHUB_DIV_MIN_PROB": "0.65"})
# v34 60-clip sweep (9/27): flow tight 7.0->6.0 (+0.0010) and flow K 12->16 (+0.0005), combo +0.0016 proxy / +0.0018 adj; 8.0 was -0.010.
_FLOW = {"BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM": "6.0", "BIOHUB_MOTION_RELINK_FLOW_K": "16"}
VARIANTS["X7"] = dict(VARIANTS["X1"], **_FLOW)                                              # == v34's selected submission
VARIANTS["X8"] = dict(VARIANTS["X3"], **_FLOW)                                              # threshold 0.75 + flow combo
VARIANTS["X9"] = dict(VARIANTS["X1"], **{"BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM": "5.0", "BIOHUB_MOTION_RELINK_FLOW_K": "16"})   # blind: one step further
VARIANTS["X10"] = dict(VARIANTS["X1"], **{"BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM": "6.0", "BIOHUB_MOTION_RELINK_FLOW_K": "24"})  # blind: more flow samples
# 9/28 (X8/X9/X10 = 0.956: flow knobs stay at x138 defaults).  Ensemble-calibration follow-ups:
VARIANTS["X11"] = dict(VARIANTS["X1"], **{"BIOHUB_DIVNET_CKPT_HINT": "v28f", "BIOHUB_DIV_MIN_PROB": "0.55"})       # v28f Output
VARIANTS["X12"] = dict(VARIANTS["X1"], **{"BIOHUB_DIVNET_CKPT_HINT": "v28f", "BIOHUB_DIV_MIN_PROB": "0.75"})       # v28f Output
VARIANTS["X13"] = dict(VARIANTS["X1"], **{"BIOHUB_DIVNET_CKPT_HINT": "v28c+v28e", "BIOHUB_DIV_MIN_PROB": "0.75"})  # v28c AND v28e Outputs
VARIANTS["X14"] = dict(VARIANTS["X1"], **{"BIOHUB_DIVNET_CKPT_HINT": "v28c+v28e", "BIOHUB_DIV_MIN_PROB": "0.80"})  # v28c AND v28e Outputs
# 9/29: X11 (v28f @0.55) = 0.960, X12 (@0.75) = 0.952 -> the 5-seed mean is compressed; walk the threshold down.
VARIANTS["X15"] = dict(VARIANTS["X1"], **{"BIOHUB_DIVNET_CKPT_HINT": "v28f", "BIOHUB_DIV_MIN_PROB": "0.45"})       # v28f Output
VARIANTS["X16"] = dict(VARIANTS["X1"], **{"BIOHUB_DIVNET_CKPT_HINT": "v28f", "BIOHUB_DIV_MIN_PROB": "0.50"})       # v28f Output
VARIANTS["X17"] = dict(VARIANTS["X1"], **{"BIOHUB_DIVNET_CKPT_HINT": "v28f", "BIOHUB_DIV_MIN_PROB": "0.60"})       # v28f Output
# 9/29 evening: 0.45/0.50/0.55 all 0.960 (plateau). Last slot: widen the ensemble with a differently-trained model.
VARIANTS["X18"] = dict(VARIANTS["X1"], **{"BIOHUB_DIVNET_CKPT_HINT": "v28f+v28c", "BIOHUB_DIV_MIN_PROB": "0.50"})  # v28f AND v28c Outputs (5 seeds + v28c = 6 models)
VARIANTS["X19"] = dict(VARIANTS["X1"], **{"BIOHUB_DIVNET_CKPT_HINT": "v28f+v28e", "BIOHUB_DIV_MIN_PROB": "0.50"})  # v28f AND v28e Outputs (6 models)
for _k, _v in VARIANTS[VARIANT].items():
    os.environ[_k] = _v
print("VARIANT:", VARIANT, VARIANTS[VARIANT])
print("  divnet safe-div gate :", os.environ.get("BIOHUB_DIVNET_SAFE_DIV_GATE", "0"), "| min prob:", os.environ.get("BIOHUB_DIV_MIN_PROB"), "| hint:", os.environ.get("BIOHUB_DIVNET_CKPT_HINT"))
print("  deepcenter safe-div  :", os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1"))
print("  flow relink / readmit / gapfill:", os.environ.get("BIOHUB_MOTION_RELINK_FLOW_MODE"), os.environ.get("BIOHUB_READMIT_RADIUS_UM"), os.environ.get("BIOHUB_GAPFILL_MAX_GAP"))
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
# OURS (v33, 9/28): hint "a+b" loads the primary model of EACH hinted checkpoint into bundle["models"] (scorer averages sigmoids)
_h_old = """        _ours = _hinted; print("  hint", repr(_hint), "->", str(_ours[0]))
"""
_h_new = """        _ours = _hinted; print("  hint", repr(_hint), "->", str(_ours[0]))
    if "+" in _hint:
        _parts = [h.strip() for h in _hint.split("+") if h.strip()]
        _models, _in_c, _first = [], None, None
        for _part in _parts:
            _pp = [p for p in _ours if _part in str(p)]
            if len(_pp) != 1:
                raise RuntimeError(f"multi-hint part {_part!r} matches {len(_pp)} checkpoints: {[str(p) for p in _pp]}")
            _ck = torch.load(_pp[0], map_location=device, weights_only=False)
            _st = _ck.get("model_state", _ck) if isinstance(_ck, dict) else _ck
            _c = int(_st['b1.conv.0.weight'].shape[1]) if 'b1.conv.0.weight' in _st else 1
            if _in_c is not None and _c != _in_c:
                raise RuntimeError("multi-hint checkpoints disagree on in_channels")
            _in_c = _c
            _states = list(_ck.get("ensemble_states", [])) if isinstance(_ck, dict) else []
            _states = _states if _states else [_st]          # a 5-seed checkpoint contributes all its seeds; a single one its model
            for _k, _s1 in enumerate(_states):
                _m = DivNetMitosisClassifier(_c); _mi, _mu = _m.load_state_dict(_s1, strict=False); _m.to(device); _m.eval()
                print(f"  multi-hint member {_part}[{_k}]: {_pp[0]} missing={len(_mi)} unexpected={len(_mu)}")
                _models.append(_m); _first = _first or _m
        print(f"  DivNet models in bundle (multi-hint): {len(_models)}")
        return {"model": _first, "device": device, "torch": torch, "in_c": _in_c, "models": _models}
"""
assert blk1.count(_h_old) == 1, "hint anchor"
blk1 = blk1.replace(_h_old, _h_new)
# the multi-hint filter must keep every part: replace the single-substring filter
_f_old = "        _hinted = [p for p in _ours if _hint in str(p)]\n"
_f_new = "        _hinted = [p for p in _ours if any(h.strip() and h.strip() in str(p) for h in _hint.split('+'))]\n"
assert blk1.count(_f_old) == 1, "filter anchor"
blk1 = blk1.replace(_f_old, _f_new)
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
                    _dn_prob = divnet_score_division(dataset, int(source["t"]), source, existing_child, candidate, globals().get("DIVNET_BUNDLE"), frame_cache)
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

def code(src): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src.strip("\n")}
new = [{"cell_type": "markdown", "metadata": {}, "source": "# Biohub · v33 — public x138 (flow relink + readmit + gapfill) + our division classifier gate\n\nEdit `VARIANT` in cell 2. X0 = x138 as published (Inputs: competition + pilkwang ×3). X1 = + our gate (also attach the **v28e** notebook Output)."},
       cells[0], code(VARIANT_CELL)] + cells[1:]
json.dump({"cells": new, "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}, open(OUT, "w"), indent=1)
print("wrote", OUT, "cells:", len(new))
