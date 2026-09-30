"""v30 = DIAGNOSTIC (no submission value): harmonic-fusion with the 60-clip validator, then for EVERY GT division in those
clips classify why it is / is not recovered: parent or daughter never detected, removed by post-processing, daughter linked
elsewhere, or which safe-division geometry gate rejected the (parent, child, candidate) triple.  Decides whether a targeted
gate relaxation can buy division recall.  GPU T4x2, ~3.5h.  Output: division_diag.csv + printed summary."""
import json, sys
from pathlib import Path
SRC = Path("public_notebooks/7c6cee3f-biohub-harmonic-fusion.ipynb")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "biohub_v30_division_diag.ipynb")
nb = json.load(open(SRC))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for c in cells:
    c["outputs"] = []; c["execution_count"] = None; c["source"] = "".join(c["source"])

CONFIG_CELL = r'''
# ---------------------------------------------------------------- OURS: v30 diagnostic configuration (Q1/R5 geometry)
os.environ["BIOHUB_DET_THRESHOLD"] = "0.96"
os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"
os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"
os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "30"      # 60 train clips (same set as v28b)
print("v30: validator clips per type =", os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"])
'''
g = cells[1]["source"]
old = 'if _drift:\n    raise RuntimeError(\n        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)\n    )'
assert g.count(old) == 1; cells[1]["source"] = g.replace(old, 'if _drift:\n    print("NOTE (ours):", _guard_json.dumps(_drift, sort_keys=True))')
p = cells[4]["source"]
for a, b in {
    "os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'": "os.environ.setdefault('BIOHUB_EDGE_FEATURE_TTA', '1')",
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "1")',
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "0.75")',
}.items():
    assert p.count(a) == 1, a; p = p.replace(a, b)
cells[4]["source"] = p

DIAG_CELL = r'''
# ================================================================ OURS (v30): where do the GT divisions go?
import copy as _copy, csv
from collections import Counter, defaultdict
assert VALIDATOR_ENABLE and val_stems and VAL_RAW_GRAPHS, "validator must have run"
_real_test_dir = TEST_DIR
globals()["TEST_DIR"] = TRAIN_DIR
reasons = Counter(); details = []; n_gt_div = 0
t0 = time.time()
try:
    for k_, stem in enumerate(val_stems):
        raw_nodes, raw_edges = VAL_RAW_GRAPHS[stem]
        pn, pe, _st = filter_output_graph(_copy.deepcopy(raw_nodes), _copy.deepcopy(raw_edges), dataset=stem,
                                          deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"))
        gt_nodes, gt_edges, _ = VAL_GT[stem]
        p2g, g2p = match_nodes_bipartite(nodes_by_id_to_plain(pn), gt_nodes, VALIDATOR_MATCH_RADIUS_UM)
        rp2g, rg2p = match_nodes_bipartite(nodes_by_id_to_plain(raw_nodes), gt_nodes, VALIDATOR_MATCH_RADIUS_UM)
        out = defaultdict(list); inc = {}
        for e in pe:
            out[int(e["source_id"])].append(int(e["target_id"])); inc[int(e["target_id"])] = int(e["source_id"])
        gt_out = defaultdict(list)
        for s_, t_ in gt_edges:
            gt_out[s_].append(t_)
        for P, ch in gt_out.items():
            if len(ch) < 2:
                continue
            n_gt_div += 1
            D1, D2 = ch[0], ch[1]
            pp = g2p.get(P); d = [g2p.get(D1), g2p.get(D2)]
            extra = ""
            if pp is not None and None not in d and len(out.get(pp, [])) >= 2 and all(x in out[pp] for x in d):
                r = "A_tp_fork_at_t"
            elif pp is not None and None not in d and any(len(out.get(x, [])) >= 2 for x in d):
                r = "A_fork_at_t+1(maybe tp via +-1 frame)"
            elif pp is not None and pp in inc and len(out.get(inc[pp], [])) >= 2:
                r = "A_fork_at_t-1(maybe tp via +-1 frame)"
            elif pp is None:
                r = "B_parent_never_detected" if rg2p.get(P) is None else "C_parent_removed_by_postprocess"
            elif None in d:
                miss = [D for D, x in zip((D1, D2), d) if x is None]
                r = "B_daughter_never_detected" if any(rg2p.get(D) is None for D in miss) else "C_daughter_removed_by_postprocess"
            else:
                kids = out.get(pp, [])
                if len(kids) == 0:
                    r = "D_parent_track_ends_both_daughters_unlinked"
                elif len(kids) >= 2:
                    r = "D_fork_to_wrong_nodes"
                else:
                    ec = kids[0]
                    if ec not in d:
                        r = "D_single_child_is_wrong_node"
                    else:
                        q = d[0] if d[1] == ec else d[1]
                        if q in inc:
                            r = "D_daughter2_linked_to_other_parent"
                        else:
                            src, exc, cand = pn[pp], pn[ec], pn[q]
                            child_dist = edge_distance_um(src, exc); parent_dist = edge_distance_um(src, cand); sister = edge_distance_um(exc, cand)
                            extra = f"child={child_dist:.1f} parent={parent_dist:.1f} sister={sister:.1f}"
                            starts = [nid for nid, nd in pn.items() if int(nd["t"]) == int(cand["t"]) and nid not in inc]
                            nn = min(starts, key=lambda nid: edge_distance_um(exc, pn[nid])) if starts else None
                            c1s = out.get(ec, []); qs = out.get(q, [])
                            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM: r = "E_gate_existing_child_dist"
                            elif parent_dist > SAFE_DIV_MAX_UM: r = "E_gate_parent_dist"
                            elif sister > SAFE_DIV_SISTER_MAX_UM: r = "E_gate_sister_dist"
                            elif SAFE_DIV_REQUIRE_MUTUAL_NN and nn != q: r = "E_gate_mutual_nn"
                            elif SAFE_DIV_REQUIRE_DIVERGENCE and (len(c1s) != 1 or len(qs) != 1): r = "E_gate_divergence_no_grandchild"
                            elif SAFE_DIV_REQUIRE_DIVERGENCE and edge_distance_um(pn[c1s[0]], pn[qs[0]]) - sister < SAFE_DIV_DIVERGE_UM: r = "E_gate_divergence_dist"
                            elif SAFE_DIV_SISTER_SYMMETRY_TAU > 0 and abs(child_dist - parent_dist) / max((child_dist + parent_dist) / 2.0, 1e-6) > SAFE_DIV_SISTER_SYMMETRY_TAU: r = "E_gate_symmetry"
                            else: r = "F_geometry_ok_not_added(DeepCenter_veto_or_cap)"
            reasons[r] += 1
            details.append({"stem": stem, "gt_parent": int(P), "t": int(gt_nodes[P][0]), "reason": r, "geom": extra})
        if (k_ + 1) % 10 == 0:
            print(f"  {k_ + 1}/{len(val_stems)} clips, {time.time() - t0:.0f}s", flush=True)
finally:
    globals()["TEST_DIR"] = _real_test_dir

print("=" * 78)
print(f"GT divisions in {len(val_stems)} clips: {n_gt_div}   (validator base: tp/fp/fn = {PP_RESULTS['base']['div_tp']}/{PP_RESULTS['base']['div_fp']}/{PP_RESULTS['base']['div_fn']})")
print("A = already a fork | B = detector miss (unfixable) | C = post-process removed it | D = linking put it elsewhere | E = safe-div geometry gate | F = passed geometry")
for r, n in sorted(reasons.items()):
    print(f"  {n:4d}  {r}")
by_prefix = defaultdict(Counter)
for row in details:
    by_prefix[row["stem"].split("_")[0]][row["reason"][0]] += 1
print("by embryo type (letter = category):", {k: dict(v) for k, v in by_prefix.items()})
fixable = sum(n for r, n in reasons.items() if r[0] in "CEF")
print(f"POTENTIALLY FIXABLE (C+E+F): {fixable} of {n_gt_div}  -> if ~half became tp: divJ +{0.5 * fixable / max(1, n_gt_div + PP_RESULTS['base']['div_fp']):.3f}  ~ LB +{0.05 * fixable / max(1, n_gt_div + PP_RESULTS['base']['div_fp']):.4f}")
with (WORKING_DIR / "division_diag.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["stem", "gt_parent", "t", "reason", "geom"]); w.writeheader(); w.writerows(details)
print("wrote", WORKING_DIR / "division_diag.csv")
'''
def code(src): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src.strip("\n")}
keep = cells[:10]
new = [{"cell_type": "markdown", "metadata": {}, "source": "# Biohub · v30 — division recall diagnostic (not for submission)\n\nGPU T4×2, ~3.5h. Inputs: competition + pilkwang ×3. Prints, for every GT division in 60 held-out train clips, why it was or was not recovered; `division_diag.csv` in Output."},
       keep[0], code(CONFIG_CELL)] + keep[1:] + [code(DIAG_CELL)]
json.dump({"cells": new, "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}, open(OUT, "w"), indent=1)
print("wrote", OUT, "cells:", len(new))
