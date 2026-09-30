"""v31 = harmonic-fusion base (has the 60-clip validator + PP sweep) + our DivNet safe-division gate (R5/R14 config) +
NEW post-process step 'duplicate-parent fork merge' (v30 diagnostic: 29/78 GT divisions lose daughter #2 to a nearby
duplicate detection of the parent -> reroute that edge onto the real parent, classifier-gated).  The sweep scores
base vs dup-merge radii vs SAFE_DIV_MAX_UM 11 vs threshold 0.75 on 60 held-out clips and re-writes the submission
with the best config.  GPU T4x2, ~4h.  Inputs: competition + pilkwang x3 + v28f Output (checkpoint)."""
import json, sys
from pathlib import Path
SC = Path("../src")
SRC = Path("public_notebooks/7c6cee3f-biohub-harmonic-fusion.ipynb")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "biohub_v31_dupmerge_sweep.ipynb")
nb = json.load(open(SRC))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for c in cells:
    c["outputs"] = []; c["execution_count"] = None; c["source"] = "".join(c["source"])

CONFIG_CELL = r'''
# ---------------------------------------------------------------- OURS: v31 configuration = R14 (0.953) + validator + dup-merge sweep
os.environ["BIOHUB_DET_THRESHOLD"] = "0.96"
os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"          # our classifier replaces the DeepCenter veto (R5+)
os.environ["BIOHUB_DIVNET_SAFE_DIV_GATE"] = "1"
os.environ["BIOHUB_DIV_MIN_PROB"] = "0.80"
os.environ["BIOHUB_DIVNET_REQUIRE_OURS"] = "1"
os.environ["BIOHUB_DIVNET_CKPT_HINT"] = os.environ.get("BIOHUB_DIVNET_CKPT_HINT", "v28f")
os.environ["BIOHUB_DIVNET_TTA"] = "0"
os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"
os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "30"             # 60 held-out train clips (same set as v28b / v30)
# NEW step (default OFF in base; the sweep turns it on):
os.environ["BIOHUB_DUP_PARENT_MERGE"] = "0"
os.environ["BIOHUB_DUP_PARENT_MAX_UM"] = "3.0"               # two 'parents' at the same t closer than this = duplicate detection
os.environ["BIOHUB_DUP_PARENT_GLOBAL_FRAC_CAP"] = "0.004"
print("v31: R14 gate config + validator 30/type + duplicate-parent merge sweep; ckpt hint =", os.environ["BIOHUB_DIVNET_CKPT_HINT"])
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

# ---------------- post-process cell
q = cells[5]["source"]
blk1 = (SC / "divnet_blk1.py").read_text()
blk2 = (SC / "divnet_blk2.py").read_text()
DIVNET_GLOBALS = r'''
# ---- OURS: DivNet gate switches (ported from v25e)
DIVNET_VERIFY = True
DIVNET_MIN_PROB = float(os.environ.get("BIOHUB_DIV_MIN_PROB", "0.80"))
DIVNET_SAFE_DIV_GATE = os.environ.get("BIOHUB_DIVNET_SAFE_DIV_GATE", "0") != "0"
DIVNET_TTA = os.environ.get("BIOHUB_DIVNET_TTA", "0") != "0"
DIVNET_RANK_PROPOSALS = False
DUP_PARENT_MERGE = os.environ.get("BIOHUB_DUP_PARENT_MERGE", "0") != "0"
DUP_PARENT_MAX_UM = float(os.environ.get("BIOHUB_DUP_PARENT_MAX_UM", "3.0"))
DUP_PARENT_GLOBAL_FRAC_CAP = float(os.environ.get("BIOHUB_DUP_PARENT_GLOBAL_FRAC_CAP", "0.004"))
'''
MERGE_FN = r'''

# ---- OURS (v31): duplicate-parent fork merge.  v30 diagnostic on 60 held-out clips: 29 of 78 GT divisions had BOTH daughters
# detected and linked, but daughter #2 hung off a second node at the same t within a few um of the real parent (a duplicate
# detection) instead of the parent.  The safe-division generator never sees these because daughter #2 is not a track start.
# Fix: for two single-child nodes at the same t closer than DUP_PARENT_MAX_UM, reroute the weaker one's child edge onto the
# stronger one (the node with an incoming edge / higher child edge_prob), forming a fork -- but only if the classifier says
# the triple is a division (>= DIVNET_MIN_PROB).  The dropped node keeps its own incoming edge (becomes a track end).
def merge_duplicate_parent_forks(nodes_by_id, edges, stats, dataset=None, frame_cache=None):
    st = {"dup_pairs": 0, "dup_geom_ok": 0, "dup_scored": 0, "dup_rejected": 0, "dup_unscored": 0, "dup_merged": 0, "dup_cap_skipped": 0}
    stats.update(st)
    if not DUP_PARENT_MERGE or not edges or globals().get("DIVNET_BUNDLE") is None:
        return edges
    frame_cache = frame_cache if frame_cache is not None else {}
    out = {}; inc = {}
    for e in edges:
        out.setdefault(int(e["source_id"]), []).append(e); inc[int(e["target_id"])] = int(e["source_id"])
    ids_by_t = {}
    for nid, nd in nodes_by_id.items():
        ids_by_t.setdefault(int(nd["t"]), []).append(nid)
    global_cap = max(1, int(round(len(edges) * DUP_PARENT_GLOBAL_FRAC_CAP)))
    used = set(); merged = 0
    for t in sorted(ids_by_t):
        singles = [n for n in ids_by_t[t] if len(out.get(n, [])) == 1]
        if len(singles) < 2:
            continue
        tree = cKDTree(np.stack([_position_um(nodes_by_id[n]) for n in singles]))
        for i, j in sorted(tree.query_pairs(DUP_PARENT_MAX_UM)):
            a, b = singles[i], singles[j]
            if a in used or b in used:
                continue
            ca, cb = int(out[a][0]["target_id"]), int(out[b][0]["target_id"])
            nca, ncb = nodes_by_id.get(ca), nodes_by_id.get(cb)
            if ca == cb or nca is None or ncb is None or int(nca["t"]) != t + 1 or int(ncb["t"]) != t + 1:
                continue
            stats["dup_pairs"] += 1
            def _strength(n):
                e = out[n][0]; ep = e.get("edge_prob"); return (1 if n in inc else 0, float(ep) if ep is not None else 0.0, -n)
            keep, drop = (a, b) if _strength(a) >= _strength(b) else (b, a)
            ck, cd = (ca, cb) if keep == a else (cb, ca)
            nk = nodes_by_id[keep]
            if edge_distance_um(nodes_by_id[ck], nodes_by_id[cd]) > SAFE_DIV_SISTER_MAX_UM or edge_distance_um(nk, nodes_by_id[cd]) > SAFE_DIV_MAX_UM:
                continue
            stats["dup_geom_ok"] += 1
            prob = divnet_score_division(dataset, t, nk, nodes_by_id[ck], nodes_by_id[cd], globals().get("DIVNET_BUNDLE"), frame_cache)
            if prob is None:
                stats["dup_unscored"] += 1; continue
            stats["dup_scored"] += 1
            if prob < DIVNET_MIN_PROB:
                stats["dup_rejected"] += 1; continue
            if merged >= global_cap:
                stats["dup_cap_skipped"] += 1; continue
            e_old = out[drop][0]
            e_old["source_id"] = keep; e_old["safe_division"] = 1; e_old["dup_merge"] = 1
            out[keep].append(e_old); out[drop] = []; inc[cd] = keep
            used.update((a, b)); merged += 1
    stats["dup_merged"] = merged
    return edges

'''
# (1) DivNet globals + module after the DeepCenter detector line
anchor_dc = "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()"
assert q.count(anchor_dc) == 1
q = q.replace(anchor_dc, anchor_dc + "\n" + DIVNET_GLOBALS + "\n" + blk1 + "\n\n" + blk2 + "\n" + MERGE_FN)
# (2) gate inside the safe-division loop (same anchor as v25)
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
assert q.count(anchor) == 1, "safe-div anchor not unique"
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
# (3) call the merge step right after safe divisions
call_anchor = '''    _geo_cands = stats['safe_division_geometric_candidates']'''
assert q.count(call_anchor) == 1
q = q.replace(call_anchor, '''    edges = merge_duplicate_parent_forks(nodes_by_id, edges, stats, dataset=dataset, frame_cache=repair_frame_cache)
    print(f"  [{dataset}] dup-parent merge: on={DUP_PARENT_MERGE} r={DUP_PARENT_MAX_UM} pairs={stats['dup_pairs']} geom_ok={stats['dup_geom_ok']}"
          f" scored={stats['dup_scored']} rejected={stats['dup_rejected']} unscored={stats['dup_unscored']} MERGED={stats['dup_merged']} cap_skipped={stats['dup_cap_skipped']}"
          f" | divnet safe-div rejected={stats.get('divnet_safe_div_rejected', 0)} accepted={stats.get('divnet_safe_div_accepted', 0)} unscored={stats.get('divnet_safe_div_unscored', 0)}")
''' + call_anchor)
cells[5]["source"] = q

# ---------------- sweep cells: add our keys + our candidates
s9 = cells[9]["source"]
old_keys = '''    "GAP_CLOSE_REUSE_UM", "OUTPUT_EDGE_MAX_UM",
]'''
assert s9.count(old_keys) == 1
cells[9]["source"] = s9.replace(old_keys, '''    "GAP_CLOSE_REUSE_UM", "OUTPUT_EDGE_MAX_UM",
    "DUP_PARENT_MERGE", "DUP_PARENT_MAX_UM", "DIVNET_MIN_PROB",          # OURS (v31)
]''')
s10 = cells[10]["source"]
old_c = s10[s10.find('PP_CANDIDATES: dict[str, dict] = {'):s10.find('}\n', s10.find('PP_CANDIDATES: dict[str, dict] = {')) + 2]
cells[10]["source"] = s10.replace(old_c, '''PP_CANDIDATES: dict[str, dict] = {
    # OURS (v31): base = R14 config (gate 0.80, no dup-merge).  Candidates target the v30 loss buckets.
    "dup35": {"DUP_PARENT_MERGE": True, "DUP_PARENT_MAX_UM": 3.5},
    "dup50": {"DUP_PARENT_MERGE": True, "DUP_PARENT_MAX_UM": 5.0},
    "pdist11": {"SAFE_DIV_MAX_UM": 11.0},                 # E_gate_parent_dist: 8 of 78
    "thr075": {"DIVNET_MIN_PROB": 0.75},
}
''')


DIAG2_CELL = r"""
# ================================================================ OURS (v31): what IS the 'other parent' of daughter #2?  (v30 bucket D = 29/78)
import copy as _copy
from collections import Counter, defaultdict
_real_test_dir = TEST_DIR; globals()["TEST_DIR"] = TRAIN_DIR
_saved = pp_apply({"DUP_PARENT_MERGE": False})
rows_d = []
try:
    for stem in val_stems:
        raw_nodes, raw_edges = VAL_RAW_GRAPHS[stem]
        pn, pe, _ = filter_output_graph(_copy.deepcopy(raw_nodes), _copy.deepcopy(raw_edges), dataset=stem, deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"))
        gt_nodes, gt_edges, _ = VAL_GT[stem]
        p2g, g2p = match_nodes_bipartite(nodes_by_id_to_plain(pn), gt_nodes, VALIDATOR_MATCH_RADIUS_UM)
        out = defaultdict(list); inc = {}
        for e in pe:
            out[int(e["source_id"])].append(int(e["target_id"])); inc[int(e["target_id"])] = int(e["source_id"])
        gt_out = defaultdict(list)
        for s_, t_ in gt_edges: gt_out[s_].append(t_)
        for P, ch in gt_out.items():
            if len(ch) < 2: continue
            pp = g2p.get(P); d = [g2p.get(ch[0]), g2p.get(ch[1])]
            if pp is None or None in d: continue
            kids = out.get(pp, [])
            if len(kids) != 1 or kids[0] not in d: continue
            q = d[0] if d[1] == kids[0] else d[1]
            if q not in inc: continue
            other = inc[q]
            rows_d.append({"stem": stem, "dist_um": edge_distance_um(pn[pp], pn[other]), "other_matched_gt": other in p2g,
                           "other_has_incoming": other in inc, "other_outdeg": len(out.get(other, [])), "same_t": int(pn[other]["t"]) == int(pn[pp]["t"])})
finally:
    pp_restore(_saved); globals()["TEST_DIR"] = _real_test_dir
print("=" * 78); print(f"bucket D cases: {len(rows_d)}")
bins = [(0, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 7.0), (7.0, 999)]
tab = Counter()
for r in rows_d:
    b = next(f"{lo}-{hi}um" for lo, hi in bins if lo <= r["dist_um"] < hi)
    tab[(b, "other=UNMATCHED(duplicate?)" if not r["other_matched_gt"] else "other=GT-matched(real neighbour)")] += 1
for k in sorted(tab): print(f"  {tab[k]:3d}  dist {k[0]:<10} {k[1]}")
n_dup = sum(1 for r in rows_d if not r["other_matched_gt"] and r["dist_um"] <= 5.0 and r["other_outdeg"] == 1)
print(f"mergeable by dup50 rule (unmatched, <=5um, single child): {n_dup} of {len(rows_d)}   |  other has incoming edge: {sum(r['other_has_incoming'] for r in rows_d)}")
print("=" * 78)
"""

def code(src): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src.strip("\n")}
new = [{"cell_type": "markdown", "metadata": {}, "source": "# Biohub · v31 — R14 gate + duplicate-parent fork merge, chosen on the 60-clip validator\n\nGPU T4×2, ~4h. Inputs: competition + pilkwang ×3 + **v28f Output** (`divnet_ours/best_overall.pt`). The sweep table at the end shows base(R14) vs each candidate; the submission is re-written with the selected config (or kept at base)."},
       cells[0], code(CONFIG_CELL)] + cells[1:10] + [code(DIAG2_CELL)] + cells[10:]
json.dump({"cells": new, "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}, open(OUT, "w"), indent=1)
print("wrote", OUT, "cells:", len(new))
