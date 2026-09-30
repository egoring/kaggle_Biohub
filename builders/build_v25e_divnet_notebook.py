"""v25 = public 'sota-0.948 density-adaptive' notebook (harmonic-fusion + per-clip density-adaptive motion relink +
DivNet checkpoint) with our VARIANT switch and ONE real change: the DivNet mitosis classifier is wired into the
safe-division repair stage (where divisions are actually decided).  In the public notebook DivNet is only called
inside `if OUTPUT_DIVISION_GEOMETRY_FILTER:` which is False, so it is loaded but never used.

Source cells kept verbatim except:
  * config-drift guard raise -> print
  * hard-coded TTA env assignments -> setdefault (VARIANT can switch them)
  * post-process cell: DivNet gate inserted right after the DeepCenter veto in the safe-division candidate loop,
    controlled by BIOHUB_DIVNET_SAFE_DIV_GATE; a hard assert that the checkpoint loaded when the gate is on;
    the 'after safe-division repair' log line also prints divnet_rejected=N.
"""
import json, sys
from pathlib import Path

SRC = Path("public_notebooks/3f23da76-biohub-sota-0-948-density-adaptive-2xt4-22m.ipynb")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "biohub_v25e_density_divnet_variants.ipynb")

nb = json.load(open(SRC))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for c in cells:
    c["outputs"] = []; c["execution_count"] = None; c["source"] = "".join(c["source"])
cells = cells[:7]            # drop the final matplotlib plotting cell (needs nothing, but useless for a submission)

VARIANT_CELL = r'''
# ---------------------------------------------------------------- OURS: variant switch (edit VARIANT only)
# R0 = the public 'sota 0.948' notebook exactly (det 0.965, edge/DeepCenter TTA, density-adaptive relink; DivNet loaded but unused)
# R1 = R0 + our detection threshold 0.96
# R2 = R1 + DivNet gate inside safe-division repair, AFTER the DeepCenter veto (precision only)   <- needs giorgosi/biohub-divnet-v2 attached
# R3 = R1 + DeepCenter safe-div veto OFF, DivNet gate (p>=0.50) decides instead                  (recall experiment)
# R4 = R3 with DivNet threshold 0.35 (more divisions)
VARIANT = "R1"
VARIANTS = {
    "R0": {},
    "R1": {"BIOHUB_DET_THRESHOLD": "0.96"},
    "R2": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1"},
    "R3": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1", "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0"},
    "R4": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1", "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0",
           "BIOHUB_DIV_MIN_PROB": "0.35"},
    # R5/R6: OUR v28 classifier (attach the v28 notebook Output; its divnet_ours/best_overall.pt is preferred over giorgosi's).
    #        Set BIOHUB_DIV_MIN_PROB to the 'recommended_threshold' printed by v28.  Q1's tight 5.5 is included.
    "R5": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
           "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1"},
    "R6": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
           "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1"},
    # R7: R5 + FORK GATE = our classifier also scores every final division fork (ILP-native ones included); forks below
    #     BIOHUB_DIVNET_FORK_MIN_PROB lose their weaker child edge.  Targets the 35 FP divisions / 60 clips (precision 0.35).
    "R7": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
           "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1",
           "BIOHUB_DIVNET_FORK_GATE": "1", "BIOHUB_DIVNET_FORK_MIN_PROB": "0.80"},
    # R8: R5 + RELAXED geometric candidate gates (divergence + mutual-NN checks off, caps x2): more true divisions reach the
    #     classifier (P14-style loosening cost -0.012 with DeepCenter; with a real classifier it may pay off).  Recall lever.
    "R8": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
           "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1",
           "BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE": "0", "BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN": "0",
           "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.016", "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.008"},
    # R9: R7 + R8 together
    "R9": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
           "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1",
           "BIOHUB_DIVNET_FORK_GATE": "1", "BIOHUB_DIVNET_FORK_MIN_PROB": "0.80",
           "BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE": "0", "BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN": "0",
           "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.016", "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.008"},
    # ---- v25c additions (after R5 = 0.952 LB) --------------------------------------------------------------------
    # R10: R7 + ILP division weight 1.0 (the sota base hard-codes 1.2 = fewer ILP forks; with the fork gate pruning bad
    #      forks we can afford more proposals).  Recall lever on the ILP side.
    "R10": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1",
            "BIOHUB_DIVNET_FORK_GATE": "1", "BIOHUB_DIVNET_FORK_MIN_PROB": "0.80",
            "BIOHUB_ILP_DIVISION_WEIGHT": "1.0"},
    # R11: R10 with ILP division weight 0.8 (more aggressive)
    "R11": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1",
            "BIOHUB_DIVNET_FORK_GATE": "1", "BIOHUB_DIVNET_FORK_MIN_PROB": "0.80",
            "BIOHUB_ILP_DIVISION_WEIGHT": "0.8"},
    # R12: R5 + RANK safe-division proposals by classifier probability (the caps then keep the most confident proposals
    #      instead of the geometrically closest) + sister-symmetry check relaxed (classifier decides).  Precision lever
    #      that only matters when 'safe_division_skipped_cap' > 0 in the R5 log.
    "R12": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1",
            "BIOHUB_DIVNET_RANK_PROPOSALS": "1", "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "1.0"},
    # R13: everything = R9 + R10 + R12  (only after each piece is >= R5 on its own)
    "R13": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1",
            "BIOHUB_DIVNET_FORK_GATE": "1", "BIOHUB_DIVNET_FORK_MIN_PROB": "0.80",
            "BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE": "0", "BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN": "0",
            "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.016", "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.008",
            "BIOHUB_ILP_DIVISION_WEIGHT": "1.0",
            "BIOHUB_DIVNET_RANK_PROPOSALS": "1", "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "1.0"},
    # ---- v25d: R14 = R5 exactly, but with the v28e checkpoint (more clips + 3-seed ensemble).  Attach the v28e Output;
    #      BIOHUB_DIVNET_CKPT_HINT picks the checkpoint whose path contains this text when several divnet_ours are attached.
    "R14": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1",
            "BIOHUB_DIVNET_CKPT_HINT": "v28e"},
    # ---- v25e: R15 = R14 + classifier test-time augmentation (D4 in yx x z-flip = 16 views averaged, same augs as training)
    "R15": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1",
            "BIOHUB_DIVNET_CKPT_HINT": "v28e", "BIOHUB_DIVNET_TTA": "1"},
    # R16 = R15 + DeepCenter safe-div veto kept ON as a second, independent filter (both gates must pass)
    "R16": {"BIOHUB_DET_THRESHOLD": "0.96", "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5", "BIOHUB_DIVNET_SAFE_DIV_GATE": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "1", "BIOHUB_DIV_MIN_PROB": "0.80", "BIOHUB_DIVNET_REQUIRE_OURS": "1",
            "BIOHUB_DIVNET_CKPT_HINT": "v28e", "BIOHUB_DIVNET_TTA": "1"},
}
for _k, _v in VARIANTS[VARIANT].items():
    os.environ[_k] = _v
print("VARIANT:", VARIANT, VARIANTS[VARIANT])
print("  det_threshold        :", os.environ.get("BIOHUB_DET_THRESHOLD"))
print("  deepcenter safe-div  :", os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1"))
print("  divnet safe-div gate :", os.environ.get("BIOHUB_DIVNET_SAFE_DIV_GATE", "0"), "| min prob:", os.environ.get("BIOHUB_DIV_MIN_PROB"))
print("  validator/sweep      :", os.environ.get("BIOHUB_VALIDATOR_ENABLE", "1"))
print("  fork gate            :", os.environ.get("BIOHUB_DIVNET_FORK_GATE", "0"), "| fork min prob:", os.environ.get("BIOHUB_DIVNET_FORK_MIN_PROB"))
print("  ILP division weight  :", os.environ.get("BIOHUB_ILP_DIVISION_WEIGHT", "(base: 1.2)"), "| rank proposals by prob:", os.environ.get("BIOHUB_DIVNET_RANK_PROPOSALS", "0"), "| sister symmetry tau:", os.environ.get("BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU", "(base: 0.6)"))
print("  classifier TTA       :", os.environ.get("BIOHUB_DIVNET_TTA", "0"))
print("  checkpoint hint      :", os.environ.get("BIOHUB_DIVNET_CKPT_HINT", "(none: first divnet_ours found)"))
print("  relaxed geometry     : divergence", os.environ.get("BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE", "1"), "mutual_nn", os.environ.get("BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN", "1"))
'''

# --- guard cell
g = cells[1]["source"]
old = 'if _drift:\n    raise RuntimeError(\n        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)\n    )'
assert g.count(old) == 1; cells[1]["source"] = g.replace(old, 'if _drift:\n    print("NOTE (ours): configuration differs from the published line on purpose:", _guard_json.dumps(_drift, sort_keys=True))')

# --- predict cell: TTA env -> setdefault
p = cells[4]["source"]
for a, b in {
    "os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'": "os.environ.setdefault('BIOHUB_EDGE_FEATURE_TTA', '1')",
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "1")',
    'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"': 'os.environ.setdefault("BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "0.75")',
}.items():
    assert p.count(a) == 1, a; p = p.replace(a, b)
cells[4]["source"] = p

# --- post-process cell: DivNet gate in safe-division repair
q = cells[5]["source"]
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
gate = anchor + '''                # ---- OURS: DivNet mitosis gate on the (parent, existing child, candidate) triple
                _dn_prob = None
                if DIVNET_SAFE_DIV_GATE:
                    _dn_prob = divnet_score_division(
                        dataset, int(source["t"]), source, existing_child, candidate,
                        globals().get("DIVNET_BUNDLE"), frame_cache,
                    )
                    if _dn_prob is None:
                        stats["divnet_safe_div_unscored"] = stats.get("divnet_safe_div_unscored", 0) + 1
                    elif _dn_prob < DIVNET_MIN_PROB:
                        stats["divnet_safe_div_rejected"] = stats.get("divnet_safe_div_rejected", 0) + 1
                        continue
                    else:
                        stats["divnet_safe_div_accepted"] = stats.get("divnet_safe_div_accepted", 0) + 1
'''
q = q.replace(anchor, gate)
# read the switch next to the other toggles (top of the cell, after the density overrides block)
q = q.replace('DENSITY_GROUP_OVERRIDES = {', 'DIVNET_SAFE_DIV_GATE = os.environ.get("BIOHUB_DIVNET_SAFE_DIV_GATE", "0") != "0"\nDENSITY_GROUP_OVERRIDES = {', 1)


# ---- OURS (v25c): rank safe-division proposals by classifier probability so the frame/global caps keep the most confident
_sc_old = "                score = parent_dist + 0.15 * sister_dist\n                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))"
assert q.count(_sc_old) == 1, "proposal score line not found"
q = q.replace(_sc_old, "                score = parent_dist + 0.15 * sister_dist\n                if DIVNET_RANK_PROPOSALS and _dn_prob is not None:\n                    score = (1.0 - float(_dn_prob)) * 100.0 + score   # OURS: probability first, geometry as tie-break\n                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))")
q = q.replace('DIVNET_SAFE_DIV_GATE = os.environ.get("BIOHUB_DIVNET_SAFE_DIV_GATE", "0") != "0"\n', 'DIVNET_SAFE_DIV_GATE = os.environ.get("BIOHUB_DIVNET_SAFE_DIV_GATE", "0") != "0"\nDIVNET_RANK_PROPOSALS = os.environ.get("BIOHUB_DIVNET_RANK_PROPOSALS", "0") != "0"\nDIVNET_TTA = os.environ.get("BIOHUB_DIVNET_TTA", "0") != "0"\n', 1)

# log line
old_log = '''        f" divergence_rejected={stats['safe_division_divergence_rejected']})"
    )'''
assert q.count(old_log) == 1
q = q.replace(old_log, '''        f" divergence_rejected={stats['safe_division_divergence_rejected']},"
        f" divnet_rejected={stats.get('divnet_safe_div_rejected', 0)}, divnet_accepted={stats.get('divnet_safe_div_accepted', 0)},"
        f" divnet_unscored={stats.get('divnet_safe_div_unscored', 0)})"
    )''')
_ok_old = '                print(f"[OK] DivNet loaded successfully from {ckpt_path}")'
assert q.count(_ok_old) == 1
q = q.replace(_ok_old, _ok_old + '\n                if isinstance(ckpt, dict) and "meta" in ckpt: print("  checkpoint meta:", {k: v for k, v in ckpt["meta"].items() if k != "val_clips" and k != "label_stats"})')

# ---- OURS: channel-aware DivNet (our v28 checkpoint uses the 4 frames as input channels; giorgosi's used in_c=1)
_cls_old = "class DivNetMitosisClassifier(torch.nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.b1 = _DivNetConvBlock(1, 16)"
assert q.count(_cls_old) == 1, "DivNet class head not found"
q = q.replace(_cls_old, "class DivNetMitosisClassifier(torch.nn.Module):\n        def __init__(self, in_c=1):\n            super().__init__()\n            self.b1 = _DivNetConvBlock(in_c, 16)")
_ld_old = "                model = DivNetMitosisClassifier()\n                model.load_state_dict(state, strict=False)"
assert q.count(_ld_old) == 1, "DivNet load lines not found"
q = q.replace(_ld_old, "                _in_c = int(state['b1.conv.0.weight'].shape[1]) if 'b1.conv.0.weight' in state else 1\n                model = DivNetMitosisClassifier(_in_c)\n                _missing, _unexpected = model.load_state_dict(state, strict=False)\n                print(f'  DivNet in_channels={_in_c} missing={len(_missing)} unexpected={len(_unexpected)}')")
_ret_old = '                return {"model": model, "device": device, "torch": torch}'
assert q.count(_ret_old) == 1
q = q.replace(_ret_old, '                return {"model": model, "device": device, "torch": torch, "in_c": _in_c}')
_ten_old = "        tensor = t_mod.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device=device, dtype=t_mod.float32)"
assert q.count(_ten_old) == 1, "divnet tensor line not found"
q = q.replace(_ten_old, "        # OURS: (4,Z,Y,X) -> (1,4,Z,Y,X) when the model takes the frames as channels; the original double-unsqueeze made a 6-D\n        # tensor that conv3d rejects (silently caught -> None -> the public DivNet never scored anything)\n        tensor = t_mod.from_numpy(padded).unsqueeze(0).to(device=device, dtype=t_mod.float32)\n        if int(divnet_bundle.get('in_c', 1)) == 1:\n            tensor = tensor.unsqueeze(0)")
_fr_old = "        frames = [read_test_frame(dataset, max(0, t + dt), frame_cache) for dt in range(4)]"
assert q.count(_fr_old) == 1, "divnet frames line not found"
q = q.replace(_fr_old, "        _T = int(json.loads((TEST_DIR / f\"{dataset}.zarr\" / \"0\" / \"zarr.json\").read_text())[\"shape\"][0])\n        frames = [read_test_frame(dataset, min(max(0, t + dt), _T - 1), frame_cache) for dt in range(4)]   # OURS: clamp like training")
_exc_old = "    except Exception as e:\n        return None"
assert q.count(_exc_old) >= 1
q = q.replace(_exc_old, "    except Exception as e:\n        globals()['_DIVNET_ERR'] = repr(e)\n        return None", 1)


# ---- OURS: fork gate (score every final division fork with our classifier)
_fg_anchor = "DIVNET_BUNDLE = load_divnet_mitosis_model()"
assert q.count(_fg_anchor) == 1
q = q.replace(_fg_anchor, _fg_anchor + """

DIVNET_FORK_GATE = os.environ.get("BIOHUB_DIVNET_FORK_GATE", "0") != "0"
DIVNET_FORK_MIN_PROB = float(os.environ.get("BIOHUB_DIVNET_FORK_MIN_PROB", os.environ.get("BIOHUB_DIV_MIN_PROB", "0.5")))

def apply_divnet_fork_gate(dataset, nodes_by_id, edges):
    \"\"\"OURS: after all post-processing, score each 2-child fork (parent, child1, child2) with the classifier; forks below
    DIVNET_FORK_MIN_PROB keep only their best child edge (sorted by edge_prob, then shorter distance).\"\"\"
    st = {"forks": 0, "dropped": 0, "unscored": 0}
    if not DIVNET_FORK_GATE or globals().get("DIVNET_BUNDLE") is None:
        return edges, st
    by_source = {}
    for e in edges:
        by_source.setdefault(int(e["source_id"]), []).append(e)
    frame_cache = {}; drop = set()
    def _key(e):
        pr = e.get("edge_prob"); pr = float(pr) if pr is not None else 0.0
        return (pr, -float(e.get("distance_um", 0.0)))
    for sid, es in by_source.items():
        if len(es) < 2 or sid not in nodes_by_id:
            continue
        st["forks"] += 1
        src = nodes_by_id[sid]; ranked = sorted(es, key=_key, reverse=True)
        c1 = nodes_by_id.get(int(ranked[0]["target_id"])); c2 = nodes_by_id.get(int(ranked[1]["target_id"]))
        if c1 is None or c2 is None:
            continue
        p = divnet_score_division(dataset, int(src["t"]), src, c1, c2, DIVNET_BUNDLE, frame_cache)
        if p is None:
            st["unscored"] += 1; continue
        if p < DIVNET_FORK_MIN_PROB:
            st["dropped"] += 1
            for e in ranked[1:]:
                drop.add(id(e))
        if len(frame_cache) > 16:
            frame_cache.clear()
    out = [e for e in edges if id(e) not in drop]
    print(f"  [{dataset}] fork gate: forks={st['forks']} dropped={st['dropped']} unscored={st['unscored']} (min prob {DIVNET_FORK_MIN_PROB})")
    return out, st
print("DIVNET_FORK_GATE:", DIVNET_FORK_GATE, "| fork min prob:", DIVNET_FORK_MIN_PROB)""")
_call_old = """        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")

        division_sources: dict[int, int] = {}"""
assert q.count(_call_old) == 1, "fork-gate call site not found"
q = q.replace(_call_old, """        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")
        edges, _fork_stats = apply_divnet_fork_gate(dataset, nodes_by_id, edges)     # OURS

        division_sources: dict[int, int] = {}""")

# hard assert after the checkpoint load
_cand_old = '''    candidates = [
        Path("/kaggle/input/biohub-divnet-v2/best_overall.pt"),'''
assert q.count(_cand_old) == 1, "divnet candidate list not found"
q = q.replace(_cand_old, '''    _ours = sorted(Path("/kaggle/input/notebooks").glob("*/*/divnet_ours/best_overall.pt")) if Path("/kaggle/input/notebooks").exists() else []
    _ours += sorted(Path("/kaggle/input/datasets").glob("*/*/divnet_ours/best_overall.pt")) if Path("/kaggle/input/datasets").exists() else []
    _ours += sorted(Path("/kaggle/input").glob("*/divnet_ours/best_overall.pt"))
    if os.environ.get("BIOHUB_DIVNET_REQUIRE_OURS", "0") != "0" and not _ours:
        raise RuntimeError("BIOHUB_DIVNET_REQUIRE_OURS=1 but no divnet_ours/best_overall.pt found: attach the v28 notebook Output")
    print("DivNet checkpoints (ours first):", [str(p) for p in _ours])
    candidates = [*_ours,
        Path("/kaggle/input/biohub-divnet-v2/best_overall.pt"),''')
# ---- OURS (v25d): checkpoint hint + 3-seed ensemble support
_h_old = '    print("DivNet checkpoints (ours first):", [str(p) for p in _ours])'
assert q.count(_h_old) == 1
q = q.replace(_h_old, _h_old + """
    _hint = os.environ.get("BIOHUB_DIVNET_CKPT_HINT", "")
    if _hint:
        _hinted = [p for p in _ours if _hint in str(p)]
        if not _hinted:
            raise RuntimeError(f"BIOHUB_DIVNET_CKPT_HINT={_hint!r} matches none of {[str(p) for p in _ours]}")
        _ours = _hinted; print("  hint", repr(_hint), "->", str(_ours[0]))""")
_r_old = '                return {"model": model, "device": device, "torch": torch, "in_c": _in_c}'
assert q.count(_r_old) == 1
q = q.replace(_r_old, """                _models = [model]
                for _k, _st in enumerate(ckpt.get("ensemble_states", []) if isinstance(ckpt, dict) else []):
                    _m = DivNetMitosisClassifier(_in_c); _mi, _mu = _m.load_state_dict(_st, strict=False); _m.to(device); _m.eval(); _models.append(_m)
                    print(f"  ensemble member {_k}: missing={len(_mi)} unexpected={len(_mu)}")
                if len(_models) > 1:
                    _models = _models[1:]          # ensemble_states already contains the best seed; drop the duplicate
                print(f"  DivNet models in bundle: {len(_models)}")
                return {"model": model, "device": device, "torch": torch, "in_c": _in_c, "models": _models}""")
_p_old = """        with t_mod.inference_mode():
            logit = model(tensor)
            prob = float(t_mod.sigmoid(logit).squeeze().cpu().item())
        return prob"""
assert q.count(_p_old) == 1, "divnet prob block not found"
q = q.replace(_p_old, """        with t_mod.inference_mode():
            _ms = divnet_bundle.get("models") or [model]
            if globals().get("DIVNET_TTA", False):
                # OURS (v25e): 16 views = D4 in yx (4 rotations x flip) x z-flip -> one batch per model, mean of sigmoids
                _views = []
                for _fz in (False, True):
                    _b = tensor.flip(-3) if _fz else tensor
                    for _k in range(4):
                        _r = t_mod.rot90(_b, _k, dims=(-2, -1))
                        _views.append(_r); _views.append(_r.flip(-1))
                _batch = t_mod.cat(_views, dim=0)
                prob = float(np.mean([float(t_mod.sigmoid(_m(_batch)).mean().cpu().item()) for _m in _ms]))
            else:
                prob = float(np.mean([float(t_mod.sigmoid(_m(tensor)).squeeze().cpu().item()) for _m in _ms]))   # OURS (v25d): ensemble mean
        return prob""")
old_load = 'DIVNET_BUNDLE = load_divnet_mitosis_model()'
assert q.count(old_load) == 1
q = q.replace(old_load, old_load + '''
if DIVNET_SAFE_DIV_GATE and DIVNET_BUNDLE is None:
    raise RuntimeError("DIVNET gate requested but no checkpoint loaded: attach the dataset giorgosi/biohub-divnet-v2")
print("DIVNET_SAFE_DIV_GATE:", DIVNET_SAFE_DIV_GATE, "| bundle loaded:", DIVNET_BUNDLE is not None, "| min prob:", DIVNET_MIN_PROB)
if DIVNET_SAFE_DIV_GATE and DIVNET_BUNDLE is not None:
    # smoke test: score a synthetic crop once so a shape bug fails loudly here instead of being swallowed as 'unscored'
    _t = torch.zeros((1, 4 if DIVNET_BUNDLE.get("in_c", 1) == 4 else 1, *((4, 16, 32, 32) if DIVNET_BUNDLE.get("in_c", 1) == 1 else (16, 32, 32))), device=DIVNET_BUNDLE["device"])
    with torch.inference_mode():
        print("  DivNet smoke test output:", float(torch.sigmoid(DIVNET_BUNDLE["model"](_t)).item()))''')
cells[5]["source"] = q

md = {"cell_type": "markdown", "metadata": {}, "source": "# Biohub · v25 — density-adaptive public line + DivNet wired into safe-division repair\n\nEdit `VARIANT` in cell 2 only. Inputs: competition + pilkwang ×3 (+ `giorgosi/biohub-divnet-v2` for R2–R4)."}
new = [md, cells[0], {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": VARIANT_CELL.strip("\n")}] + cells[1:]
json.dump({"cells": new, "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}, open(OUT, "w"), indent=1)
print("wrote", OUT, "cells:", len(new))
