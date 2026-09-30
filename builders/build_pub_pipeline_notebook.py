"""Builds biohub_v21_pub942_variants.ipynb: the public 0.942 pipeline (analyticaobscura / nusrati / pilkwang lineage,
fork 'biohub0942lboneknobpastthepublicline') kept cell-for-cell, plus ONE cell of ours that applies a named VARIANT of
environment-variable overrides, the audit cells that need the busyaprime dataset removed, the config guard turned into a
report, and the (LB-disagreeing) validator switched off by default."""
import json, sys
from pathlib import Path
import nbformat as nbf

SRC = Path("public_notebooks/dcbb6a63-biohub0942lboneknobpastthepublicline.ipynb")
src = nbf.read(str(SRC), as_version=4)
c = src.cells
assert len(c) == 15 and c[4].source.startswith("# Cell-tracking-during-development submission pipeline")

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

VARIANT_CELL = '''# ------------------------------------------------------------------ OURS: named variants on top of the published 0.942 configuration
# Change VARIANT, Save & Run All, submit. "P0" = the published configuration untouched (LB 0.942).
VARIANT = "P0"
RUN_VALIDATOR = (VARIANT == "P5")   # the notebook's own validator disagrees with the LB (see its section 1) -> off by default (saves ~1-2 h)
VARIANTS = {
    "P0": {},                                                                                   # published 0.942 (det 0.96, bidir 0.15)
    "P1": {"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.25"},                                          # author: bidir 0.25 alone also scored 0.942 -> combine with det 0.96
    "P2": {"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.25", "BIOHUB_SAFE_DIV_DIVERGE_UM": "4.0"},     # + division divergence gate where a public sweep peaked (4.0-4.5)
    "P3": {"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.20"},                                          # midpoint
    "P4": {"BIOHUB_DET_THRESHOLD": "0.955"},                                                     # detection threshold half a step below the published optimum
    # --- round 2 (P1/P3 = 0.942, P4 = 0.941: linking knobs are saturated) -> division budget + track-length knobs nobody moved
    "P5": {},                                                                                   # DIAGNOSTIC: run with RUN_VALIDATOR=True to see edge vs division score split
    "P6": {"BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.012", "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.006"},  # allow ~60% more divisions through the safe-div gates
    "P7": {"BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.005", "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.0025"}, # fewer divisions (if P6 hurts, this is the other direction)
    "P8": {"BIOHUB_OUTPUT_MIN_TRACK_LEN": "4"},                                                  # keep shorter tracks (published 6)
    "P9": {"BIOHUB_OUTPUT_MIN_TRACK_LEN": "8"},                                                  # drop more short tracks
    "P10": {"BIOHUB_ILP_DISAPPEARANCE_WEIGHT": "1.5"},                                           # published 2.0
    "P11": {"BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.35", "BIOHUB_DEEPCENTER_GAP_THRESHOLD": "0.35"},  # stricter DeepCenter vetoes (author probed gap 0.35 alone: 0.941)
    # --- round 3 (P5 validator: division tp/fp/fn = 1/0/4 -> recall-limited; DeepCenter vetoes 65-95% of geometric candidates)
    "P12": {"BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.15"},                                    # looser DeepCenter division veto (published 0.25, code default 0.12)
    "P13": {"BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.15", "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.012", "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.006"},  # P12 + P6 budget
    "P14": {"BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "0.0", "BIOHUB_SAFE_DIV_DIVERGE_UM": "1.5"},   # relax the two geometric precision gates (symmetry off, divergence 2.25->1.5)
    "P15": {"BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0"},                                            # DeepCenter division veto OFF entirely (upper bound of how many divisions the geometry admits)
}
import os
for _k, _v in VARIANTS[VARIANT].items():
    print(f"override {_k}: {os.environ.get(_k)} -> {_v}"); os.environ[_k] = _v
os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1" if RUN_VALIDATOR else "0"
print("VARIANT:", VARIANT, VARIANTS[VARIANT], "| validator:", RUN_VALIDATOR)
'''

GUARD_CELL = c[5].source.replace('''if _drift:
    raise RuntimeError(
        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)
    )

print("Configuration guard: PASS")''', '''if _drift:
    print("Configuration differs from the published 0.942 run (intended when VARIANT != P0): " + _guard_json.dumps(_drift, sort_keys=True))
else:
    print("Configuration guard: PASS (identical to the published 0.942 configuration)")''')
assert GUARD_CELL != c[5].source

C8 = c[8].source
_old_guard = """if not _bidirectional_math.isclose(
    _bidirectional_weight_guard, 0.15, rel_tol=0.0, abs_tol=1e-12
):
    raise ValueError({
        "expected_bidirectional_weight": 0.15,
        "actual_bidirectional_weight": _bidirectional_weight_guard,
    })"""
assert C8.count(_old_guard) == 1
C8 = C8.replace(_old_guard, """if not _bidirectional_math.isclose(
    _bidirectional_weight_guard, 0.15, rel_tol=0.0, abs_tol=1e-12
):
    print("NOTE: bidirectional edge weight differs from the published 0.15 ->", _bidirectional_weight_guard, "(intended when VARIANT != P0)")""")
c8 = nbf.v4.new_code_cell(C8)

cells = [
    md("""# Biohub · v21 — public 0.942 pipeline + named variants

Base: the public notebook *"Biohub 0.942 LB, one knob past the public line"* (analyticaobscura's biohub-lb-941 → nusrati's 0.940,
on pilkwang's frozen dual-seed weights), kept **cell for cell**. Ours: the `VARIANT` cell (environment-variable overrides),
the config guard turned into a report, the audit cells that needed the `busyaprime/biohub-knob-provenance` dataset removed,
and the validator off by default.

**Inputs** (Add Input → Datasets, all by pilkwang): `biohub-tracking-support-pack-50ep-v1`, `biohub-temporal-unet3d-seed314159-v1`,
`biohub-deepcenter-unet3d-center-prior-v1` + the competition data. GPU T4, Internet off.
"""),
    c[4],            # published configuration (env vars)
    code(VARIANT_CELL),
    code(GUARD_CELL),
    c[6], c[7], c8, c[9],     # pipeline (c8: hard-coded 0.15 guard -> note)
    c[10],           # frame-retention guard
    c[11], c[12], c[13],      # validator (disabled via env) + manifest
]
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out = sys.argv[1] if len(sys.argv) > 1 else "biohub_v21d_pub942_variants.ipynb"
Path(out).write_text(nbf.writes(nb)); print("written", out, len(cells), "cells")
