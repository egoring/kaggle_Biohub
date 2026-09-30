"""Unit test for cellmot_infer.postprocess / prune_short_tracks on a synthetic detection graph (no network needed)."""
import sys, time; sys.path.insert(0, "."); from pathlib import Path; import numpy as np
import cellmot_infer as CI
P = CI.setup_predict(Path("fake_input/prep-v2/repo"))
# synthetic graph: track A len 12 (t0..11), track B len 4, track C len 2, plus 5 isolated nodes; a weak wrong edge A->B
coords = []; edges = []; idx = lambda: len(coords) - 1
A = []; B = []; C = []
for t in range(12): coords.append((t, 10, 50, 50 + t)); A.append(idx())
for t in range(4):  coords.append((t, 30, 100, 100 + t)); B.append(idx())
for t in range(2):  coords.append((t, 50, 150, 150 + t)); C.append(idx())
for t in range(5):  coords.append((t * 2, 60, 200, 200 + 5 * t))
for a, b in zip(A, A[1:]): edges.append((a, b, 0.95, 1.0))
for a, b in zip(B, B[1:]): edges.append((a, b, 0.9, 1.0))
for a, b in zip(C, C[1:]): edges.append((a, b, 0.6, 1.0))
edges.append((A[0], B[1], 0.3, 80.0))   # weak cross edge (competes with A0->A1 and B0->B1)
coords = np.array(coords, np.int16); edges = np.array(edges, np.float32)
cfg = CI.make_cfg(det_threshold=0.95, use_ilp=True, det_tta=False)
t = time.time(); n, e, w = CI.postprocess(coords, edges, cfg); print("ilp time", round(time.time() - t, 1))
print("ILP nodes", len(n), "edges", len(e))
assert len(n) == 16 and len(e) == 14, (len(n), len(e))   # A(12)+B(4) kept; C (2 nodes, reward 0.6 < disappearance 1.4) and isolated nodes dropped; cross edge rejected
for L, expect_n, expect_e in [(3, 16, 14), (5, 12, 11), (13, 0, 0)]:
    n2, e2, w2 = CI.prune_short_tracks(n, e, w, L); print(f"L={L}: {len(n2)} nodes {len(e2)} edges"); assert (len(n2), len(e2)) == (expect_n, expect_e)
    ids = [x[4] for x in n2]; assert ids == list(range(1, len(n2) + 1)); assert all(1 <= s <= len(n2) and 1 <= t <= len(n2) for s, t in e2)
# edge threshold on cached edges: 0.5 drops the 0.3 cross edge only
cfg2 = CI.make_cfg(use_ilp=False, edge_threshold=0.5, det_tta=False); n3, e3, w3 = CI.postprocess(coords, edges, cfg2, log=lambda *a: None)
assert len(e3) == 15 and len(n3) == 23, (len(e3), len(n3)); print("greedy e>0.5: nodes", len(n3), "edges", len(e3))
for app in (0.0, 0.5, 1.0):
    c = CI.make_cfg(use_ilp=True, ilp_appearance_weight=app, det_tta=False); nn, ee, _ = CI.postprocess(coords, edges, c, log=lambda *a: None)
    print(f"app={app}: nodes {len(nn)} edges {len(ee)}")
print("OK")
