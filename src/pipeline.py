"""
Biohub Cell Tracking - inference pipeline (shared by notebook cells).

Steps per test sample:
  1. load zarr v3 volume (T,Z,Y,X) uint16
  2. per-frame 3D blob detection (DoG + local maxima, anisotropic voxel scale)
  3. spherical pseudo-masks around detections
  4. Trackastra (pretrained CTC 3D transformer) greedy linking with divisions
     -> fallback: scaled nearest-neighbour linear assignment
  5. rows for submission.csv
"""
from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- config
SCALE_ZYX = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)  # um / voxel

CFG = dict(
    cell_radius_um=3.5,      # approx nucleus radius, drives DoG sigma & NMS footprint
    detect_percentile=99.8,  # per-frame intensity normalisation
    thresh_k=1.0,            # threshold = mean + k*std of DoG response (foreground only)
    min_rel_peak=0.08,       # peak must exceed this fraction of max DoG response
    max_nodes_per_frame=6000,
    merge_min_dist_um=None,  # if set: within a frame, suppress weaker detections closer than this (um) to a stronger one
    # per-clip adaptive radius: r = clip(a * N^(-1/3), rmin, rmax), N = median detections/frame at probe radius.
    # Rationale: early zebrafish embryo volume is ~constant, so more cells <=> smaller cells. None = fixed radius.
    adaptive_radius=None,    # e.g. dict(a=26.5, rmin=3.0, rmax=6.0, probe_r=4.0, n_probe=5)
    mask_radius_vox=(1, 3, 3),  # (z,y,x) radius of pseudo-mask spheres (used when mask_scale is None)
    mask_scale=None,         # if set: pseudo-mask radius = mask_scale * cell_radius_um (in um), converted per axis
    link_max_dist_um=12.0,   # fallback linker gate
    trackastra_mode="greedy",
    trackastra_max_distance=64,  # voxel (isotropic-ish) neighbourhood for candidate graph
    # division pruning (greedy over-predicts divisions massively on this data):
    div_min_weight=0.5,      # 2nd child edge kept only if its trackastra weight >= this (None = keep all)
    div_max_dist_um=8.0,     # ... and both children within this distance of the parent
    div_max_frac=0.006,      # ... and at most this fraction of nodes may be division parents (keep strongest)
)


# ---------------------------------------------------------------- io
def open_zarr_array(zarr_dir: Path):
    """Return array-like (T,Z,Y,X). Works for zarr v3 (array at '0')."""
    import zarr

    zarr_dir = Path(zarr_dir)
    try:
        g = zarr.open_group(str(zarr_dir), mode="r")
        arr = g["0"]
    except Exception:
        arr = zarr.open_array(str(zarr_dir / "0"), mode="r")
    return arr


def list_test_samples(comp_dir: Path):
    test_dir = Path(comp_dir) / "test"
    zarrs = sorted(p for p in test_dir.iterdir() if p.suffix == ".zarr" and p.is_dir())
    return zarrs


# ---------------------------------------------------------------- detection
def _torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gauss_kernel1d(sigma: float, device):
    import torch

    if sigma <= 0.05:
        return torch.ones(1, device=device)
    r = max(1, int(np.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gauss3d(vol, sigmas_zyx):
    """Separable Gaussian blur, vol: torch (Z,Y,X) float32."""
    import torch
    import torch.nn.functional as F

    x = vol[None, None]
    for axis, s in enumerate(sigmas_zyx):
        k = _gauss_kernel1d(float(s), vol.device)
        if k.numel() == 1:
            continue
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]
        # F.pad order: (x_l, x_r, y_l, y_r, z_l, z_r)
        pad_idx = (2 - axis) * 2
        pad[pad_idx] = pad[pad_idx + 1] = k.numel() // 2
        x = F.pad(x, pad, mode="replicate")
        x = F.conv3d(x, k.view(*shape))
    return x[0, 0]


def _detect_frame_cpu(frame_u16: np.ndarray, cfg=CFG):
    """scipy.ndimage version (much faster than torch conv3d on CPU)."""
    from scipy import ndimage as ndi

    v = frame_u16.astype(np.float32)
    sub = v.ravel()[:: max(1, v.size // 200000)]
    lo, hi = np.percentile(sub, 1.0), np.percentile(sub, cfg["detect_percentile"])
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)

    r_vox = cfg["cell_radius_um"] / SCALE_ZYX
    s1 = r_vox / np.sqrt(3.0) * 0.8
    s2 = s1 * 1.6
    dog = ndi.gaussian_filter(v, s1, mode="nearest") - ndi.gaussian_filter(v, s2, mode="nearest")

    fp = tuple(int(max(1, np.floor(r * 0.8))) * 2 + 1 for r in r_vox)
    mp = ndi.maximum_filter(dog, size=fp, mode="nearest")
    is_peak = dog >= mp

    pos = dog[dog > 0]
    if pos.size < 10:
        return np.zeros((0, 3), np.int32), np.zeros((0,), np.float32)
    thr = float(pos.mean() + cfg["thresh_k"] * pos.std())
    thr = max(thr, cfg["min_rel_peak"] * float(dog.max()))
    cand = is_peak & (dog > thr)
    idx = np.argwhere(cand)
    scores = dog[cand]
    if idx.shape[0] > cfg["max_nodes_per_frame"]:
        top = np.argsort(-scores)[: cfg["max_nodes_per_frame"]]
        idx, scores = idx[top], scores[top]
    return idx.astype(np.int32), scores.astype(np.float32)


def detect_frame(frame_u16: np.ndarray, cfg=CFG, device=None):
    """Detect cell centroids in one (Z,Y,X) frame. Returns int array (N,3) zyx + scores (N,)."""
    import torch
    import torch.nn.functional as F

    device = device or _torch_device()
    if device.type == "cpu":
        return _detect_frame_cpu(frame_u16, cfg)
    v = torch.from_numpy(frame_u16.astype(np.float32)).to(device)

    lo = torch.quantile(v.flatten()[:: max(1, v.numel() // 200000)], 0.01)
    hi = torch.quantile(v.flatten()[:: max(1, v.numel() // 200000)], cfg["detect_percentile"] / 100.0)
    v = ((v - lo) / (hi - lo + 1e-6)).clamp_(0, 1)

    r_vox = cfg["cell_radius_um"] / SCALE_ZYX  # radius in voxels per axis (z,y,x)
    s1 = r_vox / np.sqrt(3.0) * 0.8
    s2 = s1 * 1.6
    dog = _gauss3d(v, s1) - _gauss3d(v, s2)

    # local maxima via max-pool with anisotropic footprint (~ cell radius)
    fp = [int(max(1, np.floor(r * 0.8))) * 2 + 1 for r in r_vox]
    mp = F.max_pool3d(dog[None, None], kernel_size=fp, stride=1, padding=[f // 2 for f in fp])[0, 0]
    is_peak = dog >= mp

    pos = dog[dog > 0]
    if pos.numel() < 10:
        return np.zeros((0, 3), np.int32), np.zeros((0,), np.float32)
    thr = float(pos.mean() + cfg["thresh_k"] * pos.std())
    thr = max(thr, cfg["min_rel_peak"] * float(dog.max()))
    cand = is_peak & (dog > thr)

    idx = torch.nonzero(cand, as_tuple=False)
    scores = dog[cand]
    if idx.shape[0] > cfg["max_nodes_per_frame"]:
        top = torch.topk(scores, cfg["max_nodes_per_frame"]).indices
        idx, scores = idx[top], scores[top]
    return idx.cpu().numpy().astype(np.int32), scores.cpu().numpy().astype(np.float32)


def merge_close_detections(zyx, scores, min_dist_um):
    """Greedy NMS on physical distance: keep strongest, drop any weaker detection within min_dist_um."""
    if min_dist_um is None or len(zyx) < 2:
        return zyx
    from scipy.spatial import cKDTree

    P_ = zyx.astype(np.float32) * SCALE_ZYX
    order = np.argsort(-scores)
    tree = cKDTree(P_)
    alive = np.ones(len(zyx), bool)
    for i in order:
        if not alive[i]:
            continue
        for j in tree.query_ball_point(P_[i], min_dist_um):
            if j != i and alive[j] and scores[j] <= scores[i]:
                alive[j] = False
    return zyx[alive]


def detect_volume(arr, cfg=CFG, device=None, log=print):
    """arr: (T,Z,Y,X) array-like. Returns list of (N_t,3) int zyx arrays and imgs float16 (T,Z,Y,X)."""
    T = arr.shape[0]
    dets, imgs = [], []
    t0 = time.time()
    for t in range(T):
        frame = np.asarray(arr[t])
        zyx, sc = detect_frame(frame, cfg, device)
        zyx = merge_close_detections(zyx, sc, cfg.get("merge_min_dist_um"))
        dets.append(zyx)
        imgs.append(frame)
        if t % 25 == 0 or t == T - 1:
            log(f"    frame {t:3d}/{T}: {len(zyx):5d} detections  ({time.time()-t0:.1f}s)")
    return dets, np.stack(imgs)


# ---------------------------------------------------------------- masks
def _sphere_offsets(radius_zyx):
    rz, ry, rx = radius_zyx
    zz, yy, xx = np.mgrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
    inside = (zz / (rz + 0.5)) ** 2 + (yy / (ry + 0.5)) ** 2 + (xx / (rx + 0.5)) ** 2 <= 1.0
    return np.stack([zz[inside], yy[inside], xx[inside]], axis=1)


def make_masks(dets, shape_zyx, radius_zyx=CFG["mask_radius_vox"]):
    """Label volume (T,Z,Y,X) int32; label = 1..N_t per frame (matches detection order)."""
    T = len(dets)
    Z, Y, X = shape_zyx
    masks = np.zeros((T, Z, Y, X), dtype=np.int32)
    off = _sphere_offsets(radius_zyx)
    for t, zyx in enumerate(dets):
        if len(zyx) == 0:
            continue
        pts = (zyx[:, None, :] + off[None, :, :]).reshape(-1, 3)
        labels = np.repeat(np.arange(1, len(zyx) + 1, dtype=np.int32), len(off))
        ok = (
            (pts[:, 0] >= 0) & (pts[:, 0] < Z)
            & (pts[:, 1] >= 0) & (pts[:, 1] < Y)
            & (pts[:, 2] >= 0) & (pts[:, 2] < X)
        )
        pts, labels = pts[ok], labels[ok]
        # later labels overwrite earlier -> fine (centre voxel always set last below)
        masks[t, pts[:, 0], pts[:, 1], pts[:, 2]] = labels
        masks[t, zyx[:, 0], zyx[:, 1], zyx[:, 2]] = np.arange(1, len(zyx) + 1, dtype=np.int32)
    return masks


# ---------------------------------------------------------------- linking
def link_trackastra(model, imgs, masks, dets, cfg=CFG, log=print):
    """Returns (nodes: list of (t, z, y, x, node_id), edges: list of (src_id, tgt_id))."""
    t0 = time.time()
    graph, _ = model.track(
        imgs,
        masks,
        mode=cfg["trackastra_mode"],
        max_distance=cfg["trackastra_max_distance"],
    )
    log(f"    trackastra: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges ({time.time()-t0:.1f}s)")

    # map (time,label) -> our detection coordinate
    key2id = {}
    nodes = []
    nid = 1
    for t, zyx in enumerate(dets):
        for lab in range(1, len(zyx) + 1):
            key2id[(t, lab)] = nid
            z, y, x = zyx[lab - 1]
            nodes.append((t, int(z), int(y), int(x), nid))
            nid += 1

    edges, weights = [], []
    for u, v, d in graph.edges(data=True):
        nu, nv = graph.nodes[u], graph.nodes[v]
        ku = (int(nu["time"]), int(nu["label"]))
        kv = (int(nv["time"]), int(nv["label"]))
        if ku in key2id and kv in key2id:
            edges.append((key2id[ku], key2id[kv]))
            weights.append(float(d.get("weight", 1.0)))
    edges = prune_divisions(nodes, edges, weights, cfg, log=log)
    return nodes, edges


def prune_divisions(nodes, edges, weights, cfg=CFG, log=print):
    """Keep at most one outgoing edge per parent unless the 2nd edge is confident, close, and
    the global division budget (div_max_frac * n_nodes) is not exceeded."""
    if cfg.get("div_min_weight") is None:
        return edges
    pos = {nid: np.array([z, y, x], np.float32) * SCALE_ZYX for t, z, y, x, nid in nodes}
    by_parent = {}
    for (s, t_), w in zip(edges, weights):
        by_parent.setdefault(s, []).append((w, t_))
    n_div_before = sum(1 for c in by_parent.values() if len(c) >= 2)
    kept, candidates = [], []
    for s, ch in by_parent.items():
        ch.sort(reverse=True)
        kept.append((s, ch[0][1]))                      # strongest child always kept
        for w, t_ in ch[1:2]:                            # at most one extra child
            d1 = np.linalg.norm(pos[ch[0][1]] - pos[s])
            d2 = np.linalg.norm(pos[t_] - pos[s])
            if w >= cfg["div_min_weight"] and max(d1, d2) <= cfg["div_max_dist_um"]:
                candidates.append((w, s, t_))
    budget = int(cfg["div_max_frac"] * len(nodes))
    candidates.sort(reverse=True)
    kept += [(s, t_) for _, s, t_ in candidates[:budget]]
    log(f"    divisions: {n_div_before} -> {min(len(candidates), budget)} after pruning "
        f"(min_w={cfg['div_min_weight']}, max_d={cfg['div_max_dist_um']}um, budget={budget})")
    return kept


def link_nearest(dets, cfg=CFG, log=print):
    """Fallback: frame-to-frame linear assignment on physical distance, + division by
    attaching unmatched t+1 nodes to the nearest matched parent within the gate."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    gate = cfg["link_max_dist_um"]
    nodes, edges = [], []
    ids = []
    nid = 1
    for t, zyx in enumerate(dets):
        cur = np.arange(nid, nid + len(zyx))
        ids.append(cur)
        for (z, y, x), i in zip(zyx, cur):
            nodes.append((t, int(z), int(y), int(x), int(i)))
        nid += len(zyx)

    for t in range(len(dets) - 1):
        a, b = dets[t].astype(np.float32) * SCALE_ZYX, dets[t + 1].astype(np.float32) * SCALE_ZYX
        if len(a) == 0 or len(b) == 0:
            continue
        tree = cKDTree(a)
        d, j = tree.query(b, k=min(5, len(a)), distance_upper_bound=gate)
        d = np.atleast_2d(d.T).T if d.ndim == 1 else d
        j = np.atleast_2d(j.T).T if j.ndim == 1 else j
        big = 1e6
        cost = np.full((len(a), len(b)), big, np.float32)
        for bi in range(len(b)):
            for dd, aj in zip(d[bi], j[bi]):
                if np.isfinite(dd) and aj < len(a):
                    cost[aj, bi] = dd
        r, c = linear_sum_assignment(cost)
        matched_b = set()
        for ai, bi in zip(r, c):
            if cost[ai, bi] < big:
                edges.append((int(ids[t][ai]), int(ids[t + 1][bi])))
                matched_b.add(bi)
        # divisions: unmatched children -> nearest parent within gate
        parent_children = {}
        for ai, bi in zip(r, c):
            if cost[ai, bi] < big:
                parent_children[ai] = 1
        for bi in range(len(b)):
            if bi in matched_b:
                continue
            dd, aj = d[bi][0], j[bi][0]
            if np.isfinite(dd) and aj < len(a) and parent_children.get(aj, 0) < 2:
                edges.append((int(ids[t][aj]), int(ids[t + 1][bi])))
                parent_children[aj] = parent_children.get(aj, 0) + 1
    log(f"    nearest-neighbour linker: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges


# ---------------------------------------------------------------- validation on train (.geff)
def read_geff(geff_dir: Path):
    """Return dict(ids, t, z, y, x, edges) from a .geff (zarr v2 or v3)."""
    import zarr

    g = zarr.open_group(str(geff_dir), mode="r")
    out = dict(ids=np.asarray(g["nodes/ids"][:]))
    for k in ("t", "z", "y", "x"):
        out[k] = np.asarray(g[f"nodes/props/{k}/values"][:])
    try:
        out["edges"] = np.asarray(g["edges/ids"][:]).reshape(-1, 2)
    except Exception:
        out["edges"] = np.zeros((0, 2), np.int64)
    def _find(obj, key):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                r = _find(v, key)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = _find(v, key)
                if r is not None:
                    return r
        return None
    try:
        out["estimated_number_of_nodes"] = _find(dict(g.attrs), "estimated_number_of_nodes")
    except Exception:
        out["estimated_number_of_nodes"] = None
    return out


def evaluate_against_geff(nodes, edges, gt, match_um=7.0):
    """Sparse-GT diagnostics: node recall, edge recall/precision-on-matched, divisions.
    (Official metric is more involved; this is a sanity check only.)"""
    from scipy.optimize import linear_sum_assignment

    pred_by_t = {}
    for t, z, y, x, nid in nodes:
        pred_by_t.setdefault(t, []).append((nid, z, y, x))
    gt_by_t = {}
    for i, t, z, y, x in zip(gt["ids"], gt["t"], gt["z"], gt["y"], gt["x"]):
        gt_by_t.setdefault(int(t), []).append((int(i), z, y, x))

    pred2gt, n_gt, n_hit = {}, 0, 0
    for t, gl in gt_by_t.items():
        pl = pred_by_t.get(t, [])
        n_gt += len(gl)
        if not pl:
            continue
        P_ = np.array([p[1:] for p in pl], np.float32) * SCALE_ZYX
        G_ = np.array([g[1:] for g in gl], np.float32) * SCALE_ZYX
        D = np.linalg.norm(P_[:, None, :] - G_[None, :, :], axis=2)
        r, c = linear_sum_assignment(D)
        for i, j in zip(r, c):
            if D[i, j] <= match_um:
                pred2gt[pl[i][0]] = gl[j][0]
                n_hit += 1

    gt_edges = set(map(tuple, gt["edges"].tolist()))
    gt2pred = {v: k for k, v in pred2gt.items()}
    pred_edges = set(edges)
    # edge recall: GT edges whose both endpoints were matched and the predicted edge exists
    e_hit = e_eval = 0
    for s, t_ in gt_edges:
        if s in gt2pred and t_ in gt2pred:
            e_eval += 1
            if (gt2pred[s], gt2pred[t_]) in pred_edges:
                e_hit += 1
    # divisions
    def _divs(E):
        cnt = {}
        for s, _ in E:
            cnt[s] = cnt.get(s, 0) + 1
        return {s for s, c in cnt.items() if c >= 2}

    gt_div = _divs(gt_edges)
    pred_div = _divs(pred_edges)
    div_hit = sum(1 for s in gt_div if gt2pred.get(s) in pred_div)
    return dict(
        gt_nodes=n_gt,
        node_recall=n_hit / max(1, n_gt),
        gt_edges_evaluable=e_eval,
        edge_recall_on_matched=e_hit / max(1, e_eval),
        gt_edges_total=len(gt_edges),
        edge_recall_total=e_hit / max(1, len(gt_edges)),
        gt_divisions=len(gt_div),
        pred_divisions=len(pred_div),
        division_recall=div_hit / max(1, len(gt_div)),
        pred_nodes=len(nodes),
        est_total_cells=gt.get("estimated_number_of_nodes"),
        pred_over_est=(len(nodes) / gt["estimated_number_of_nodes"]) if gt.get("estimated_number_of_nodes") else None,
    )


# ---------------------------------------------------------------- official-style metric (local CV)
def score_official(nodes, edges, gt, match_um=7.0, a=0.1, w=0.1):
    """Re-implementation of the competition metric (metrics.md):
    - nodes matched per timepoint by optimal bipartite assignment, <= 7 um
    - edge TP: both endpoints matched to GT nodes joined by a GT edge
      edge FP: predicted edge whose matched source has a *different* GT target, or whose matched target has a different GT source
      edge FN: GT edges not recovered
    - adjusted J = max(0, J * (1 - a*(T_pred - T_true)/T_true))  with T_true = estimated_number_of_nodes
    - division TP/FP/FN (simplified: parent matched + both daughters matched & linked)
    Returns dict with counts and score."""
    from scipy.optimize import linear_sum_assignment

    pred_by_t, gt_by_t = {}, {}
    for t, z, y, x, nid in nodes:
        pred_by_t.setdefault(int(t), []).append((nid, z, y, x))
    for i, t, z, y, x in zip(gt["ids"], gt["t"], gt["z"], gt["y"], gt["x"]):
        gt_by_t.setdefault(int(t), []).append((int(i), z, y, x))
    p2g = {}
    for t, gl in gt_by_t.items():
        pl = pred_by_t.get(t, [])
        if not pl:
            continue
        P_ = np.array([p[1:] for p in pl], np.float32) * SCALE_ZYX
        G_ = np.array([g[1:] for g in gl], np.float32) * SCALE_ZYX
        D = np.linalg.norm(P_[:, None, :] - G_[None, :, :], axis=2)
        r, c = linear_sum_assignment(D)
        for i, j in zip(r, c):
            if D[i, j] <= match_um:
                p2g[pl[i][0]] = gl[j][0]
    g2p = {v: k for k, v in p2g.items()}
    n_gt = len(gt["ids"]); n_matched = len(p2g)

    gt_edges = set(map(tuple, gt["edges"].tolist()))
    gt_out, gt_in = {}, {}
    for s_, t_ in gt_edges:
        gt_out.setdefault(s_, set()).add(t_)
        gt_in.setdefault(t_, set()).add(s_)

    tp = fp = 0
    pred_edges = set(edges)
    for s_, t_ in pred_edges:
        gs, gtn = p2g.get(s_), p2g.get(t_)
        if gs is not None and gtn is not None and gtn in gt_out.get(gs, ()):
            tp += 1
            continue
        # FP: matched source whose GT target is someone else, or matched target whose GT source is someone else
        if (gs is not None and gs in gt_out and gtn not in gt_out[gs]) or \
           (gtn is not None and gtn in gt_in and gs not in gt_in[gtn]):
            fp += 1
    fn = len(gt_edges) - tp
    J = tp / max(1, tp + fp + fn)
    est = gt.get("estimated_number_of_nodes")
    adj = J
    if est:
        adj = max(0.0, J * (1 - a * (len(nodes) - est) / est))

    # divisions
    gt_div = {s_ for s_, ch in gt_out.items() if len(ch) >= 2}
    pred_out = {}
    for s_, t_ in pred_edges:
        pred_out.setdefault(s_, set()).add(t_)
    pred_div = {s_ for s_, ch in pred_out.items() if len(ch) >= 2}
    dtp = 0
    for gp in gt_div:
        pp = g2p.get(gp)
        if pp is None:
            continue
        daughters = gt_out[gp]
        if all(g2p.get(d) in pred_out.get(pp, ()) for d in daughters):
            dtp += 1
    # FP divisions: predicted division whose parent is a matched GT node that does not divide in GT
    dfp = sum(1 for pp in pred_div if pp in p2g and p2g[pp] not in gt_div)
    dfn = len(gt_div) - dtp
    dJ = dtp / max(1, dtp + dfp + dfn)
    return dict(edge_tp=tp, edge_fp=fp, edge_fn=fn, edge_J=round(J, 4), adj_edge_J=round(adj, 4),
                div_tp=dtp, div_fp=dfp, div_fn=dfn, div_J=round(dJ, 4),
                score=round(adj + w * dJ, 4), node_recall=round(n_matched / max(1, n_gt), 4),
                n_pred=len(nodes), est=est, gt_div=len(gt_div))


def micro_average(rows, w=0.1):
    """Combine per-clip dicts from score_official the way the competition does (micro over videos)."""
    tp = sum(r["edge_tp"] for r in rows); fp = sum(r["edge_fp"] for r in rows); fn = sum(r["edge_fn"] for r in rows)
    dtp = sum(r["div_tp"] for r in rows); dfp = sum(r["div_fp"] for r in rows); dfn = sum(r["div_fn"] for r in rows)
    # per-sample adjusted J weighted by (TP+FP+FN), as described on the competition page
    wts = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows]
    adj = sum(r["adj_edge_J"] * wt for r, wt in zip(rows, wts)) / max(1, sum(wts))
    dJ = dtp / max(1, dtp + dfp + dfn)
    return dict(edge_J=round(tp / max(1, tp + fp + fn), 4), adj_edge_J=round(adj, 4), div_J=round(dJ, 4),
                score=round(adj + w * dJ, 4), edge_tp=tp, edge_fp=fp, edge_fn=fn)


def run_cv(train_zarrs, model, configs, log=print):
    """Evaluate several CFG variants on the same train clips with the official-style metric.
    configs: dict name -> dict of CFG overrides. Returns list of rows (one per config x clip) and summary."""
    results, summary = [], []
    gts = {zp: read_geff(zp.parent / (zp.name[:-5] + ".geff")) for zp in train_zarrs}
    for name, over in configs.items():
        cfg = dict(CFG); cfg.update(over)
        rows = []
        for zp in train_zarrs:
            r_, info = process_sample(zp, model, cfg, log=lambda *a: None)
            nodes = [(r[3], r[4], r[5], r[6], r[2]) for r in r_ if r[1] == "node"]
            edges = [(r[7], r[8]) for r in r_ if r[1] == "edge"]
            sc = score_official(nodes, edges, gts[zp])
            sc.update(config=name, sample=info["name"])
            rows.append(sc)
            log(f"    [{name}] {info['name']}: score={sc['score']:.3f} adjJ={sc['adj_edge_J']:.3f} "
                f"(tp={sc['edge_tp']} fp={sc['edge_fp']} fn={sc['edge_fn']}) node_recall={sc['node_recall']:.3f} "
                f"divJ={sc['div_J']:.2f} (gt_div={sc['gt_div']}) n_pred/est={sc['n_pred']}/{sc['est']}")
        ma = micro_average(rows); ma["config"] = name
        summary.append(ma)
        log(f"  ==> [{name}] micro score={ma['score']:.4f} adjJ={ma['adj_edge_J']:.4f} divJ={ma['div_J']:.3f}")
        results += rows
    return results, summary


# ---------------------------------------------------------------- detection calibration on train
def calibrate_thresh_k(train_zarrs, cfg=CFG, ks=(0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0), log=print):
    """Sweep thresh_k on a few train clips; pick the k whose total detections best match the
    estimated_number_of_nodes from the .geff metadata (ratio closest to 1). Also reports GT node recall.
    Returns (best_k, table) or (None, table) when no estimate is available."""
    import torch
    from scipy.optimize import linear_sum_assignment

    rows = []
    frames_cache = []
    for zp in train_zarrs:
        gt = read_geff(zp.parent / (zp.name[:-5] + ".geff"))
        est = gt.get("estimated_number_of_nodes")
        arr = open_zarr_array(zp)
        T = arr.shape[0]
        # evaluate recall on a subset of frames for speed; count on all frames
        frames_cache.append((zp.name[:-5], arr, gt, est))
    for k in ks:
        c = dict(cfg); c["thresh_k"] = k
        for name, arr, gt, est in frames_cache:
            n_det, n_hit, n_gt = 0, 0, 0
            gt_by_t = {}
            for i, t, z, y, x in zip(gt["ids"], gt["t"], gt["z"], gt["y"], gt["x"]):
                gt_by_t.setdefault(int(t), []).append((z, y, x))
            for t in range(arr.shape[0]):
                zyx, sc_ = detect_frame(np.asarray(arr[t]), c)
                zyx = merge_close_detections(zyx, sc_, c.get("merge_min_dist_um"))
                n_det += len(zyx)
                gl = gt_by_t.get(t)
                if gl and len(zyx):
                    P_ = zyx.astype(np.float32) * SCALE_ZYX
                    G_ = np.array(gl, np.float32) * SCALE_ZYX
                    D = np.linalg.norm(P_[:, None, :] - G_[None, :, :], axis=2)
                    r, cidx = linear_sum_assignment(D)
                    n_hit += int((D[r, cidx] <= 7.0).sum())
                    n_gt += len(gl)
                elif gl:
                    n_gt += len(gl)
            rows.append(dict(sample=name, thresh_k=k, n_det=n_det, est=est,
                             ratio=(n_det / est) if est else None, node_recall=n_hit / max(1, n_gt)))
            log(f"    k={k:<4} {name}: det={n_det:6d} est={est} ratio={rows[-1]['ratio'] and round(rows[-1]['ratio'],3)} recall={rows[-1]['node_recall']:.3f}")
    with_est = [r for r in rows if r["ratio"] is not None]
    if not with_est:
        log("    no estimated_number_of_nodes in geff metadata -> keeping default thresh_k")
        return None, rows
    best, best_err = None, 1e9
    for k in ks:
        rs = [r for r in with_est if r["thresh_k"] == k]
        err = float(np.median([abs(np.log(r["ratio"])) for r in rs]))   # log-ratio: over/under symmetric
        if err < best_err:
            best, best_err = k, err
    log(f"    -> best thresh_k = {best} (median |log ratio| = {best_err:.3f})")
    return best, rows


# ---------------------------------------------------------------- submission
def rows_for_sample(dataset: str, nodes, edges):
    rows = []
    for t, z, y, x, nid in nodes:
        rows.append((dataset, "node", nid, t, z, y, x, -1, -1))
    for s, tgt in edges:
        rows.append((dataset, "edge", -1, -1, -1, -1, -1, s, tgt))
    return rows


def write_submission(all_rows, out_path: Path):
    import pandas as pd

    df = pd.DataFrame(
        all_rows, columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    )
    df.insert(0, "id", np.arange(len(df)))
    for c in ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]:
        df[c] = df[c].astype(np.int64)
    df.to_csv(out_path, index=False)
    return df


def choose_radius(arr, cfg=CFG, log=print):
    """Probe a few frames at probe_r, take the median detection count, map to a radius."""
    ar = cfg.get("adaptive_radius")
    if not ar:
        return cfg["cell_radius_um"]
    probe = dict(cfg); probe["cell_radius_um"] = ar.get("probe_r", 4.0); probe["adaptive_radius"] = None
    T = arr.shape[0]
    idx = np.linspace(0, T - 1, ar.get("n_probe", 5)).round().astype(int)
    counts = []
    for t in idx:
        zyx, sc = detect_frame(np.asarray(arr[t]), probe)
        zyx = merge_close_detections(zyx, sc, probe.get("merge_min_dist_um"))
        counts.append(len(zyx))
    n = max(1.0, float(np.median(counts)))
    r = float(np.clip(ar.get("a", 26.5) * n ** (-1.0 / 3.0), ar.get("rmin", 3.0), ar.get("rmax", 6.0)))
    r = round(r * 4) / 4  # quarter-um steps
    log(f"    adaptive radius: median {n:.0f} cells/frame at probe r={probe['cell_radius_um']} -> r={r} um")
    return r


def process_sample(zarr_path: Path, model, cfg=CFG, log=print):
    name = zarr_path.name.replace(".zarr", "")
    log(f"[{name}]")
    arr = open_zarr_array(zarr_path)
    log(f"    shape={arr.shape} dtype={arr.dtype}")
    cfg = dict(cfg)
    if cfg.get("adaptive_radius"):
        cfg["cell_radius_um"] = choose_radius(arr, cfg, log=log)
    if cfg.get("mask_scale"):
        r_um = cfg["mask_scale"] * cfg["cell_radius_um"]
        cfg["mask_radius_vox"] = tuple(int(max(1, round(r_um / s_))) for s_ in SCALE_ZYX)
        log(f"    pseudo-mask radius (z,y,x) vox = {cfg['mask_radius_vox']}  (~{r_um:.1f} um)")
    dets, imgs = detect_volume(arr, cfg, log=log)
    n_det = sum(len(d) for d in dets)
    log(f"    total detections: {n_det}")
    nodes, edges = None, None
    if model is not None and n_det > 0:
        try:
            masks = make_masks(dets, arr.shape[1:], cfg["mask_radius_vox"])
            nodes, edges = link_trackastra(model, imgs, masks, dets, cfg, log=log)
            del masks
        except Exception as e:  # noqa
            log(f"    !! trackastra failed ({type(e).__name__}: {e}); falling back to NN linker")
            nodes = None
    if nodes is None:
        nodes, edges = link_nearest(dets, cfg, log=log)
    del imgs
    gc.collect()
    return rows_for_sample(name, nodes, edges), dict(name=name, n_nodes=len(nodes), n_edges=len(edges))
