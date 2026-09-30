"""Inference helpers around the repo's predict_unet_transformer.py: coords/edges -> our node/edge lists,
optional tracksdata ILP, short-track pruning, scoring with our official-style scorer, submission rows.

Two stages so post-processing can be tuned without re-running the network:
  predict_raw(...)  -> (coords, edges)       expensive: UNet + transformer (cacheable as .npz)
  postprocess(...)  -> (nodes, out_edges, w) cheap: ILP / greedy, edge threshold, short-track pruning
predict_clip(...) = both.
"""
from __future__ import annotations
import sys, time, contextlib, io
from pathlib import Path
import numpy as np

_P = None


def setup_predict(repo_dir: Path):
    global _P
    import cellmot_train as CT
    CT.setup_repo(repo_dir)
    import predict_unet_transformer as P
    import tracking_cellmot.io as cio
    P.open_dataset = cio.open_dataset          # quantile fallback patch
    _P = P
    return P


def make_cfg(det_threshold=0.95, use_ilp=True, ilp_edge_weight=-1.0, ilp_appearance_weight=0.0,
             ilp_disappearance_weight=1.4, ilp_division_weight=1.0, edge_threshold=0.25, pool_kernel_um=3.0, det_tta=True,
             min_track_len=1):
    cfg = _P.PredictConfig(det_threshold=det_threshold, det_tta=det_tta, pool_kernel_um=pool_kernel_um,
                           threshold=edge_threshold, use_ilp=use_ilp, ilp_edge_weight=ilp_edge_weight,
                           ilp_appearance_weight=ilp_appearance_weight, ilp_disappearance_weight=ilp_disappearance_weight,
                           ilp_division_weight=ilp_division_weight)
    cfg.min_track_len = int(min_track_len)     # ours: drop connected components with fewer nodes (after linking)
    return cfg


def load_model(weights_path: Path, device):
    return _P.load_model(Path(weights_path), device)


# ------------------------------------------------------------------ stage 1: network
def predict_raw(model, zarr_path: Path, device, cfg, window_size, downsample):
    """coords (N,4) int16 [t,z,y,x] in original resolution; edges (M,4) float32 [src_idx, tgt_idx, prob, dist]."""
    coords, edges = _P.predict_video(model, Path(zarr_path), device, cfg=cfg, window_size=window_size, downsample=downsample)
    ed = np.asarray(edges, dtype=np.float32).reshape(-1, 4)
    return np.asarray(coords, dtype=np.int16), ed


def save_raw(path: Path, coords, edges):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coords=coords, edges=edges)


def load_raw(path: Path):
    z = np.load(path)
    return z["coords"], z["edges"]


# ------------------------------------------------------------------ stage 2: linking + pruning
def _components(n_nodes, edges):
    """union-find over (src, tgt) index pairs -> component id per node."""
    parent = np.arange(n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for s, t in edges:
        rs, rt = find(int(s)), find(int(t))
        if rs != rt:
            parent[rs] = rt
    return np.array([find(i) for i in range(n_nodes)])


def prune_short_tracks(nodes, edges, weights, min_len):
    """nodes [(t,z,y,x,id)], edges [(src_id,tgt_id)] -> drop connected components with < min_len nodes; ids renumbered from 1."""
    if min_len <= 1 or not nodes:
        return nodes, edges, weights
    id2idx = {n[4]: k for k, n in enumerate(nodes)}
    comp = _components(len(nodes), [(id2idx[s], id2idx[t]) for s, t in edges])
    sizes = np.bincount(comp, minlength=len(nodes))
    keep = sizes[comp] >= min_len
    new_id = {}
    out_nodes = []
    for k, n in enumerate(nodes):
        if keep[k]:
            new_id[n[4]] = len(out_nodes) + 1
            out_nodes.append((n[0], n[1], n[2], n[3], new_id[n[4]]))
    out_edges, out_w = [], []
    for (s, t), w in zip(edges, weights):
        if s in new_id and t in new_id:
            out_edges.append((new_id[s], new_id[t])); out_w.append(w)
    return out_nodes, out_edges, out_w


def postprocess(coords, edges, cfg, log=print):
    """(coords, edges) from predict_raw -> (nodes [(t,z,y,x,id)], edges [(src_id,tgt_id)], weights); ids start at 1."""
    t0 = time.time()
    n = len(coords)
    ed = np.asarray(edges, dtype=np.float32).reshape(-1, 4)
    if len(ed) and cfg.threshold > 0:
        ed = ed[ed[:, 2] > cfg.threshold]          # cached edges were filtered at the threshold used at predict time (<= this one)
    edge_list = [(int(s), int(t), float(p), float(d)) for s, t, p, d in ed.tolist()]
    if cfg.use_ilp and edge_list:
        import tracksdata as td
        graph = _P.build_graph(coords, edge_list)
        solver = td.solvers.ILPSolver(edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                                      appearance_weight=cfg.ilp_appearance_weight,
                                      disappearance_weight=cfg.ilp_disappearance_weight,
                                      division_weight=cfg.ilp_division_weight)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            solver.solve(graph)                       # writes a boolean "solution" attr on nodes and edges
        na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x", "solution"])
        ea = graph.edge_attrs(attr_keys=["source_id", "target_id", "solution"])
        na = na.filter(na["solution"]); ea = ea.filter(ea["solution"])
        if len(na) == 0:                              # degenerate solution -> keep the raw detections
            na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        id2new = {int(i): k + 1 for k, i in enumerate(na["node_id"].to_list())}
        nodes = [(int(t), int(round(z)), int(round(y)), int(round(x)), id2new[int(i)])
                 for i, t, z, y, x in zip(na["node_id"].to_list(), na["t"].to_list(), na["z"].to_list(), na["y"].to_list(), na["x"].to_list())]
        out_edges = [(id2new[int(s)], id2new[int(t)]) for s, t in zip(ea["source_id"].to_list(), ea["target_id"].to_list())
                     if int(s) in id2new and int(t) in id2new]
        weights = [1.0] * len(out_edges)
    else:
        nodes = [(int(t), int(z), int(y), int(x), k + 1) for k, (t, z, y, x) in enumerate(np.asarray(coords).tolist())]
        out_edges = [(s + 1, t + 1) for s, t, p, d in edge_list]
        weights = [p for s, t, p, d in edge_list]
    n_before = len(nodes)
    nodes, out_edges, weights = prune_short_tracks(nodes, out_edges, weights, getattr(cfg, "min_track_len", 1))
    log(f"    cellmot post: {n} detections, {len(edge_list)} candidate edges -> {n_before} linked nodes -> "
        f"{len(nodes)} nodes / {len(out_edges)} edges after pruning ({time.time()-t0:.0f}s)")
    return nodes, out_edges, weights


def predict_clip(model, zarr_path: Path, device, cfg, window_size, downsample, log=print):
    """Returns (nodes [(t,z,y,x,id)], edges [(src_id,tgt_id)], weights) in OUR format (ids start at 1)."""
    t0 = time.time()
    coords, edges = predict_raw(model, zarr_path, device, cfg, window_size, downsample)
    log(f"    cellmot net: {len(coords)} detections, {len(edges)} candidate edges ({time.time()-t0:.0f}s)")
    return postprocess(coords, edges, cfg, log=log)


# ------------------------------------------------------------------ parallel post-processing (ILP is single-threaded CPU work;
# run it in worker processes while the GPU keeps producing the next clip's detections)
def init_worker(code_dir: str, repo_dir: str):
    import os, sys
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # workers never touch the GPU
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    setup_predict(Path(repo_dir))


def post_task(args):
    """(coords, edges, cfg_overrides dict, det_tta) -> (nodes, edges, seconds)"""
    coords, edges, over, det_tta = args
    t0 = time.time()
    cfg = make_cfg(det_tta=det_tta, **over)
    nodes, out_edges, w = postprocess(coords, edges, cfg, log=lambda *a: None)
    return nodes, out_edges, time.time() - t0
