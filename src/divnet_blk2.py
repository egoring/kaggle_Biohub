def divnet_score_division(
    dataset: str,
    t: int,
    parent_node: dict[str, object],
    daughter1: dict[str, object],
    daughter2: dict[str, object],
    divnet_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    z_pad: int = 8,
    xy_pad: int = 16,
) -> float | None:
    if divnet_bundle is None or dataset is None:
        return None
    model = divnet_bundle["model"]
    device = divnet_bundle["device"]
    t_mod = divnet_bundle["torch"]
    cz = int(round(float(parent_node["z"])))
    cy = int(round(float(parent_node["y"])))
    cx = int(round(float(parent_node["x"])))
    try:
        _T = int(json.loads((TEST_DIR / f"{dataset}.zarr" / "0" / "zarr.json").read_text())["shape"][0])
        frames = [read_test_frame(dataset, min(max(0, t + dt), _T - 1), frame_cache) for dt in range(4)]   # OURS: clamp like training
        vol = np.stack(frames, axis=0) # (4, Z, Y, X)
        Z, Y, X = vol.shape[1], vol.shape[2], vol.shape[3]
        z0, z1 = max(0, cz - z_pad), min(Z, cz + z_pad)
        y0, y1 = max(0, cy - xy_pad), min(Y, cy + xy_pad)
        x0, x1 = max(0, cx - xy_pad), min(X, cx + xy_pad)
        crop = vol[:, z0:z1, y0:y1, x0:x1]
        padded = np.zeros((4, 2 * z_pad, 2 * xy_pad, 2 * xy_pad), dtype=np.float32)
        dz0 = z_pad - (cz - z0)
        dy0 = xy_pad - (cy - y0)
        dx0 = xy_pad - (cx - x0)
        padded[:, dz0:dz0 + (z1 - z0), dy0:dy0 + (y1 - y0), dx0:dx0 + (x1 - x0)] = crop
        mean_val = float(np.mean(padded))
        std_val = float(np.std(padded)) + 1e-6
        padded = (padded - mean_val) / std_val
        # OURS: (4,Z,Y,X) -> (1,4,Z,Y,X) when the model takes the frames as channels; the original double-unsqueeze made a 6-D
        # tensor that conv3d rejects (silently caught -> None -> the public DivNet never scored anything)
        tensor = t_mod.from_numpy(padded).unsqueeze(0).to(device=device, dtype=t_mod.float32)
        if int(divnet_bundle.get('in_c', 1)) == 1:
            tensor = tensor.unsqueeze(0)
        with t_mod.inference_mode():
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
        return prob
    except Exception as e:
        globals()['_DIVNET_ERR'] = repr(e)
        return None