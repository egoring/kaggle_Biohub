"""Create a fake competition folder: test/<id>.zarr (zarr v3, array '0', (T,Z,Y,X) uint16,
chunks (1,Z,Y,X), blosc/zstd) with moving + dividing gaussian blobs, plus GT graph for checking."""
import json
import sys
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec

SCALE = np.array([1.625, 0.40625, 0.40625])


def make_sample(out_dir: Path, name: str, T=20, Z=64, Y=256, X=256, n0=40, seed=0):
    rng = np.random.default_rng(seed)
    # cells: physical positions (um)
    ext = np.array([Z, Y, X]) * SCALE
    pos = rng.uniform(0.15, 0.85, size=(n0, 3)) * ext
    vel = rng.normal(0, 0.8, size=(n0, 3))
    vel[:, 0] *= 0.4
    ids = np.arange(1, n0 + 1)
    next_id = n0 + 1
    nodes, edges = [], []
    prev_ids = None
    vol = np.zeros((T, Z, Y, X), np.uint16)
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    r_um = 3.2
    for t in range(T):
        frame = rng.normal(180, 25, size=(Z, Y, X)).astype(np.float32)
        node_ids_t = []
        for p, i in zip(pos, ids):
            vz, vy, vx = np.round(p / SCALE).astype(int)
            nodes.append((t, vz, vy, vx, int(i)))
            node_ids_t.append(int(i))
            d2 = ((zz - p[0] / SCALE[0]) * SCALE[0]) ** 2 + ((yy - p[1] / SCALE[1]) * SCALE[1]) ** 2 + ((xx - p[2] / SCALE[2]) * SCALE[2]) ** 2
            frame += 1500 * np.exp(-d2 / (2 * (r_um / 1.6) ** 2))
        vol[t] = np.clip(frame, 0, 65535).astype(np.uint16)
        # advance
        new_pos, new_vel, new_ids = [], [], []
        for p, v, i in zip(pos, vel, ids):
            if t < T - 1 and rng.random() < 0.03 and len(ids) < 90:  # division
                for sgn in (-1, 1):
                    nid = next_id
                    next_id += 1
                    off = rng.normal(0, 1, 3)
                    off /= np.linalg.norm(off)
                    new_pos.append(p + v + sgn * off * 3.0)
                    new_vel.append(v + rng.normal(0, 0.3, 3))
                    new_ids.append(nid)
                    edges.append((int(i), int(nid)))
            else:
                nid = next_id
                next_id += 1
                new_pos.append(p + v + rng.normal(0, 0.3, 3))
                new_vel.append(v * 0.9 + rng.normal(0, 0.2, 3))
                new_ids.append(nid)
                edges.append((int(i), int(nid)))
        pos = np.clip(np.array(new_pos), 2.0, ext - 2.0)
        vel = np.array(new_vel)
        ids = np.array(new_ids)
    # last frame ids appended edges to non-existent nodes -> drop
    valid = {n[4] for n in nodes}
    edges = [e for e in edges if e[0] in valid and e[1] in valid]

    zdir = out_dir / "test" / f"{name}.zarr"
    zdir.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(zdir), mode="w")
    a = g.create_array("0", shape=vol.shape, chunks=(1, Z, Y, X), dtype="uint16",
                       compressors=[BloscCodec(cname="zstd", clevel=3)])
    a[:] = vol
    (out_dir / "gt").mkdir(exist_ok=True)
    with open(out_dir / "gt" / f"{name}.json", "w") as f:
        json.dump({"nodes": [list(map(int, n)) for n in nodes], "edges": edges}, f)
    print(name, vol.shape, "nodes", len(nodes), "edges", len(edges), "divisions",
          sum(1 for s in set(e[0] for e in edges) if sum(1 for e in edges if e[0] == s) > 1))


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    make_sample(out, "aaaa_0000synth", T=20, seed=1)
    make_sample(out, "bbbb_0001synth", T=12, n0=25, seed=2)
    with open(out / "sample_submission.csv", "w") as f:
        f.write("id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n0,aaaa_0000synth,node,1,0,32,128,128,-1,-1\n")
