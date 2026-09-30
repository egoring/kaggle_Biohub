"""post_task through a spawn ProcessPoolExecutor (as in the v17 notebook) must give the same result as in-process postprocess."""
import sys, time, os; sys.path.insert(0, "."); from pathlib import Path; import numpy as np


def build_graph():
    coords = []; edges = []; idx = lambda: len(coords) - 1
    A = []; B = []; C = []
    for t in range(12): coords.append((t, 10, 50, 50 + t)); A.append(idx())
    for t in range(4):  coords.append((t, 30, 100, 100 + t)); B.append(idx())
    for t in range(2):  coords.append((t, 50, 150, 150 + t)); C.append(idx())
    for t in range(5):  coords.append((t * 2, 60, 200, 200 + 5 * t))
    for a, b in zip(A, A[1:]): edges.append((a, b, 0.95, 1.0))
    for a, b in zip(B, B[1:]): edges.append((a, b, 0.9, 1.0))
    for a, b in zip(C, C[1:]): edges.append((a, b, 0.6, 1.0))
    edges.append((A[0], B[1], 0.3, 80.0))
    return np.array(coords, np.int16), np.array(edges, np.float32)


def main():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import cellmot_infer as CI
    repo = Path("fake_input/prep-v2/repo").resolve()
    CI.setup_predict(repo)
    coords, edges = build_graph()
    over = dict(det_threshold=0.99, use_ilp=True, min_track_len=5)
    ref = CI.postprocess(coords, edges, CI.make_cfg(det_tta=False, **over), log=lambda *a: None)
    ex = ProcessPoolExecutor(2, mp_context=mp.get_context("spawn"), initializer=CI.init_worker, initargs=(os.getcwd(), str(repo)))
    t = time.time()
    futs = [ex.submit(CI.post_task, (coords, edges, over, False)) for _ in range(3)]
    res = [f.result(timeout=600) for f in futs]
    ex.shutdown()
    print(f"3 tasks on 2 workers in {time.time()-t:.0f}s; worker result nodes/edges:", len(res[0][0]), len(res[0][1]), "| t_post", [round(r[2], 2) for r in res])
    assert all(r[0] == ref[0] and r[1] == ref[1] for r in res), "worker result != in-process result"
    assert len(res[0][0]) == 12 and len(res[0][1]) == 11
    print("OK")


if __name__ == "__main__":
    main()
