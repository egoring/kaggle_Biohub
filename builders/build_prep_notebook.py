from pathlib import Path
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Biohub prep notebook v2 — offline bundle (wheels + Trackastra ctc + official baseline repo)

Run this **with Internet ON** (Settings → Internet → On). No accelerator needed.
It produces, in this notebook's Output:

- `packages/` — wheels for `trackastra`, `zarr`, `geff`, `motile`, `ilpy`, … built for this exact Kaggle Python
- `models/ctc/` — Trackastra pretrained *ctc* model (2D+3D), downloaded from the official release

Then in the **submission notebook** (Internet OFF): `+ Add Input → Your Work / Notebooks → this notebook`
and it will find `model.pt` and the wheels automatically.
""")

code('''import os, sys, glob, shutil, subprocess, importlib, time, json
from pathlib import Path
OUT = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
PKG = OUT / "packages"; MODELS = OUT / "models"
PKG.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
print("python", sys.version.split()[0], "| output:", OUT)

import urllib.request
try:
    urllib.request.urlopen("https://pypi.org/simple/zarr/", timeout=15).read(100)
    print("internet: OK")
except Exception as e:
    raise SystemExit("!! No internet. Settings -> Internet -> On, then re-run.  (" + str(e) + ")")
''')

code('''# 1. download wheels (full dependency resolution for THIS python), then drop what Kaggle already ships
PKGS = ["trackastra", "zarr", "geff", "motile", "ilpy", "pyscipopt", "tracksdata", "polars", "rustworkx", "blosc2", "pyarrow"]
t0 = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "download", "-q", "-d", str(PKG)] + PKGS, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
assert r.returncode == 0, "pip download failed"

HARD_DROP = ("torch-", "torchvision-", "triton-", "nvidia_", "cuda_", "numpy-", "scipy-", "pandas-", "scikit_image-",
             "numba-", "llvmlite-", "sympy-", "pillow-", "setuptools-")
ALWAYS_KEEP = ("trackastra", "zarr", "geff", "geff_spec", "numcodecs", "motile", "ilpy", "pyscipopt", "structsvm", "edt",
               "imagecodecs", "donfig", "google_crc32c", "lz4", "chardet",
               "tracksdata", "polars", "polars_runtime_32", "rustworkx", "bidict", "psygnal", "blosc2", "rstar_python",
               "sqlalchemy", "pyarrow", "rich", "ndindex", "msgpack", "numexpr")
from packaging.version import Version
removed = []
for w in sorted(PKG.glob("*.whl")) + sorted(PKG.glob("*.tar.gz")):
    name, ver = w.name.split("-")[0], w.name.split("-")[1]
    if w.name.startswith(HARD_DROP):
        w.unlink(); removed.append(w.name); continue
    if name in ALWAYS_KEEP:
        continue
    try:                      # drop only if Kaggle already has this distribution at >= the downloaded version
        inst = importlib.metadata.version(name.replace("_", "-"))
        if Version(inst) >= Version(ver):
            w.unlink(); removed.append(w.name)
    except Exception:
        pass                  # not installed -> keep
kept = sorted(p.name for p in PKG.iterdir())
print(f"kept {len(kept)} wheels, removed {len(removed)}, {sum(p.stat().st_size for p in PKG.iterdir())/1e6:.0f} MB, {time.time()-t0:.0f}s")
print(kept)
''')

code('''# 2. install from the bundle exactly like the submission notebook will (offline, --no-deps) and verify imports
import importlib.metadata
def _pip(args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"] + args, capture_output=True, text=True)
    if r.returncode != 0: print(r.stdout[-1500:], r.stderr[-2500:])
    return r.returncode == 0
whls = sorted(str(p) for p in PKG.glob("*.whl"))
if whls: _pip(whls)
sd = sorted(PKG.glob("*.tar.gz"))
if sd: print("source-only packages (no wheel):", [p.name for p in sd]); _pip([str(p) for p in sd])
else: print("nothing to install (everything already present)")
importlib.invalidate_caches()
for m in ["zarr", "numcodecs", "geff", "trackastra", "motile", "ilpy", "edt", "imagecodecs", "tracksdata", "polars", "rustworkx", "blosc2"]:
    try:
        importlib.import_module(m); print("  import", m, "OK")
    except Exception as e:
        print("  import", m, "FAILED:", type(e).__name__, str(e)[:200])
''')

code('''# 3. download the pretrained Trackastra "ctc" model (2D+3D) and place it at models/ctc
from trackastra.model import Trackastra
model = Trackastra.from_pretrained("ctc", device="cpu", download_dir=OUT / "_dl")
src = OUT / "_dl" / "ctc"
dst = MODELS / "ctc"
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst); shutil.rmtree(OUT / "_dl", ignore_errors=True)
print("model files:", sorted(p.name for p in dst.iterdir()))
print("train features:", model.train_args.get("features"), "| window:", model.train_args.get("window"), "| ndim:", model.train_args.get("ndim"))
''')

code('''# 4. smoke test: read one competition clip with zarr (proves the bundle can read the data)
import zarr, numpy as np
COMP = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
if not COMP.exists():
    COMP = next((p for p in Path("/kaggle/input").glob("*biohub*") if (p / "test").is_dir()), None)
if COMP is not None:
    z = sorted((COMP / "test").glob("*.zarr"))[0]
    a = zarr.open_group(str(z), mode="r")["0"]
    f = np.asarray(a[0])
    print(z.name, a.shape, a.dtype, "frame0 min/max/mean:", f.min(), f.max(), round(float(f.mean()), 1))
else:
    print("competition data not attached — fine for this prep notebook, but attach it if you want the smoke test")
''')

code('''# 4b. official baseline repo (royerlab/kaggle-cell-tracking-competition) -> repo/  (training + prediction code)
REPO = OUT / "repo"
if REPO.exists(): shutil.rmtree(REPO)
r = subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/royerlab/kaggle-cell-tracking-competition.git", str(REPO)],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-1000:]
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
shutil.rmtree(REPO / ".git", ignore_errors=True)
(REPO / "COMMIT.txt").write_text(commit)
print("repo cloned at commit", commit, "|", sorted(p.name for p in REPO.iterdir()))
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import tracking_cellmot.io, tracking_cellmot.models, train_unet_transformer, predict_unet_transformer
print("import tracking_cellmot + scripts OK")
''')

code('''# 5. summary
tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
print(f"Output ready: {tot:.0f} MB")
print(" packages/:", len(list(PKG.glob("*.whl"))), "wheels")
print(" models/ctc/:", sorted(p.name for p in (MODELS / "ctc").iterdir()))
print(" repo/:", (REPO / "COMMIT.txt").read_text())
print("\\nNext: Save Version -> Save & Run All. Then in the submission notebook: + Add Input -> Your Work -> this notebook.")
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
import sys as _s
Path(_s.argv[1] if len(_s.argv) > 1 else "biohub_prep_bundle.ipynb").write_text(nbf.writes(nb))
print("written", len(cells), "cells")
