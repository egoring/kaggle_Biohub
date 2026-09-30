"""Notebook cells shared by the inference and fine-tuning notebooks."""

LOCATE_CELL = '''# ------------------------------------------------------------------ 1. locate inputs (fast: no recursive glob over the 87 GB dataset)
COMP_SLUG = "biohub-cell-tracking-during-development"

def _find_comp_dir():
    for c in [KAGGLE_INPUT / "competitions" / COMP_SLUG, KAGGLE_INPUT / COMP_SLUG]:
        if (c / "test").is_dir():
            return c
    for c in KAGGLE_INPUT.glob("*/"):              # one level deep only
        if (c / "test").is_dir() and (c / "sample_submission.csv").exists():
            return c
    for c in KAGGLE_INPUT.glob("*/*/"):
        if (c / "test").is_dir() and (c / "sample_submission.csv").exists():
            return c
    return None

def _walk_pruned(root, max_depth=8):
    """os.walk that never descends into the competition data or zarr/geff stores."""
    root = str(root)
    for dp, dns, fns in os.walk(root):
        depth = dp[len(root):].count(os.sep)
        dns[:] = [d for d in dns if not d.endswith((".zarr", ".geff")) and d != COMP_SLUG and d != "competitions"]
        if depth >= max_depth:
            dns[:] = []
        yield Path(dp), fns

COMP_DIR = _find_comp_dir()
MODEL_DIRS, PKG_DIR, CACHE_DIRS, REPO_DIRS, CKPT_DIRS = [], None, [], [], []
for dp, fns in _walk_pruned(KAGGLE_INPUT):
    if "model.pt" in fns and "train_config.yaml" in fns:
        MODEL_DIRS.append(dp)
    if PKG_DIR is None and any(f.startswith("trackastra-") and f.endswith(".whl") for f in fns):
        PKG_DIR = dp
    if dp.name == "feat_cache" and any(f.endswith(".npz") for f in fns):
        CACHE_DIRS.append(dp)
    if dp.name == "tracking_cellmot" and "io.py" in fns:            # official baseline repo (prep-bundle v2: repo/src/tracking_cellmot)
        REPO_DIRS.append(dp.parent.parent)
    if "edge_predictor_best.pth" in fns or "last.pth" in fns:        # baseline checkpoints (ours or the repo's layout)
        CKPT_DIRS.append(dp)
# model folder preference: MODEL_PREFER (e.g. "ctc_ft" = fine-tuned) first, then the pretrained "ctc"
_pref = globals().get("MODEL_PREFER", "ctc_ft")
MODEL_DIRS.sort(key=lambda p: (p.name != _pref, p.name != "ctc", str(p)))
MODEL_DIR = MODEL_DIRS[0] if MODEL_DIRS else None
PRETRAINED_DIR = next((p for p in MODEL_DIRS if p.name == "ctc"), MODEL_DIR)

print("COMP_DIR :", COMP_DIR)
print("MODEL_DIR:", MODEL_DIR, "(all found:", [str(p) for p in MODEL_DIRS], ")")
print("PKG_DIR  :", PKG_DIR)
print("CACHE_DIRS:", [str(p) for p in CACHE_DIRS])
REPO_DIR = REPO_DIRS[0] if REPO_DIRS else None
print("REPO_DIR :", REPO_DIR, "| CKPT_DIRS:", [str(p) for p in CKPT_DIRS])
print("inputs attached:", [p.name for p in KAGGLE_INPUT.iterdir()] if KAGGLE_INPUT.exists() else "n/a")
assert COMP_DIR is not None, "competition data not attached (Add Input -> Competitions -> " + COMP_SLUG + ")"
if MODEL_DIR is None or PKG_DIR is None:
    print("\\n" + "!" * 78)
    print("!! Trackastra bundle is NOT attached. Add Input -> Your Work -> 'biohub-prep-bundle' notebook output")
    print("!! (it holds the offline wheels for zarr/trackastra and the pretrained Trackastra weights).")
    print("!" * 78 + "\\n")

TEST_ZARRS = sorted(p for p in (COMP_DIR / "test").iterdir() if p.name.endswith(".zarr"))
n_train = sum(1 for p in (COMP_DIR / "train").iterdir() if p.name.endswith(".zarr")) if (COMP_DIR / "train").is_dir() else -1
print(f"{len(TEST_ZARRS)} test samples (hidden set is ~ train size = {n_train}):", [p.name for p in TEST_ZARRS[:4]], "...")
import torch
if not torch.cuda.is_available():
    print("!! No GPU: set Settings -> Accelerator -> GPU (T4/P100).")
'''

PIP_CELL = '''# ------------------------------------------------------------------ 2. offline install (no internet)
NEEDED = {  # import name -> wheel project name
    "trackastra": "trackastra", "geff": "geff", "geff_spec": "geff_spec", "zarr": "zarr",
    "numcodecs": "numcodecs", "donfig": "donfig", "google_crc32c": "google_crc32c",
    "edt": "edt", "lz4": "lz4", "imagecodecs": "imagecodecs", "chardet": "chardet",
    "platformdirs": "platformdirs", "psutil": "psutil", "dask": "dask", "numba": "numba",
    "skimage": "scikit_image", "yaml": "pyyaml", "joblib": "joblib", "tqdm": "tqdm",
    "pandas": "pandas", "scipy": "scipy", "torch": "torch", "torchvision": "torchvision",
    "typing_extensions": "typing_extensions", "pydantic": "pydantic", "requests": "requests",
    "motile": "motile", "ilpy": "ilpy", "pyscipopt": "pyscipopt", "structsvm": "structsvm",  # optional: enables mode="ilp"
    # official-baseline (tracking_cellmot) stack, present only in prep-bundle v2
    "tracksdata": "tracksdata", "polars": "polars", "rustworkx": "rustworkx", "bidict": "bidict", "psygnal": "psygnal",
    "blosc2": "blosc2", "rstar_python": "rstar_python", "sqlalchemy": "sqlalchemy", "pyarrow": "pyarrow", "rich": "rich",
    "ndindex": "ndindex", "msgpack": "msgpack", "numexpr": "numexpr",
}

def _importable(mod):
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False

def _pip(args):
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-warn-script-location", "-q"] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
    return r.returncode == 0

def _installed(mod):
    """present on disk? (importlib.util.find_spec does NOT import the module -> nothing stale gets loaded)"""
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False

TRACKASTRA_OK = False
if PKG_DIR is not None:
    import importlib.util, importlib.metadata as im
    from packaging.version import Version
    # step A0: Kaggle ships OLD versions of some libs that tracksdata needs newer (polars < 1.36 lacks pl.Float16).
    # Upgrade from the bundle BEFORE anything imports them (re-importing a native lib after upgrade can crash the kernel).
    UPGRADE = ["polars", "polars_runtime_32", "rustworkx", "psygnal", "bidict", "blosc2", "rstar_python", "numcodecs"]
    ups = []
    for proj in UPGRADE:
        w = sorted(glob.glob(str(PKG_DIR / f"{proj}-*.whl")))
        if not w:
            continue
        ver = Path(w[-1]).name.split("-")[1]
        try:
            inst = im.version(proj.replace("_", "-"))
        except Exception:
            inst = None
        if inst is None or Version(inst) < Version(ver):
            ups.append(w[-1]); print(f"    upgrade {proj}: {inst} -> {ver}")
    if ups:
        _pip(["--no-deps", "--upgrade"] + ups); importlib.invalidate_caches()
    # step A: only the missing projects (checked without importing), no dependency resolution (keeps Kaggle torch/numpy intact)
    missing = [proj for mod, proj in NEEDED.items() if not _installed(mod)]
    print("missing before install:", missing)
    wheels = []
    for proj in missing:
        w = sorted(glob.glob(str(PKG_DIR / f"{proj}-*.whl")))
        wheels += w[:1] if w else []
    if wheels:
        _pip(["--no-deps"] + wheels); importlib.invalidate_caches()
    still = [proj for mod, proj in NEEDED.items() if not _importable(mod)]      # now really import to validate
    print("missing after step A:", still)
    if still:
        # step B: let pip resolve, but pin already-installed heavy libs so they are not replaced
        cons = OUT_DIR / "constraints.txt"
        pins = []
        for proj in ["torch", "torchvision", "numpy", "scipy", "pandas", "scikit-image", "numba", "llvmlite"]:
            try: pins.append(f"{proj}=={im.version(proj)}")
            except Exception: pass
        cons.write_text("\\n".join(pins))
        extra = ["tracksdata"] if glob.glob(str(PKG_DIR / "tracksdata-*.whl")) else []
        _pip(["--find-links", str(PKG_DIR), "-c", str(cons), "trackastra", "zarr", "geff"] + extra)
        importlib.invalidate_caches()
    TRACKASTRA_OK = _importable("trackastra") and _importable("zarr")
else:
    TRACKASTRA_OK = _importable("trackastra") and _importable("zarr")

import numpy as np, torch
print("python", sys.version.split()[0], "| numpy", np.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(), "| trackastra importable:", TRACKASTRA_OK)
assert _importable("zarr"), ("zarr is not installed and no wheel bundle found -> attach the prep-bundle notebook output "
    "(it ships zarr/numcodecs/trackastra wheels for Python 3.12)")
'''
