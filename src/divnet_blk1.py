# =============================================================================
# DivNet 3D-CNN Mitosis Verification Module (Fast & Memory-Safe)
# =============================================================================
if torch is not None:
    class _DivNetConvBlock(torch.nn.Module):
        def __init__(self, in_c: int, out_c: int):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(out_c),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(out_c),
                torch.nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.conv(x)

    class DivNetMitosisClassifier(torch.nn.Module):
        def __init__(self, in_c=1):
            super().__init__()
            self.b1 = _DivNetConvBlock(in_c, 16)
            self.p1 = torch.nn.MaxPool3d((1, 2, 2))
            self.b2 = _DivNetConvBlock(16, 32)
            self.p2 = torch.nn.MaxPool3d((2, 2, 2))
            self.b3 = _DivNetConvBlock(32, 64)
            self.gap = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(32, 1),
            )
        def forward(self, x):
            x = self.p1(self.b1(x))
            x = self.p2(self.b2(x))
            x = self.gap(self.b3(x))
            x = torch.flatten(x, 1)
            return self.fc(x)

def load_divnet_mitosis_model() -> dict[str, object] | None:
    if not globals().get("DIVNET_VERIFY", True):
        print("DivNet mitosis gate disabled by configuration.")
        return None
    if torch is None:
        print("DivNet mitosis gate skipped: torch is unavailable.")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Direct I/O check first (avoid expensive FUSE glob)
    _ours = sorted(Path("/kaggle/input/notebooks").glob("*/*/divnet_ours/best_overall.pt")) if Path("/kaggle/input/notebooks").exists() else []
    _ours += sorted(Path("/kaggle/input/datasets").glob("*/*/divnet_ours/best_overall.pt")) if Path("/kaggle/input/datasets").exists() else []
    _ours += sorted(Path("/kaggle/input").glob("*/divnet_ours/best_overall.pt"))
    if os.environ.get("BIOHUB_DIVNET_REQUIRE_OURS", "0") != "0" and not _ours:
        raise RuntimeError("BIOHUB_DIVNET_REQUIRE_OURS=1 but no divnet_ours/best_overall.pt found: attach the v28 notebook Output")
    print("DivNet checkpoints (ours first):", [str(p) for p in _ours])
    _hint = os.environ.get("BIOHUB_DIVNET_CKPT_HINT", "")
    if _hint:
        _hinted = [p for p in _ours if _hint in str(p)]
        if not _hinted:
            raise RuntimeError(f"BIOHUB_DIVNET_CKPT_HINT={_hint!r} matches none of {[str(p) for p in _ours]}")
        _ours = _hinted; print("  hint", repr(_hint), "->", str(_ours[0]))
    candidates = [*_ours,
        Path("/kaggle/input/biohub-divnet-v2/best_overall.pt"),
        Path("/kaggle/input/datasets/giorgosi/biohub-divnet-v2/best_overall.pt"),
        Path("/kaggle/input/biohub-divnet-v2/model.pt"),
        Path("/kaggle/input/biohub-divnet-v2/best.pt"),
    ]
    for ckpt_path in candidates:
        if ckpt_path.exists():
            try:
                print(f"Loading DivNet checkpoint from direct path: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
                _in_c = int(state['b1.conv.0.weight'].shape[1]) if 'b1.conv.0.weight' in state else 1
                model = DivNetMitosisClassifier(_in_c)
                _missing, _unexpected = model.load_state_dict(state, strict=False)
                print(f'  DivNet in_channels={_in_c} missing={len(_missing)} unexpected={len(_unexpected)}')
                model.to(device)
                model.eval()
                print(f"[OK] DivNet loaded successfully from {ckpt_path}")
                if isinstance(ckpt, dict) and "meta" in ckpt: print("  checkpoint meta:", {k: v for k, v in ckpt["meta"].items() if k != "val_clips" and k != "label_stats"})
                _models = [model]
                for _k, _st in enumerate(ckpt.get("ensemble_states", []) if isinstance(ckpt, dict) else []):
                    _m = DivNetMitosisClassifier(_in_c); _mi, _mu = _m.load_state_dict(_st, strict=False); _m.to(device); _m.eval(); _models.append(_m)
                    print(f"  ensemble member {_k}: missing={len(_mi)} unexpected={len(_mu)}")
                if len(_models) > 1:
                    _models = _models[1:]          # ensemble_states already contains the best seed; drop the duplicate
                print(f"  DivNet models in bundle: {len(_models)}")
                return {"model": model, "device": device, "torch": torch, "in_c": _in_c, "models": _models}
            except Exception as exc:
                print(f"Failed loading DivNet checkpoint {ckpt_path}: {exc}")
    print("DivNet mitosis checkpoint not found. Operating in geometric baseline mode.")
    return None

DIVNET_BUNDLE = load_divnet_mitosis_model()
if DIVNET_SAFE_DIV_GATE and DIVNET_BUNDLE is None:
    raise RuntimeError("DIVNET gate requested but no checkpoint loaded: attach the dataset giorgosi/biohub-divnet-v2")
print("DIVNET_SAFE_DIV_GATE:", DIVNET_SAFE_DIV_GATE, "| bundle loaded:", DIVNET_BUNDLE is not None, "| min prob:", DIVNET_MIN_PROB)
if DIVNET_SAFE_DIV_GATE and DIVNET_BUNDLE is not None:
    # smoke test: score a synthetic crop once so a shape bug fails loudly here instead of being swallowed as 'unscored'
    _t = torch.zeros((1, 4 if DIVNET_BUNDLE.get("in_c", 1) == 4 else 1, *((4, 16, 32, 32) if DIVNET_BUNDLE.get("in_c", 1) == 1 else (16, 32, 32))), device=DIVNET_BUNDLE["device"])
    with torch.inference_mode():
        print("  DivNet smoke test output:", float(torch.sigmoid(DIVNET_BUNDLE["model"](_t)).item()))
