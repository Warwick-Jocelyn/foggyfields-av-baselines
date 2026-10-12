"""Import nerfstudio's full method registry; pip-install any missing module (max 12 rounds)."""
import importlib, subprocess, sys
MAP = {"cv2": "opencv-python-headless<4.11", "skimage": "scikit-image", "PIL": "Pillow", "yaml": "pyyaml",
       "msgpack_numpy": "msgpack-numpy", "nuscenes": "nuscenes-devkit", "pandaset": "git+https://github.com/scaleapi/pandaset-devkit.git#egg=pandaset&subdirectory=python",
       "viser": "git+https://github.com/atonderski/viser.git@57142e42df8edd4de33fd60a08d6bb6c35970aa1", "sklearn": "scikit-learn",
       "google": "protobuf<=3.20.3", "lpips": "lpips", "torchmetrics": "torchmetrics[image]", "socketio": "python-socketio"}
for i in range(12):
    try:
        import nerfstudio.configs.method_configs  # noqa: imports every model/dataparser/datamanager
        import nerfstudio.scripts.train, nerfstudio.scripts.render, nerfstudio.scripts.eval  # noqa
        print("ALL IMPORTS OK"); sys.exit(0)
    except ImportError as e:
        if not isinstance(e, ModuleNotFoundError):
            # version-skew ImportError: pin the known offenders, else give up loudly
            PINS = {"dataclass_wizard": "dataclass-wizard==0.22.3", "protobuf": "protobuf<=3.20.3,!=3.20.0", "google": "protobuf<=3.20.3,!=3.20.0"}
            key = next((k for k in PINS if k in str(e)), None)
            if key is None: print(f"!! ImportError not auto-fixable: {e}"); sys.exit(1)
            mod, pkg = key, PINS[key]
        else:
            mod = e.name.split(".")[0]; pkg = MAP.get(mod, mod)
        print(f"[round {i}] missing module {mod!r} -> pip install {pkg}", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])
        if r.returncode != 0: print(f"!! pip failed for {pkg}"); sys.exit(1)
        for k in [k for k in sys.modules if k.startswith("nerfstudio")]: del sys.modules[k]
print("!! still failing after 12 rounds"); sys.exit(1)
