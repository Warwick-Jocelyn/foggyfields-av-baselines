#!/usr/bin/env python3
"""Cross-check an EmerNeRF render tree against a SplatAD (ns-render) tree of the same clip:
same camera dirs, EmerNeRF filenames a subset of SplatAD's (= the same held-out timesteps),
and the gt-rgb PNGs byte-identical in pixels (= same frame, same safe_crop, same pixels).
Optionally also the LiDAR dumps: gt_xyz of the same sweep identical up to float32 rounding.
CPU only."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--emer", type=Path, required=True, help="<emernerf run>/renders/val")
ap.add_argument("--splatad", type=Path, required=True, help="<splatad run>/renders/val")
ap.add_argument("--emer-dump", type=Path); ap.add_argument("--splatad-dump", type=Path)
ap.add_argument("-o", "--output", type=Path)
a = ap.parse_args()
out = {"emer": str(a.emer), "splatad": str(a.splatad), "cameras": {}}
ok = True
for cd in sorted((a.emer / "gt-rgb").iterdir()):
    sp = a.splatad / "gt-rgb" / cd.name
    e = sorted(p.name for p in cd.glob("*.png")); s = sorted(p.name for p in sp.glob("*.png"))
    sub = set(e) <= set(s)
    same = all(np.array_equal(np.asarray(Image.open(cd / f)), np.asarray(Image.open(sp / f))) for f in e if f in s)
    out["cameras"][cd.name] = {"n_emer": len(e), "n_splatad": len(s), "emer_subset_of_splatad": sub, "gt_pixels_identical": same}
    ok &= sub and same
    print(f"  {cd.name:28s} emer {len(e):3d} splatad {len(s):3d} subset={sub} gt_identical={same}")
if a.emer_dump and a.splatad_dump:
    out["lidar"] = {}
    for f in sorted(a.emer_dump.glob("frame_*.npz")):
        g = a.splatad_dump / f.name
        if not g.is_file():
            continue
        x, y = np.load(f)["gt_xyz"], np.load(g)["gt_xyz"]
        d = float(np.abs(x - y).max()) if x.shape == y.shape else None
        out["lidar"][f.stem] = {"n": [len(x), len(y)], "max_abs_diff_m": d}
        ok &= d is not None and d < 1e-3
        print(f"  lidar {f.stem}: n {len(x)} vs {len(y)}  max|diff| {d}")
out["ALL_OK"] = bool(ok)
if a.output:
    a.output.write_text(json.dumps(out, indent=2))
print("CROSSCHECK", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
