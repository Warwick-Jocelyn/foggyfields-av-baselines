#!/usr/bin/env python3
"""Shared path/geometry resolution for the NCF crop-mask ablation.

Everything in this harness needs the same three facts about a clip:

  * where the clip directory is (``<data_root>/clips/NCF-02__<hash>`` today, possibly
    ``<data_root>/processed/<seq>/processed_pinhole`` once the shim lands),
  * the per-camera ``safe_crop`` rectangle from ``calib/K_rectified_v2.json``, and
  * the full v2 canvas size.

The resolution order mirrors ``nvidia_av_fog_dataparser._resolve_clip_root`` exactly.
``--cross-check`` imports that function and asserts the two agree, so the harness can
never silently score a different clip than the one that was trained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Fixed camera order (same tuple as the dataparser's NVIDIA_AV_FOG_CAMERAS).
CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_front_tele_30fov",
    "camera_rear_tele_30fov",
)


def resolve_clip_root(data_root: Path, sequence: str) -> Path:
    """Locate the clip directory for ``sequence`` under ``data_root``.

    Mirrors the dataparser: live ``clips/`` layout first (optionally under a
    ``NVIDIA_*_Field`` subset dir), then the legacy ``processed/<x>/processed_pinhole``
    layout. Ambiguity is an error, never a guess.
    """
    data_root = Path(data_root)
    roots: List[Path] = []
    if (data_root / "clips").is_dir():
        roots.append(data_root / "clips")
    for subset in ("NVIDIA_FOGGY_Field", "NVIDIA_CLEAR_Field"):
        if (data_root / subset / "clips").is_dir():
            roots.append(data_root / subset / "clips")

    candidates: List[Path] = []
    for root in roots:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            clip_id, _, clip_hash = child.name.partition("__")
            if sequence in (child.name, clip_id, clip_hash) or clip_id.lower() == sequence.lower():
                candidates.append(child)
    if len(candidates) > 1:
        raise ValueError(f"sequence '{sequence}' matches several clips: {[str(c) for c in candidates]}")
    if len(candidates) == 1:
        return candidates[0]

    processed = data_root / "processed"
    if not processed.is_dir():
        avail = sorted(c.name for r in roots for c in r.iterdir() if c.is_dir())
        raise FileNotFoundError(
            f"sequence '{sequence}' not found. roots={[str(r) for r in roots]} available={avail} "
            f"(and {data_root}/processed does not exist)"
        )
    legacy = [c for c in sorted(processed.iterdir()) if c.is_dir() and (c.name.startswith(f"{sequence}_") or sequence in c.name)]
    if len(legacy) != 1:
        raise ValueError(
            f"sequence '{sequence}' matched {len(legacy)} legacy dirs under {processed}: {[c.name for c in legacy]}"
        )
    clip = legacy[0] / "processed_pinhole"
    if not clip.is_dir():
        raise FileNotFoundError(f"no processed_pinhole/ under {legacy[0]}")
    return clip


def cross_check(data_root: Path, sequence: str, resolved: Path) -> str:
    """Assert our resolver agrees with the dataparser's. Returns a status string."""
    try:
        from nerfstudio.data.dataparsers.nvidia_av_fog_dataparser import _resolve_clip_root
    except Exception as exc:  # noqa: BLE001
        return f"SKIPPED (cannot import dataparser: {type(exc).__name__}: {exc})"
    theirs = Path(_resolve_clip_root(Path(data_root), sequence)).resolve()
    ours = Path(resolved).resolve()
    if theirs != ours:
        raise AssertionError(
            f"clip resolution disagrees for '{sequence}': harness={ours} dataparser={theirs}. "
            "Refusing to continue -- metrics would be computed against the wrong clip."
        )
    return "OK"


def load_geometry(clip_root: Path) -> Dict[str, dict]:
    """Per-camera v2 canvas + safe_crop rectangle.

    Returns ``{cam: {canvas_w, canvas_h, valid_fraction, x, y, w, h}}`` where x/y/w/h is
    the safe_crop rectangle in FULL-CANVAS pixel coordinates.
    """
    k2 = Path(clip_root) / "calib" / "K_rectified_v2.json"
    if not k2.is_file():
        raise FileNotFoundError(
            f"{k2} missing -- this ablation is defined only on rectification v2 "
            "(safe_crop lives in that file)."
        )
    raw = json.loads(k2.read_text())
    out: Dict[str, dict] = {}
    for cam, K in raw.items():
        sc = K.get("safe_crop")
        if sc is None:
            raise ValueError(f"{k2}: camera {cam} has no safe_crop block; the ablation is undefined without it")
        out[cam] = {
            "canvas_w": int(K["w"]),
            "canvas_h": int(K["h"]),
            "valid_fraction": float(K.get("valid_fraction", float("nan"))),
            "x": int(sc["x"]),
            "y": int(sc["y"]),
            "w": int(sc["w"]),
            "h": int(sc["h"]),
        }
        g = out[cam]
        if g["x"] < 0 or g["y"] < 0 or g["x"] + g["w"] > g["canvas_w"] or g["y"] + g["h"] > g["canvas_h"]:
            raise ValueError(f"{k2}: safe_crop for {cam} is not inside the canvas: {g}")
    missing = [c for c in CAMERAS if c not in out]
    if missing:
        raise ValueError(f"{k2} is missing cameras {missing}; expected all of {list(CAMERAS)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--cross-check", action="store_true", help="assert agreement with the dataparser's resolver")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--print-root", action="store_true", help="print only the clip root (for shell use)")
    a = ap.parse_args()

    root = resolve_clip_root(a.data, a.sequence)
    if a.print_root:
        print(root)
        return 0
    status = cross_check(a.data, a.sequence, root) if a.cross_check else "not requested"
    geo = load_geometry(root)
    if a.json:
        print(json.dumps({"clip_root": str(root), "cross_check": status, "geometry": geo}, indent=2))
    else:
        print(f"clip_root   {root}")
        print(f"cross-check {status}")
        for cam in CAMERAS:
            g = geo[cam]
            print(
                f"  {cam:28s} canvas {g['canvas_w']}x{g['canvas_h']}  "
                f"safe_crop ({g['x']},{g['y']}) {g['w']}x{g['h']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
