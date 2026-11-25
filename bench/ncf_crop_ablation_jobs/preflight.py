#!/usr/bin/env python3
"""CPU-only pre-flight for the NCF crop/mask ablation. Runs before any GPU is touched.

It fails loudly on anything that would otherwise waste GPU hours or, worse, produce a
plausible-looking but wrong number:

  1. every requested clip resolves, and our resolver agrees with the dataparser's;
  2. images_rectv2/, masks_rectv2/, calib/K_rectified_v2.json exist with all 7 cameras;
  3. **the premise of the whole study**: valid_mask is 100% valid inside safe_crop, so
     ``safe_crop n valid_mask == safe_crop`` and the common region really is common;
  4. the split file exists and has the same held-out frame count on every clip;
  5. the dataparser exposes BOTH ``apply_safe_crop`` and ``use_valid_masks``;
  6. a usable mp4 encoder exists (no system ffmpeg on this cluster);
  7. the projected host-RAM image cache for NLANES concurrent runs fits in the node.

Writes ``$RES_ROOT/safe_crop_audit.json`` -- the per-camera "how much of the valid region
does safe_crop keep" table that the SUMMARY cites.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_paths import CAMERAS, load_geometry, resolve_clip_root, cross_check  # noqa: E402

FAIL: list[str] = []
WARN: list[str] = []


def fail(msg: str) -> None:
    FAIL.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARN.append(msg)
    print(f"  WARN  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def check_encoder() -> None:
    print("[7/7] mp4 encoder")
    found = []
    try:
        import av  # noqa: F401

        import av.codec

        av.codec.Codec("libx264", "w")
        found.append("PyAV/libx264")
    except Exception as exc:  # noqa: BLE001
        warn(f"PyAV libx264 unavailable ({type(exc).__name__}: {exc})")
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(exe).is_file():
            found.append(f"imageio-ffmpeg ({Path(exe).name})")
    except Exception as exc:  # noqa: BLE001
        warn(f"imageio-ffmpeg unavailable ({type(exc).__name__}: {exc})")
    try:
        import cv2

        found.append(f"cv2 {cv2.__version__} (mp4v fallback)")
    except Exception as exc:  # noqa: BLE001
        warn(f"cv2 unavailable ({type(exc).__name__}: {exc})")
    if shutil.which("ffmpeg"):
        found.append("system ffmpeg")
    if not found:
        fail("no mp4 encoder at all (need PyAV+libx264, imageio-ffmpeg, or cv2) -- the video stage cannot run")
    else:
        ok("encoders: " + ", ".join(found))


def check_dataparser_flags() -> None:
    print("[5/7] dataparser flags")
    try:
        from nerfstudio.data.dataparsers.nvidia_av_fog_dataparser import NvidiaAvFogDataParserConfig
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot import NvidiaAvFogDataParserConfig ({type(exc).__name__}: {exc})")
        return
    fields = NvidiaAvFogDataParserConfig.__dataclass_fields__
    for flag in ("apply_safe_crop", "use_valid_masks", "rectification", "splits_json"):
        if flag in fields:
            ok(f"--{flag.replace('_', '-')} present (default {fields[flag].default!r})")
        else:
            fail(
                f"dataparser has no '{flag}' field -- the V2 ablation branch has not landed in "
                "code/neurad-studio yet. Do not launch."
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("DATA_ROOT", ".")))
    ap.add_argument("--clips", default=os.environ.get("CLIPS", ""))
    ap.add_argument("--methods", default=os.environ.get("METHODS", "splatad neurad"))
    ap.add_argument("--variants", default=os.environ.get("VARIANTS", "safe mask"))
    ap.add_argument("--split", default=os.environ.get("SPLIT", "nvs_50_50"))
    ap.add_argument("--nlanes", type=int, default=int(os.environ.get("NLANES", "4")))
    ap.add_argument("--mem-budget-frac", type=float, default=float(os.environ.get("MEM_BUDGET_FRAC", "0.75")))
    ap.add_argument("--allow-big-cache", type=int, default=int(os.environ.get("ALLOW_BIG_CACHE", "0")))
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RES_ROOT", ".")) / "safe_crop_audit.json")
    ap.add_argument("--skip-mask-audit", action="store_true", help="skip the (slow) per-pixel mask check")
    a = ap.parse_args()

    clips = a.clips.split()
    methods = a.methods.split()
    variants = a.variants.split()
    if not clips:
        print("no clips requested (set CLIPS or --clips)", file=sys.stderr)
        return 2

    print(f"PRE-FLIGHT  data={a.data}")
    print(f"            clips={clips} methods={methods} variants={variants} split={a.split}")

    audit: dict = {"data_root": str(a.data), "split": a.split, "clips": {}}
    eval_counts: dict[str, int] = {}
    px_per_ts: dict[str, dict[str, int]] = {}  # clip -> {variant: pixels per timestep}
    n_timesteps: dict[str, int] = {}

    for clip in clips:
        print(f"[1-4/7] clip {clip}")
        try:
            root = resolve_clip_root(a.data, clip)
        except Exception as exc:  # noqa: BLE001
            fail(f"{clip}: {type(exc).__name__}: {exc}")
            continue
        ok(f"{clip}: {root}")
        try:
            status = cross_check(a.data, clip, root)
            (ok if status.startswith("OK") else warn)(f"{clip}: dataparser cross-check {status}")
        except Exception as exc:  # noqa: BLE001
            fail(f"{clip}: {exc}")

        try:
            geo = load_geometry(root)
        except Exception as exc:  # noqa: BLE001
            fail(f"{clip}: {type(exc).__name__}: {exc}")
            continue

        img_dir, mask_dir = root / "images_rectv2", root / "masks_rectv2"
        for d in (img_dir, mask_dir):
            if not d.is_dir():
                fail(f"{clip}: {d} missing")
        if not img_dir.is_dir() or not mask_dir.is_dir():
            continue

        # frame count per camera must agree
        counts = {}
        for cam in CAMERAS:
            cd = img_dir / cam
            if not cd.is_dir():
                fail(f"{clip}: {cd} missing")
                continue
            counts[cam] = len(list(cd.glob("frame_*.jpg"))) or len(list(cd.glob("frame_*.png")))
        if counts and len(set(counts.values())) != 1:
            fail(f"{clip}: cameras disagree on frame count: {counts}")
        elif counts:
            n_timesteps[clip] = next(iter(counts.values()))
            ok(f"{clip}: {n_timesteps[clip]} timesteps x {len(counts)} cameras")

        # split file
        sp = root / "splits" / f"{a.split}.json"
        if not sp.is_file():
            fail(f"{clip}: split file {sp} missing")
        else:
            sj = json.loads(sp.read_text())
            ne = len(sj.get("eval_indices", []))
            nt = len(sj.get("train_indices", []))
            eval_counts[clip] = ne
            if ne == 0:
                fail(f"{clip}: split {a.split} has no eval_indices -- nothing to score")
            else:
                ok(f"{clip}: split {a.split}: {nt} train / {ne} held-out timesteps")

        # ---- THE invariant: safe_crop contains zero invalid pixels -------------------
        cam_audit = {}
        tot_valid = tot_safe = 0
        for cam in CAMERAS:
            g = geo[cam]
            mp_ = mask_dir / cam / "valid_mask.png"
            entry = dict(g)
            if a.skip_mask_audit:
                entry["mask_checked"] = False
            elif not mp_.is_file():
                fail(f"{clip}/{cam}: {mp_} missing (both variants need it: mask for training, both for the wide metric)")
                entry["mask_checked"] = False
            else:
                m = np.asarray(Image.open(mp_))
                if m.ndim == 3:
                    m = m[..., 0]
                if (m.shape[1], m.shape[0]) != (g["canvas_w"], g["canvas_h"]):
                    fail(
                        f"{clip}/{cam}: mask is {m.shape[1]}x{m.shape[0]} but K_rectified_v2 canvas is "
                        f"{g['canvas_w']}x{g['canvas_h']}"
                    )
                    entry["mask_checked"] = False
                else:
                    valid = m > 128
                    inside = valid[g["y"] : g["y"] + g["h"], g["x"] : g["x"] + g["w"]]
                    n_invalid_inside = int((~inside).sum())
                    n_valid = int(valid.sum())
                    n_safe = g["w"] * g["h"]
                    entry.update(
                        mask_checked=True,
                        invalid_pixels_inside_safe_crop=n_invalid_inside,
                        valid_pixels=n_valid,
                        safe_crop_pixels=n_safe,
                        frac_of_valid_kept_by_safe_crop=n_safe / n_valid,
                        frac_of_canvas_kept_by_safe_crop=n_safe / (g["canvas_w"] * g["canvas_h"]),
                        extra_valid_pixels_outside_safe_crop=n_valid - n_safe,
                    )
                    tot_valid += n_valid
                    tot_safe += n_safe
                    if n_invalid_inside:
                        fail(
                            f"{clip}/{cam}: {n_invalid_inside} INVALID pixels inside safe_crop. "
                            "The common-region premise (safe_crop n valid_mask == safe_crop) is violated; "
                            "the primary metric would not be a fair common region. Do not launch."
                        )
                    if n_safe > n_valid:
                        fail(f"{clip}/{cam}: safe_crop ({n_safe}) is larger than the valid region ({n_valid})")
            cam_audit[cam] = entry
        if tot_valid:
            ok(
                f"{clip}: safe_crop keeps {tot_safe / tot_valid * 100:.1f}% of all valid pixels "
                f"({tot_safe / 1e6:.1f}M of {tot_valid / 1e6:.1f}M per timestep)"
            )
        audit["clips"][clip] = {"clip_root": str(root), "cameras": cam_audit,
                                "frac_of_valid_kept": (tot_safe / tot_valid) if tot_valid else None}
        px_per_ts[clip] = {
            "safe": sum(geo[c]["w"] * geo[c]["h"] for c in CAMERAS),
            "mask": sum(geo[c]["canvas_w"] * geo[c]["canvas_h"] for c in CAMERAS),
        }

    if eval_counts and len(set(eval_counts.values())) != 1:
        warn(f"held-out frame counts differ across clips: {eval_counts} (the paired comparison still works, "
             "but the per-clip numbers are averages over different n)")

    check_dataparser_flags()

    # ---- [6/7] host-RAM image cache ---------------------------------------------------
    print("[6/7] image cache vs node memory")
    try:
        mem_total = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) * 1024
    except Exception:  # noqa: BLE001
        mem_total = 0
    # nerfstudio caches every train AND eval image. SplatAD caches uint8 (1 B/channel),
    # NeuRAD's CacheDataloader caches float32 (4 B/channel) and defaults to all images.
    bytes_per_channel = {"splatad": 1, "neurad": 4, "unisim": 4}
    est = {}
    for m in methods:
        for v in variants:
            if v not in ("safe", "mask"):
                fail(f"unknown variant '{v}' (expected safe|mask)")
                continue
            worst = max((px_per_ts[c][v] for c in px_per_ts), default=0)
            ts = max(n_timesteps.values(), default=199)
            est[f"{m}/{v}"] = worst * ts * 3 * bytes_per_channel.get(m, 4)
    if est:
        for k, v in sorted(est.items(), key=lambda kv: -kv[1]):
            print(f"        {k:16s} ~{v / 1024 ** 3:.1f} GiB host RAM per run")
        top = sorted(est.values(), reverse=True)[: a.nlanes]
        need = sum(top)
        budget = mem_total * a.mem_budget_frac
        msg = (f"{a.nlanes} concurrent lanes need ~{need / 1024 ** 3:.0f} GiB; node MemTotal "
               f"{mem_total / 1024 ** 3:.0f} GiB, budget {budget / 1024 ** 3:.0f} GiB")
        if mem_total == 0:
            warn("could not read /proc/meminfo; skipping the memory budget check")
        elif need > budget:
            m = (f"{msg}. Reduce NLANES, or cap the cache, e.g. "
                 f"NEURAD_EXTRA='--pipeline.datamanager.train-num-images-to-sample-from 200 "
                 f"--pipeline.datamanager.train-num-times-to-repeat-images 500 "
                 f"--pipeline.datamanager.num-processes 2'. Set ALLOW_BIG_CACHE=1 to proceed anyway.")
            (warn if a.allow_big_cache else fail)(m)
        else:
            ok(msg)

    # ---- disk ------------------------------------------------------------------------
    du = shutil.disk_usage(a.data if a.data.exists() else Path("/"))
    n_runs = len(clips) * len(methods) * len(variants)
    print(f"        free disk {du.free / 1024 ** 4:.1f} TiB; {n_runs} runs x ~4-7 GiB of PNG renders "
          f"= ~{n_runs * 6} GiB in outputs/")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    audit["fail"], audit["warn"] = FAIL, WARN
    a.out.write_text(json.dumps(audit, indent=2))
    print(f"\nwrote {a.out}")
    print(f"PRE-FLIGHT: {len(FAIL)} failure(s), {len(WARN)} warning(s)")
    if FAIL:
        print("PREFLIGHT_FAILED")
        return 1
    print("PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
