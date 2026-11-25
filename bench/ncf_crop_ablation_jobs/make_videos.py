#!/usr/bin/env python3
"""Side-by-side render-vs-GT mp4 for every held-out frame and every one of the 7 cameras,
plus one small montage image per run.

There is no system ffmpeg on this cluster, so the encoder is chosen at runtime:
PyAV/libx264 (preferred; the wheel bundles its own ffmpeg), then imageio-ffmpeg (which
ships an aarch64 ffmpeg binary), then cv2's mp4v as a last resort. If none works the
script fails rather than writing nothing and returning 0.

For the `mask` variant the full canvas is shown with the safe_crop rectangle drawn on it,
so the unconstrained black-corner region -- the reason the primary metric is restricted to
that rectangle -- is visible. For the `safe` variant the frame *is* that rectangle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_paths import CAMERAS, load_geometry, resolve_clip_root  # noqa: E402

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
]


def get_font(size: int):
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:  # noqa: BLE001
            pass
    try:  # viser ships a TTF inside the venv
        import viser

        cand = Path(viser.__file__).parent / "client/public/Inter-VariableFont_slnt,wght.ttf"
        if cand.is_file():
            return ImageFont.truetype(str(cand), size)
    except Exception:  # noqa: BLE001
        pass
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


# --------------------------------------------------------------------------- encoders
class Mp4Writer:
    """Minimal RGB-frame mp4 writer with three interchangeable backends."""

    def __init__(self, path: Path, width: int, height: int, fps: int, backend: str = "auto"):
        if width % 2 or height % 2:
            raise ValueError(f"mp4 needs even dimensions, got {width}x{height}")
        self.path, self.w, self.h, self.fps = Path(path), width, height, fps
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = None
        errs = []
        order = ["av", "imageio", "cv2"] if backend == "auto" else [backend]
        for b in order:
            try:
                getattr(self, f"_open_{b}")()
                self.backend = b
                break
            except Exception as exc:  # noqa: BLE001
                errs.append(f"{b}: {type(exc).__name__}: {exc}")
        if self.backend is None:
            raise RuntimeError(f"no usable mp4 encoder for {self.path}. Tried -- " + " | ".join(errs))

    def _open_av(self):
        import av

        self._c = av.open(str(self.path), mode="w")
        self._s = self._c.add_stream("libx264", rate=self.fps)
        self._s.width, self._s.height, self._s.pix_fmt = self.w, self.h, "yuv420p"
        self._s.options = {"crf": "20", "preset": "veryfast"}
        self._av = av

    def _open_imageio(self):
        import imageio.v2 as iio

        self._w = iio.get_writer(str(self.path), fps=self.fps, codec="libx264",
                                 quality=8, macro_block_size=2, ffmpeg_log_level="error")

    def _open_cv2(self):
        import cv2

        self._cv2 = cv2
        self._vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.w, self.h))
        if not self._vw.isOpened():
            raise RuntimeError("cv2.VideoWriter did not open")

    def append(self, rgb: np.ndarray) -> None:
        if rgb.shape[:2] != (self.h, self.w):
            raise ValueError(f"frame is {rgb.shape[1]}x{rgb.shape[0]}, writer is {self.w}x{self.h}")
        if self.backend == "av":
            frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
            for pkt in self._s.encode(frame):
                self._c.mux(pkt)
        elif self.backend == "imageio":
            self._w.append_data(rgb)
        else:
            self._vw.write(self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.backend == "av":
            for pkt in self._s.encode():
                self._c.mux(pkt)
            self._c.close()
        elif self.backend == "imageio":
            self._w.close()
        else:
            self._vw.release()


# --------------------------------------------------------------------------- compositing
def frame_paths(d: Path) -> list[Path]:
    return sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))


def scaled(im: Image.Image, panel_w: int) -> Image.Image:
    if im.width == panel_w:
        return im
    h = max(2, int(round(im.height * panel_w / im.width)))
    return im.resize((panel_w, h), Image.Resampling.BILINEAR)


def banner(img: Image.Image, text: str, font) -> None:
    d = ImageDraw.Draw(img)
    pad = 4
    try:
        tw, th = d.textbbox((0, 0), text, font=font)[2:]
    except Exception:  # noqa: BLE001
        tw, th = (8 * len(text), 14)
    d.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    d.text((pad, pad), text, fill=(255, 235, 0), font=font)


def compose(pred: Image.Image, gt: Image.Image, panel_w: int, left_label: str, right_label: str,
            font, box: tuple[int, int, int, int] | None, scale: float, gap: int = 6) -> np.ndarray:
    p, g = scaled(pred, panel_w), scaled(gt, panel_w)
    if box is not None:
        x, y, w, h = [int(round(v * scale)) for v in box]
        for im in (p, g):
            ImageDraw.Draw(im).rectangle([x, y, x + w - 1, y + h - 1], outline=(255, 235, 0), width=2)
    banner(p, left_label, font)
    banner(g, right_label, font)
    H = max(p.height, g.height)
    canvas = Image.new("RGB", (panel_w * 2 + gap, H), (28, 28, 28))
    canvas.paste(p, (0, 0))
    canvas.paste(g, (panel_w + gap, 0))
    arr = np.asarray(canvas)
    return arr[: arr.shape[0] - arr.shape[0] % 2, : arr.shape[1] - arr.shape[1] % 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", type=Path, required=True, help="<ns-render output>/val (holds rgb/ and gt-rgb/)")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--variant", required=True, choices=["safe", "mask"])
    ap.add_argument("--method", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel-width", type=int, default=960)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--montage-frames", type=int, default=4)
    ap.add_argument("--montage-cell-width", type=int, default=240)
    ap.add_argument("--backend", default="auto", choices=["auto", "av", "imageio", "cv2"])
    ap.add_argument("--cameras", default="")
    a = ap.parse_args()

    geo = load_geometry(resolve_clip_root(a.data, a.sequence))
    rgb_root, gt_root = a.renders / "rgb", a.renders / "gt-rgb"
    for d in (rgb_root, gt_root):
        if not d.is_dir():
            raise FileNotFoundError(f"{d} missing")
    want = a.cameras.split() if a.cameras else list(CAMERAS)
    missing = [c for c in want if not (rgb_root / c).is_dir()]
    if missing:
        raise FileNotFoundError(f"no renders for cameras {missing} under {rgb_root}")

    a.out.mkdir(parents=True, exist_ok=True)
    font = get_font(max(13, a.panel_width // 55))
    tag = f"{a.method}/{a.variant}" if a.method else a.variant
    manifest: dict = {"method": a.method, "variant": a.variant, "clip": a.sequence,
                      "fps": a.fps, "panel_width": a.panel_width, "videos": {}}
    montage_rows: list[list[np.ndarray]] = []

    for cam in want:
        g = geo[cam]
        preds = frame_paths(rgb_root / cam)
        if not preds:
            raise FileNotFoundError(f"no frames under {rgb_root / cam}")
        box = None if a.variant == "safe" else (g["x"], g["y"], g["w"], g["h"])

        first = Image.open(preds[0])
        scale = a.panel_width / first.width
        probe = compose(first.convert("RGB"), first.convert("RGB"), a.panel_width,
                        "x", "y", font, box, scale)
        vw = Mp4Writer(a.out / f"{cam}.mp4", probe.shape[1], probe.shape[0], a.fps, a.backend)

        montage_idx = set(np.linspace(0, len(preds) - 1, min(a.montage_frames, len(preds))).round().astype(int).tolist())
        row: list[np.ndarray] = []
        for i, pp in enumerate(preds):
            gp = gt_root / cam / pp.name
            if not gp.is_file():
                raise FileNotFoundError(f"missing ground truth {gp} for {pp}")
            pred, gt = Image.open(pp).convert("RGB"), Image.open(gp).convert("RGB")
            if pred.size != gt.size:
                raise ValueError(f"{cam}/{pp.name}: render {pred.size} != gt {gt.size}")
            fr = compose(pred, gt, a.panel_width, f"{tag} {a.sequence} {cam} {pp.stem}",
                         "ground truth", font, box, scale)
            vw.append(fr)
            if i in montage_idx:
                cell = compose(pred, gt, a.montage_cell_width, cam.replace("camera_", ""), "GT",
                               get_font(11), box, a.montage_cell_width / pred.width, gap=2)
                row.append(cell)
        vw.close()
        mp4 = a.out / f"{cam}.mp4"
        size = mp4.stat().st_size if mp4.is_file() else 0
        if size < 1024:
            raise RuntimeError(f"{mp4} is {size} B -- the {vw.backend} encoder produced nothing usable")
        manifest["videos"][cam] = {"file": mp4.name, "frames": len(preds), "bytes": size,
                                   "backend": vw.backend, "size_wh": [probe.shape[1], probe.shape[0]]}
        print(f"  {cam:28s} {len(preds):4d} frames -> {mp4.name} ({size / 1e6:.1f} MB, {vw.backend})")
        montage_rows.append(row)

    # ---- montage ---------------------------------------------------------------------
    if montage_rows and montage_rows[0]:
        cw = max(c.shape[1] for r in montage_rows for c in r)
        rhs = [max(c.shape[0] for c in r) for r in montage_rows]
        ncol = max(len(r) for r in montage_rows)
        mont = Image.new("RGB", (cw * ncol, sum(rhs)), (28, 28, 28))
        y = 0
        for r, rh in zip(montage_rows, rhs):
            for j, c in enumerate(r):
                mont.paste(Image.fromarray(c), (j * cw, y))
            y += rh
        mpath = a.out / "montage.jpg"
        mont.save(mpath, quality=85)
        manifest["montage"] = {"file": mpath.name, "size_wh": list(mont.size)}
        print(f"  montage -> {mpath} ({mont.size[0]}x{mont.size[1]})")

    (a.out / "VIDEO_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['videos'])} videos to {a.out}")
    print("VIDEO_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
