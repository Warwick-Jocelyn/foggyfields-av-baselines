#!/usr/bin/env python3
"""T9: synthetic-fog clones of clear clip NCF-02, readable by the unchanged dataparser.

Per variant, creates data_fogsim/clips/<VAR>__02805273/ with:
  - symlinks to the real clip for everything untouched (calib, labels*, masks, splits, radar, ...)
  - fogged lidar/sweep_*.npy (Hahner or LISA, KNOWN beta) + REGENERATED lidar/range_image/
    (same float16+packbits format; drops become true non-returns)
  - fogged images_rectv2/<cam>/*.jpg: Koschmieder I*T + A*(1-T) inside the safe crop, using
    accumulation-normalised NeuRAD clear depth (T8 renders), d=max_range where acc<0.5
  - provenance.json copied + fogsim block; fog_manifest.yaml with beta_true/A/commits
Variants: Hb030 (Hahner beta=0.03), Hb060 (0.06), Lb030 (LISA 0.03), Sramp (per-sweep/-frame
beta ramp 0.005->0.06 along the clip = driving into a fog bank; logged simplification of
spatial fog: beta varies with EGO PROGRESS, not as a static field).
Detection rule (logged): fogged points with intensity < 1/255 are dropped (uint8 floor).
Self-check: regenerating the range image from the CLEAR sweep matches the shipped grid.
"""
import argparse, glob, json, os, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/capstor/scratch/cscs/yitiwang/foggyfields/medium')
from foggyfields.fogsim_adapter import fog_sweep_hahner, fog_sweep_lisa  # noqa: E402

SRC = Path('/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/data_all/clips/NCF-02__02805273')
FF = Path('/capstor/scratch/cscs/yitiwang/foggyfields')
OUT_ROOT = FF / 'data_fogsim' / 'clips'
DEPTH_DIR = FF / 'analysis' / 'depth_all' / 'NCF-02'
AIRLIGHT = 0.75
MAX_RANGE = 120.0
RAMP = (0.005, 0.06)

INTR = json.load(open(SRC / 'calib' / 'lidar_intrinsics.json'))
AZ0 = INTR['column_azimuths_rad'][0]
NCOL = INTR['n_columns']
def load_T_lidar_rig():
    d = json.load(open(SRC / 'calib' / 'T_lidar_rig.json'))
    if isinstance(d, dict):  # take the first 4x4-shaped value whatever the key
        for v in d.values():
            a = np.asarray(v, dtype=np.float64)
            if a.size == 16:
                return a.reshape(4, 4)
        raise KeyError(f'no 4x4 matrix in T_lidar_rig.json keys {list(d)}')
    return np.asarray(d, dtype=np.float64).reshape(4, 4)


TLR = load_T_lidar_rig()


def sweep_beta(variant, idx, n):
    if variant == 'Sramp':
        return RAMP[0] + (RAMP[1] - RAMP[0]) * idx / max(n - 1, 1)
    return {'Hb030': 0.03, 'Hb060': 0.06, 'Lb030': 0.03}[variant]


def fog_fn(variant):
    return fog_sweep_lisa if variant.startswith('L') else fog_sweep_hahner


def to_sensor(xyz):
    Tinv = np.linalg.inv(TLR)
    return xyz @ Tinv[:3, :3].T + Tinv[:3, 3]


def build_range_image(pts_ego):
    """128x3600 grid from ego-frame points, README formula, nearest-return-wins."""
    s = to_sensor(pts_ego[:, :3].astype(np.float64))
    r = np.linalg.norm(s, axis=1)
    az = np.arctan2(s[:, 1], s[:, 0])
    col = np.round((AZ0 - az) / (2 * np.pi / NCOL)).astype(np.int64) % NCOL
    row = pts_ego[:, 5].astype(np.int64)
    rng = np.zeros((128, NCOL), dtype=np.float32)
    inten = np.zeros((128, NCOL), dtype=np.float32)
    order = np.argsort(-r)  # nearest written last wins
    rr, cc = row[order], col[order]
    rng[rr, cc] = r[order].astype(np.float32)
    inten[rr, cc] = pts_ego[order, 3].astype(np.float32)
    mask = rng > 0
    return rng, inten, mask


def save_range_image(path, rng, inten, mask):
    np.savez_compressed(path, range_m=rng.astype(np.float16),
                        intensity=np.round(inten * 255).astype(np.uint8),
                        mask=np.packbits(mask, axis=-1))


def self_check():
    pts = np.load(SRC / 'lidar' / 'sweep_0000.npy')
    rng, inten, mask = build_range_image(pts)
    d = np.load(SRC / 'lidar' / 'range_image' / 'sweep_0000.npz')
    mask0 = np.unpackbits(d['mask'], axis=-1, count=NCOL).astype(bool)
    agree = (mask == mask0).mean()
    both = mask & mask0
    dr = np.abs(rng[both] - d['range_m'].astype(np.float32)[both])
    print(f'[t9] self-check: cell mask agreement {agree:.4f}, '
          f'range median|diff| {np.median(dr):.3f} m (fp16 + collision rule tolerance)', flush=True)
    assert agree > 0.97, 'range-image regeneration does not match the shipped convention'


def make_lidar(variant, dst):
    (dst / 'lidar' / 'range_image').mkdir(parents=True, exist_ok=True)
    sweeps = sorted(glob.glob(str(SRC / 'lidar' / 'sweep_*.npy')))
    for i, f in enumerate(sweeps):
        o = dst / 'lidar' / os.path.basename(f)
        if o.exists():
            continue
        pts = np.load(f)
        beta = sweep_beta(variant, i, len(sweeps))
        fogged = fog_fn(variant)(pts, beta, TLR, rng_seed=1000 + i)
        fogged = fogged[fogged[:, 3] >= 1.0 / 255.0]  # detector floor -> true drops
        np.save(str(o) + '.tmp.npy', fogged.astype(np.float32))
        os.replace(str(o) + '.tmp.npy', o)
        rng, inten, mask = build_range_image(fogged)
        save_range_image(dst / 'lidar' / 'range_image' / os.path.basename(f).replace('.npy', '.npz'),
                         rng, inten, mask)
        if i % 50 == 0:
            print(f'[t9:{variant}] lidar {i}/{len(sweeps)} beta={beta:.3f}', flush=True)
    shutil.copy(SRC / 'lidar' / 'README.md', dst / 'lidar' / 'README.md')
    ri_readme = json.load(open(SRC / 'lidar' / 'range_image' / 'README.json'))
    ri_readme['fogsim'] = f'REGENERATED from fogged points ({variant}); drops are true non-returns'
    json.dump(ri_readme, open(dst / 'lidar' / 'range_image' / 'README.json', 'w'), indent=1)


def frame_depth_index():
    """image_filename -> depth npz path (from the T8 dump, train+eval)."""
    idx = {}
    for f in glob.glob(str(DEPTH_DIR / '*.npz')):
        d = np.load(f, allow_pickle=True)
        idx[str(d['image_filename'])] = f
    return idx


def make_images(variant, dst, didx):
    from PIL import Image
    n_missing = 0
    cams = sorted(d for d in os.listdir(SRC / 'images_rectv2')
                  if (SRC / 'images_rectv2' / d).is_dir())
    n_frames = max(len(glob.glob(str(SRC / 'images_rectv2' / c / '*.jpg'))) for c in cams)
    assert n_frames > 100, f'frame count {n_frames} implausible'  # the n_frames=0 bug (T11)
    for cam in cams:
        (dst / 'images_rectv2' / cam).mkdir(parents=True, exist_ok=True)
        for f in sorted(glob.glob(str(SRC / 'images_rectv2' / cam / '*.jpg'))):
            o = dst / 'images_rectv2' / cam / os.path.basename(f)
            if o.exists():
                continue
            key = f
            if key not in didx:
                n_missing += 1
                shutil.copy(f, o)  # no depth (shouldn't happen) -> copy clear
                continue
            d = np.load(didx[key], allow_pickle=True)
            depth = d['depth'].astype(np.float32); acc = d['acc'].astype(np.float32)
            x, y, w, h = [int(v) for v in d['crop']]
            dn = depth / np.clip(acc, 0.3, None)
            dn[acc < 0.5] = MAX_RANGE
            dn = np.clip(dn, 0, MAX_RANGE)
            img = np.asarray(Image.open(f), dtype=np.float32) / 255.0
            fi = int(os.path.basename(f).split('_')[1].split('.')[0])
            beta = sweep_beta(variant, fi, n_frames)
            # upsample ray-res depth x3 to the crop size
            dcrop = np.array(Image.fromarray(dn).resize((w, h), Image.BILINEAR))
            T = np.exp(-beta * dcrop)[..., None]
            crop_img = img[y:y + h, x:x + w]
            fogged = T * crop_img + (1 - T) * AIRLIGHT
            img[y:y + h, x:x + w] = fogged
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(o, quality=98)
    if n_missing:
        print(f'[t9:{variant}] WARNING {n_missing} frames had no depth render', flush=True)
    # post-write verification: back-solve beta from one mid-clip stored frame; abort on mismatch
    cam = 'camera_front_wide_120fov'
    fi_chk = 100
    f = str(SRC / 'images_rectv2' / cam / f'frame_{fi_chk:04d}.jpg')
    d = np.load(didx[f], allow_pickle=True)
    depth = d['depth'].astype(np.float32); acc = d['acc'].astype(np.float32)
    x, y, w, h = [int(v) for v in d['crop']]
    dn = depth / np.clip(acc, 0.3, None); dn[acc < 0.5] = MAX_RANGE; dn = np.clip(dn, 0, MAX_RANGE)
    img = np.asarray(Image.open(f), dtype=np.float32) / 255.0
    dcrop = np.array(Image.fromarray(dn).resize((w, h), Image.BILINEAR))
    stored = np.asarray(Image.open(dst / 'images_rectv2' / cam / f'frame_{fi_chk:04d}.jpg'),
                        dtype=np.float32) / 255.0
    sc, ic = stored[y:y + h, x:x + w].mean(-1), img[y:y + h, x:x + w].mean(-1)
    ok = (np.abs(ic - AIRLIGHT) > 0.15) & (dcrop > 8) & (dcrop < 60)
    That = np.clip((sc[ok] - AIRLIGHT) / (ic[ok] - AIRLIGHT), 1e-3, 1)
    b_imp = float(np.median(-np.log(That) / dcrop[ok]))
    b_want = sweep_beta(variant, fi_chk, n_frames)
    print(f'[t9:{variant}] VERIFY frame {fi_chk}: implied beta {b_imp:.4f} vs intended {b_want:.4f}', flush=True)
    assert abs(b_imp - b_want) <= max(0.2 * b_want, 0.004), 'stored fog does not match intended beta'


def make_variant(variant, stage='all'):
    name = f'NCF02F-{variant}__02805273'
    dst = OUT_ROOT / name
    dst.mkdir(parents=True, exist_ok=True)
    for item in os.listdir(SRC):
        if item in ('lidar', 'images_rectv2', 'provenance.json'):
            continue
        link = dst / item
        if not link.exists():
            os.symlink(SRC / item, link)
    prov = json.load(open(SRC / 'provenance.json'))
    prov['fogsim'] = dict(variant=variant, source='NCF-02__02805273', source_root=str(SRC),
                          simulator='LISA' if variant.startswith('L') else 'Hahner',
                          beta_true=('ramp %.3f-%.3f per sweep/frame' % RAMP) if variant == 'Sramp'
                                    else sweep_beta(variant, 0, 1),
                          airlight=AIRLIGHT, camera_depth='NeuRAD clear ckpt, acc-normalised, '
                          'd=120 where acc<0.5', detector_floor='intensity < 1/255 dropped')
    json.dump(prov, open(dst / 'provenance.json', 'w'), indent=1)
    if stage in ('all', 'lidar'):
        make_lidar(variant, dst)
    if stage in ('all', 'images'):
        didx = frame_depth_index()
        make_images(variant, dst, didx)
    if stage == 'lidar':
        print(f'[t9] variant {variant} lidar stage done', flush=True)
        return
    (dst / 'fog_manifest.yaml').write_text(
        f"variant: {variant}\nbeta_true: {prov['fogsim']['beta_true']}\nairlight: {AIRLIGHT}\n"
        f"simulator: {prov['fogsim']['simulator']}\nmax_range_m: {MAX_RANGE}\n"
        f"depth_source: T8 NeuRAD clear renders (acc-normalised)\nseed_base: 1000+sweep_idx\n")
    print(f'[t9] variant {variant} DONE -> {dst}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True,
                    choices=['Hb030', 'Hb060', 'Lb030', 'Sramp', 'selfcheck'])
    ap.add_argument('--stage', default='all', choices=['all', 'lidar', 'images'])
    args = ap.parse_args()
    if args.variant == 'selfcheck':
        self_check()
    else:
        self_check()
        make_variant(args.variant, args.stage)
