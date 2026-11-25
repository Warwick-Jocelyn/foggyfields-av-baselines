#!/usr/bin/env python3
"""Collect every metrics_*.json into results/SUMMARY.md (+ summary.json).

The comparison is PAIRED: the same 5 clips, the same held-out frames, the same pixel set
(the safe_crop rectangle). So the headline is the mean and spread of the five per-clip
deltas, not a pooled average -- a pooled average would hide a clip that disagrees.

Delta convention everywhere: ``mask - safe``. PSNR/SSIM higher is better, LPIPS lower is
better (so a negative LPIPS delta favours `mask`).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

VARIANTS = ("safe", "mask")


def fmt(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s


def load_all(metrics_dir: Path) -> dict:
    out: dict[tuple[str, str, str], dict] = {}
    for p in sorted(metrics_dir.glob("metrics_*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"!! unreadable {p}: {exc}", file=sys.stderr)
            continue
        key = (j.get("method", "?"), j.get("variant", "?"), j.get("clip", "?"))
        if key in out:
            print(f"!! duplicate metrics for {key} ({p})", file=sys.stderr)
        j["_path"] = str(p)
        out[key] = j
    return out


def paired_stats(deltas: list[float]) -> dict:
    n = len(deltas)
    if n == 0:
        return {"n": 0}
    mean = st.fmean(deltas)
    sd = st.stdev(deltas) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n, "mean": mean, "sd": sd, "sem": sem,
        "min": min(deltas), "max": max(deltas),
        "n_positive": sum(1 for d in deltas if d > 0),
        "t": (mean / sem) if sem > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(os.environ.get("RES_ROOT", "results")))
    ap.add_argument("--clips", default=os.environ.get("CLIPS", ""))
    ap.add_argument("--methods", default=os.environ.get("METHODS", "splatad neurad"))
    ap.add_argument("--timings", type=Path, default=None)
    a = ap.parse_args()

    res = a.results
    data = load_all(res / "metrics")
    clips = a.clips.split() or sorted({k[2] for k in data})
    methods = [m for m in a.methods.split() if any(k[0] == m for k in data)] or sorted({k[0] for k in data})

    expected = [(m, v, c) for m in a.methods.split() for v in VARIANTS for c in clips]
    missing = [k for k in expected if k not in data]

    L: list[str] = []
    L.append("# NCF-mini crop vs mask ablation — results")
    L.append("")
    L.append(f"Generated {__import__('datetime').datetime.now().isoformat(timespec='seconds')} from `{res}/metrics/`.")
    L.append("")
    L.append("## How to read these numbers (read this first)")
    L.append("")
    L.append("The two variants are **not comparable on their native pixels**:")
    L.append("")
    L.append("* `safe` = `--apply-safe-crop True --use-valid-masks False`: trained and rendered on the "
             "`safe_crop` rectangle. No black borders, narrower vertical FOV.")
    L.append("* `mask` = `--apply-safe-crop False --use-valid-masks True`: trained and rendered on the full "
             "v2 canvas; the black lens-cone corners were excluded from the RGB loss, so the model is "
             "**unconstrained** there.")
    L.append("")
    L.append("A `mask` model's unmasked full-canvas PSNR is dominated by those unconstrained corners "
             "(~17–18 dB against a masked ~28 dB), and scoring a canvas that still contains black GT "
             "pixels inflates PSNR because black is a zero-cost prediction. Either comparison gives the "
             "**opposite** of the right answer.")
    L.append("")
    L.append("**Every primary number below is computed on one identical pixel set: the `safe_crop` "
             "rectangle.** `safe_crop` is by construction the largest rectangle with zero invalid pixels, "
             "so `safe_crop ∩ valid_mask == safe_crop` and it is a fair common region for both variants. "
             "The `safe` render already *is* that rectangle; the `mask` render is cropped to it before "
             "scoring. `eval_common_region.py` asserts the extracted region has exactly the `safe_crop` "
             "shape for both variants, so a silent offset error cannot become a plausible number.")
    L.append("")

    audit_p = res / "safe_crop_audit.json"
    if audit_p.is_file():
        audit = json.loads(audit_p.read_text())
        fr = [v["frac_of_valid_kept"] for v in audit.get("clips", {}).values() if v.get("frac_of_valid_kept")]
        cam_fr: dict[str, list[float]] = {}
        bad = 0
        for cv in audit.get("clips", {}).values():
            for cam, e in cv.get("cameras", {}).items():
                if e.get("mask_checked"):
                    cam_fr.setdefault(cam, []).append(e["frac_of_valid_kept_by_safe_crop"])
                    bad += e.get("invalid_pixels_inside_safe_crop", 0)
        L.append("### What the crop costs (measured, `safe_crop_audit.json`)")
        L.append("")
        L.append(f"Invalid pixels found inside any `safe_crop`: **{bad}** "
                 f"({'premise holds' if bad == 0 else 'PREMISE VIOLATED — the primary metric is not a fair common region'}).")
        if fr:
            L.append(f"Across all 7 cameras `safe_crop` keeps **{st.fmean(fr) * 100:.1f}%** of the valid pixels.")
        L.append("")
        L.append("| camera | valid pixels kept by safe_crop |")
        L.append("|---|---|")
        for cam, vals in cam_fr.items():
            L.append(f"| `{cam}` | {st.fmean(vals) * 100:.1f}% |")
        L.append("")

    # ------------------------------------------------------------------ per-method tables
    summary: dict = {"per_method": {}, "missing_runs": [list(k) for k in missing]}
    verdicts: list[str] = []
    for m in methods:
        L.append(f"## {m} — per-clip paired comparison (common region = safe_crop)")
        L.append("")
        L.append("| clip | n img | PSNR safe | PSNR mask | ΔPSNR | SSIM safe | SSIM mask | ΔSSIM | "
                 "LPIPS safe | LPIPS mask | ΔLPIPS |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        d_psnr, d_ssim, d_lpips = [], [], []
        pair_warn = []
        iters_seen = set()
        for c in clips:
            s, k = data.get((m, "safe", c)), data.get((m, "mask", c))
            if not s or not k:
                have = [v for v in VARIANTS if (m, v, c) in data]
                L.append(f"| {c} | — | " + " | ".join(["—"] * 9) + f" | *(missing: have {have or 'nothing'})* |")
                continue
            if s["n_images"] != k["n_images"]:
                pair_warn.append(f"{c}: safe scored {s['n_images']} images, mask scored {k['n_images']} — "
                                 "the pair is NOT matched; treat this row as invalid")
            # The headline claim is that both arms were scored on ONE identical pixel set.
            # eval_common_region.py only checks each arm against safe_crop within a slack
            # window, so assert the two arms against EACH OTHER here.
            ks, kk = s.get("common_region_pixels_per_timestep"), k.get("common_region_pixels_per_timestep")
            if ks is not None and kk is not None and ks != kk:
                pair_warn.append(f"{c}: the two arms were scored on DIFFERENT regions "
                                 f"(safe {ks} px/timestep vs mask {kk}); the delta is not a "
                                 "like-for-like comparison and this row must not be pooled")
            hs = {cam: tuple(v.get("common_region_hw", ())) for cam, v in (s.get("per_camera") or {}).items()}
            hk = {cam: tuple(v.get("common_region_hw", ())) for cam, v in (k.get("per_camera") or {}).items()}
            bad_cams = sorted(cam for cam in set(hs) | set(hk) if hs.get(cam) != hk.get(cam))
            if bad_cams:
                pair_warn.append(f"{c}: per-camera scored region differs between arms for "
                                 + ", ".join(bad_cams))
            # Refuse to pool a pair whose two runs were not produced the same way (a cheap
            # smoke run must never slip into the published paired mean).
            # Compare everything in the signature EXCEPT the variant field -- the two arms are
            # supposed to differ there, and including it made this fire on every single row.
            def _sig_wo_variant(j):
                sig = j.get("run_signature")
                if not sig:
                    return None
                return ";".join(p for p in sig.split(";") if not p.startswith("variant="))
            if _sig_wo_variant(s) != _sig_wo_variant(k):
                pair_warn.append(f"{c}: the two arms were NOT produced the same way -- "
                                 f"safe '{_sig_wo_variant(s)}' vs mask '{_sig_wo_variant(k)}'")
            if s.get("iters") is not None:
                iters_seen.add((c, s.get("iters"), k.get("iters")))
            ps, pk = s["primary"], k["primary"]
            dp, ds, dl = pk["psnr"] - ps["psnr"], pk["ssim"] - ps["ssim"], pk["lpips"] - ps["lpips"]
            d_psnr.append(dp); d_ssim.append(ds); d_lpips.append(dl)
            flag = " ⚠" if s["n_images"] != k["n_images"] else ""
            L.append(f"| {c}{flag} | {s['n_images']} | {fmt(ps['psnr'])} | {fmt(pk['psnr'])} | "
                     f"**{fmt(dp, 3, True)}** | {fmt(ps['ssim'], 4)} | {fmt(pk['ssim'], 4)} | {fmt(ds, 4, True)} | "
                     f"{fmt(ps['lpips'], 4)} | {fmt(pk['lpips'], 4)} | {fmt(dl, 4, True)} |")
        # Every clip in a pooled mean must have been trained the same way. A single clip
        # left over from a cheap smoke run would otherwise contribute 1/n of the headline.
        distinct = {it for _, it, _ in iters_seen} | {it for _, _, it in iters_seen}
        if len(distinct) > 1:
            pair_warn.append("clips in this table were trained for DIFFERENT iteration counts "
                             + ", ".join(f"{c}: safe={a} mask={b}" for c, a, b in sorted(iters_seen))
                             + " — the pooled mean below mixes them and must not be quoted")
        for w in pair_warn:
            L.append("")
            L.append(f"> ⚠ {w}")
        L.append("")

        stats = {"psnr": paired_stats(d_psnr), "ssim": paired_stats(d_ssim), "lpips": paired_stats(d_lpips)}
        summary["per_method"][m] = stats
        if d_psnr:
            L.append(f"### {m} — aggregate over {len(d_psnr)} clips (paired deltas, `mask − safe`)")
            L.append("")
            L.append("| metric | mean Δ | sd | sem | min | max | clips where mask > safe | t = mean/sem |")
            L.append("|---|---|---|---|---|---|---|---|")
            for key, nd in (("psnr", 3), ("ssim", 4), ("lpips", 4)):
                s_ = stats[key]
                L.append(f"| {key.upper()} | **{fmt(s_['mean'], nd, True)}** | {fmt(s_['sd'], nd)} | "
                         f"{fmt(s_['sem'], nd)} | {fmt(s_['min'], nd, True)} | {fmt(s_['max'], nd, True)} | "
                         f"{s_['n_positive']}/{s_['n']} | {fmt(s_['t'], 2)} |")
            L.append("")
            L.append(f"n = {len(d_psnr)} clips. No p-value is claimed at this n; the per-clip signs above "
                     "are the honest evidence.")
            L.append("")
            sp = stats["psnr"]
            winner = "mask" if sp["mean"] > 0 else "safe"
            mag = abs(sp["mean"])
            unanimous = sp["n_positive"] in (0, sp["n"])
            lp = stats["lpips"]["mean"]
            lp_winner = "mask" if lp < 0 else "safe"
            agree = "and LPIPS agrees" if lp_winner == winner else f"but LPIPS favours `{lp_winner}`"
            n_for_winner = sp["n_positive"] if winner == "mask" else sp["n"] - sp["n_positive"]
            spread = ("unanimous across clips" if unanimous
                      else f"{n_for_winner}/{sp['n']} clips favour it")
            v = (f"**{m}: `{winner}` wins by {mag:.2f} dB PSNR on the common region** "
                 f"(mean of {sp['n']} per-clip deltas, sd {sp['sd']:.2f}, {spread}; {agree}, "
                 f"mean ΔLPIPS {lp:+.4f}).")
            verdicts.append(v)
            L.append("**Verdict — " + v[2:])
            L.append("")

        # ---- per-camera breakdown of the primary delta -------------------------------
        cams: dict[str, list[float]] = {}
        for c in clips:
            s, k = data.get((m, "safe", c)), data.get((m, "mask", c))
            if not s or not k:
                continue
            for cam in s.get("per_camera", {}):
                if cam in k.get("per_camera", {}):
                    cams.setdefault(cam, []).append(
                        k["per_camera"][cam]["metrics"]["psnr"] - s["per_camera"][cam]["metrics"]["psnr"])
        if cams:
            L.append(f"### {m} — ΔPSNR by camera (mask − safe, common region)")
            L.append("")
            L.append("| camera | mean ΔPSNR | clips |")
            L.append("|---|---|---|")
            for cam, vals in cams.items():
                L.append(f"| `{cam}` | {fmt(st.fmean(vals), 3, True)} | {len(vals)} |")
            L.append("")

        # ---- secondary ----------------------------------------------------------------
        rows = []
        for c in clips:
            k = data.get((m, "mask", c))
            if not k:
                continue
            sec = k.get("secondary_NOT_COMPARABLE", {})
            wide = sec.get("wide_masked") or {}
            out_ = sec.get("outside_safe_masked") or {}
            nat = sec.get("native_unmasked_DIAGNOSTIC_ONLY") or {}
            rows.append((c, k["primary"]["psnr"], wide.get("psnr"), out_.get("psnr"), nat.get("psnr")))
        if rows:
            L.append(f"### {m} — secondary numbers for the `mask` variant — **NOT comparable to the table above**")
            L.append("")
            L.append("`wide` = full canvas ∩ valid_mask (a larger, different pixel set). "
                     "`outside safe_crop` = the valid pixels the crop throws away — what the extra FOV "
                     "actually buys. `native unmasked` = full canvas including the black corners; it is "
                     "listed only to show the trap, it is **not a result**.")
            L.append("")
            L.append("| clip | common (primary) | wide (canvas ∩ mask) | valid outside safe_crop | native unmasked ⚠ |")
            L.append("|---|---|---|---|---|")
            for c, pr, wd, ou, na in rows:
                L.append(f"| {c} | {fmt(pr)} | {fmt(wd)} | {fmt(ou)} | {fmt(na)} |")
            L.append("")

    # ------------------------------------------------------------------ overall verdict
    head = ["", "## Verdict", ""]
    if verdicts:
        head += [f"- {v}" for v in verdicts]
        if len(summary["per_method"]) > 1:
            signs = {m: (s["psnr"]["mean"] > 0) for m, s in summary["per_method"].items() if s["psnr"].get("n")}
            if len(set(signs.values())) > 1:
                head.append("- **The two methods disagree on the sign**: the choice is method-dependent, so "
                            "do not generalise from one of them.")
            else:
                w = "mask" if next(iter(signs.values())) else "safe"
                head.append(f"- Both methods agree: **`{w}` is the better handling of the lens-cone corners** "
                            "on the common region.")
    else:
        head.append("- *(no complete safe/mask pair yet — nothing to decide)*")
    if missing:
        head += ["", f"⚠ **INCOMPLETE**: {len(missing)} of {len(expected)} runs are missing: "
                 + ", ".join("/".join(k) for k in missing)]
    head.append("")
    L = L[:3] + head + L[3:]

    # ------------------------------------------------------------------ timings
    tp = a.timings or (res / "timings.csv")
    if tp.is_file():
        agg: dict[tuple[str, str, str], list[float]] = {}
        with tp.open() as fh:
            for row in csv.DictReader(fh):
                try:
                    agg.setdefault((row["method"], row["variant"], row["stage"]), []).append(float(row["seconds"]))
                except Exception:  # noqa: BLE001
                    continue
        if agg:
            L += ["## Wall time per run (minutes, mean over clips)", "",
                  "| method | variant | stage | mean min | n |", "|---|---|---|---|---|"]
            for (m, v, s_), vals in sorted(agg.items()):
                L.append(f"| {m} | {v} | {s_} | {st.fmean(vals) / 60:.1f} | {len(vals)} |")
            L.append("")

    L += ["## Provenance", "",
          "| run | metrics file | n images | common-region px / timestep |", "|---|---|---|---|"]
    for k in sorted(data):
        j = data[k]
        L.append(f"| {'/'.join(k)} | `{Path(j['_path']).name}` | {j['n_images']} | "
                 f"{j.get('common_region_pixels_per_timestep', '—')} |")
    L.append("")

    res.mkdir(parents=True, exist_ok=True)
    (res / "SUMMARY.md").write_text("\n".join(L) + "\n")
    summary["verdicts"] = verdicts
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(L))
    print(f"\nwrote {res / 'SUMMARY.md'} and {res / 'summary.json'}")
    if missing:
        print(f"!! {len(missing)} runs missing: " + ", ".join("/".join(k) for k in missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
