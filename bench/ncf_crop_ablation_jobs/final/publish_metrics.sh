#!/bin/bash
# Copy the final-ablation baseline results to the user's deliverable folder (JSON + timings + report + one-line provenance per row).
BW=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation; DST=/users/yitiwang/FoggyfieldV1/RGB-Branch-Metrics/baselines
REP=/users/yitiwang/projects/foggyfields-v1/reports/v13_overnight/final_ablation_2026-11-23/RESULTS_BASELINES.md
mkdir -p "$DST"; cp -u "$REP" "$DST/RESULTS_BASELINES.md"; : > "$DST/provenance.txt"
for tag in allfix opreg5e4 std v13; do
  for kind in metrics lidar; do
    for f in "$BW"/results_final/$tag/$kind/*.json; do
      [ -f "$f" ] || continue
      case "$tag" in allfix) m=splatad_allfix;; opreg5e4) m=splatad_opreg5e4;; std) m=$(basename "$f" | sed -E 's/^(metrics|lidar)_([a-z]+).*/\2/');; v13) m=$(basename "$f" | sed -E 's/^(metrics|lidar)_v13__([^_]+(_[^_]+)*)__NFF.*/v13_\2/');; esac
      c=$(basename "$f" | grep -oE 'NFF-[0-9]{2}'); mkdir -p "$DST/$m/$c"; cp -u "$f" "$DST/$m/$c/"
      /users/yitiwang/miniforge3/bin/python3 -I - "$f" "$m" "$c" "$kind" >> "$DST/provenance.txt" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); m,c,k=sys.argv[2:5]
if k=="metrics":
    pr=d.get("primary",{}); print(f"{m} {c} RGB: PSNR {pr.get('psnr',float('nan')):.3f} SSIM {pr.get('ssim',float('nan')):.4f} LPIPS {pr.get('lpips',float('nan')):.4f} n={d.get('n_images')} iters={d.get('iters')} sig={d.get('run_signature','')} src={sys.argv[1]}")
else:
    print(f"{m} {c} LiDAR: cd_returns {d.get('cd_returns_m')} cd2_returns {d.get('cd2_returns_m2')} depth_med {d.get('depth_median_l2_returns_m')} n_gt={d.get('n_gt')} rays={d.get('eval_ray_set')} cfg={d.get('config', d.get('dump',''))} src={sys.argv[1]}")
PY
    done
  done
  for t in "$BW"/results_final/$tag/timings*.csv; do [ -f "$t" ] && cp -u "$t" "$DST/timings_${tag}_$(basename "$t")"; done
done
echo "published -> $DST ($(wc -l < "$DST/provenance.txt") rows)"
