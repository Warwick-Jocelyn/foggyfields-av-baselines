#!/bin/bash
# Copy the comparison-figure PNGs (held-out timesteps 33/99/165, front_wide + front_tele, render + GT, one LiDAR BEV)
# of every finished baseline / V1.3 lane into HOME (PNG only):  copy_figs.sh
BW=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation; DST=/users/yitiwang/FoggyfieldV1/RGB-LiDAR-Viz/baselines
cams="camera_front_wide_120fov camera_front_tele_30fov"; frames="0033 0099 0165"
copy_one(){  # <renders/val dir> <viz lidar dir> <clip> <method label>
  local r=$1 lz=$2 c=$3 m=$4 d=$DST/$c/$m n=0
  [ -d "$r/rgb" ] || return 0
  mkdir -p "$d"
  for cam in $cams; do for f in $frames; do
    [ -f "$r/rgb/$cam/frame_$f.png" ] && { cp -u "$r/rgb/$cam/frame_$f.png" "$d/${cam#camera_}__frame_$f__render.png"; n=$((n+1)); }
    [ -f "$r/gt-rgb/$cam/frame_$f.png" ] && cp -u "$r/gt-rgb/$cam/frame_$f.png" "$DST/$c/GT/${cam#camera_}__frame_$f__gt.png" 2>/dev/null || { mkdir -p "$DST/$c/GT"; cp -u "$r/gt-rgb/$cam/frame_$f.png" "$DST/$c/GT/${cam#camera_}__frame_$f__gt.png" 2>/dev/null; }
  done; done
  [ -f "$lz/bev.png" ] && cp -u "$lz/bev.png" "$d/lidar_bev.png"
  [ -f "$lz/range.png" ] && cp -u "$lz/range.png" "$d/lidar_range.png"
  echo "$c/$m: $n renders"
}
for c in NFF-02 NFF-08 NFF-10; do
  copy_one $BW/outputs_final/allfix/splatad/safe/$c/renders/val $BW/viz_final/allfix/splatad/safe/$c/lidar $c splatad_allfix
  copy_one $BW/outputs_final/opreg5e4/splatad/safe/$c/renders/val $BW/viz_final/opreg5e4/splatad/safe/$c/lidar $c splatad_opreg5e4
  copy_one $BW/outputs_final/std/neurad/safe/$c/renders/val $BW/viz_final/std/neurad/safe/$c/lidar $c neurad_std
  copy_one $BW/outputs_final/std/emernerf/safe/$c/renders/val $BW/viz_final/std/emernerf/safe/$c/lidar $c emernerf
  copy_one $BW/outputs_final/std/unisim/safe/$c/renders/val $BW/viz_final/std/unisim/safe/$c/lidar $c unisim
  for d in $BW/outputs_final/v13/*__$c; do [ -d "$d" ] || continue; l=$(basename "$d"); copy_one $d/renders/val $BW/viz_final/v13/$l/lidar $c "v13_${l%%__*}"; done
done
du -sh $DST 2>/dev/null
