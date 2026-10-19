#!/usr/bin/env bash
# Two deliberately distinguishable training regimes for the NFF / NCF dataparser.
#
#   source jobs/regimes.sh; set_regime v2fixed        # or: legacy_prefix
#   ns-train ... nvidia-av-fog-data --sequence "$SEQ" $DP_FLAGS
#   OUT_ROOT=".../$REGIME_DIR/$SEQ"
#
# The point is that a run can never be ambiguous after the fact: the flags and the
# output directory name both say which regime produced it.
#
#   v2fixed        the corrected defaults (F1-F6 + rectification v2). This is what the
#                  dataparser does with no flags at all; they are written out anyway so
#                  the run log records them.
#   legacy_prefix  reproduces the SUPERSEDED pre-fix behaviour on purpose: LiDAR
#                  timestamps on every camera, global shutter, "ego" LiDAR frame on
#                  sensor-frame clips, no deformable actors, v1 rectification, no masks.
#                  Only for re-creating the old numbers. Do not report it as a result.

set_regime() {
  case "${1:?usage: set_regime v2fixed|legacy_prefix}" in
    v2fixed)
      REGIME="v2fixed"
      REGIME_DIR="v2fixed"
      DP_FLAGS=(
        --camera-timestamp-source auto      # real per-camera timestamps          (F1)
        --rolling-shutter-source calib      # published 30.559 ms readout         (F2)
        --lidar-points-frame auto           # provenance.lidar_frame              (F3)
        --include-deformable-actors True    # person / rider / animal             (F4)
        --unknown-class-policy error        # never drop a class silently         (F4)
        --use-published-beam-geometry True  # 128 real beams, not Pandar64's 64   (F6)
        --use-range-image-raydrop True      # real non-returns when present       (F6)
        --rectification auto                # v2 where the clip has it            (V2)
        --apply-safe-crop True              # no black corners in the loss        (V2)
        --use-valid-masks True              # non-rectangular valid region        (V2)
        --rs-time-offset-lookup True        # per-pixel readout time              (V2)
      )
      ;;
    legacy_prefix)
      REGIME="legacy_prefix"
      REGIME_DIR="legacy_prefix_SUPERSEDED"
      DP_FLAGS=(
        --camera-timestamp-source lidar
        --rolling-shutter-source config --rolling-shutter-time 0.0
        --lidar-points-frame ego
        --include-deformable-actors False
        --unknown-class-policy drop
        --use-published-beam-geometry False
        --use-range-image-raydrop False
        --rectification v1
        --apply-safe-crop False
        --use-valid-masks False
        --rs-time-offset-lookup False
      )
      ;;
    *) echo "unknown regime: $1" >&2; return 1 ;;
  esac
  export REGIME REGIME_DIR
  echo "[regimes.sh] regime=$REGIME output subdir=$REGIME_DIR"
  printf '[regimes.sh] dataparser flags: %s\n' "${DP_FLAGS[*]}"
}

# Memory guidance for rectification v2 (up to 2692x1880, 4.1x v1's pixels). The
# dataparser logs the projected cache size and warns past --image-cache-warn-gb.
#   --pipeline.datamanager.cache-images-type uint8          4x less than float32
#   --pipeline.datamanager.cache-images cpu                 keeps it off the GPU
#   --pipeline.datamanager.camera-res-scale-factor 0.5      4x less again
