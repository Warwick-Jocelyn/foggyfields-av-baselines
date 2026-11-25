# Corrected bench (2026-10-25): the ONE place that says how each method is run after the audit
# (docs/BUG_LEDGER.md). Sourced by node.sh; exports the *_EXTRA / *_DP_EXTRA that lane.sh reads.
#
# Shared dataparser fixes (neurad-studio ba964dda):
#   --camera-pose-time camera   F1b: camera pose at the camera's own exposure time, not the LiDAR spin centre
#   --actor-box-axes pandaset   F2b: actor boxes were rotated 90 deg (local x = heading vs dims (w,l,h))
FIX_DP="--actor-box-axes pandaset --camera-pose-time camera"

# NeuRAD: code verified correct apart from the above; temporal appearance off as in the bench (= neurad-paper).
# Its LR schedulers ship with max_steps 20001; we train 30000 -> stretch every group to the budget (N2).
NEURAD_SCHED=""
for _grp in camera_opt cnn fields hashgrids trajectory_opt; do   # NB: not "g" -- node.sh sources this inside a function
  NEURAD_SCHED="$NEURAD_SCHED --optimizers.$_grp.scheduler.max-steps 30000"
done; unset _grp
export NEURAD_EXTRA="--pipeline.model.use-temporal-appearance False$NEURAD_SCHED ${NEURAD_FLIP:-}"
export NEURAD_DP_EXTRA="--add-missing-points True $FIX_DP"

# UniSim: plugin fixes U1-U5 are its defaults since unisim 39ec638 (trajectory opt, sky remainder,
# per-sensor appearance, CNN trained from step 0, symmetric occupancy dilation). U6/U6b (f9e0217: lidar
# intensity decoder in fp32 with biases + LayerNorm input) and U5b (945391b: point-space occupancy dilation,
# +-0.25 m) are pinned explicitly so the run signature changes with them. No ray drop (paper).
export UNISIM_EXTRA="--pipeline.model.lidar-decoder-fp32 True --pipeline.model.lidar-decoder-layernorm True --pipeline.model.sampler.occupancy-point-dilation True"
export UNISIM_DP_EXTRA="--use-range-image-raydrop False $FIX_DP"
