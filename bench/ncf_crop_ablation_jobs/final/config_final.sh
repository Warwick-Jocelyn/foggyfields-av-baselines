# Final-ablation baselines (decision 109, 2026-11-23): the ONE place that says how each external method runs.
# Sourced by node_final.sh. Code = worktree code/neurad-studio-bench (a2f5084f = corrected bench of 10-25 +
# the 11-18 beam-classification floor 0.55 needed by NFF-08); venv = foggyfields-v1/venv_neurad (the
# October recipe rebuilt after the purge). Data = data_final/clips -> real NFF-02/08/10 dirs.
export BW=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
export REPRO=/capstor/scratch/cscs/yitiwang/splatad-nff-repro
export VENV=/capstor/scratch/cscs/yitiwang/foggyfields-v1/venv_neurad
export NS_REPO=$REPRO/code/neurad-studio-bench
export HARNESS_PY=$VENV/bin/python
export DATA_ROOT=$BW/data_final
export SPLIT=nvs_50_50
export SAVE_EVERY=2000
export VIS=tensorboard
FIX_DP="--actor-box-axes pandaset --camera-pose-time camera"      # F1b + F2b (defaults at a2f5084f, passed explicitly)
# SplatAD, fog-fixed ("allfix" of viz_nff/SPLATAD_FOG_EXPLAINED.md): mcmc opacity reg 0.005 -> 0.0005 + F1b/F2b.
# lane.sh adds: --strategy mcmc --mcmc-cap-max 5000000 --max-num-seed-points 2000000. 30000 iterations.
SPLATAD_ALLFIX_EXTRA="--pipeline.model.mcmc-opacity-reg-lambda 0.0005"
SPLATAD_ALLFIX_DP="$FIX_DP"
# Fallback variant "opreg5e4" = the variant that finished on NFF-13/NFF-15 (allfix crashed on NFF-15 at 14.5k):
# same opacity reg, PRE-fix data path (lidar-time camera poses, legacy box axes).
SPLATAD_OPREG_EXTRA="--pipeline.model.mcmc-opacity-reg-lambda 0.0005"
SPLATAD_OPREG_DP="--actor-box-axes legacy --camera-pose-time lidar"
# NeuRAD standard = jobs/fixbench/config.sh (temporal appearance off = neurad-paper, schedulers stretched to 30k).
NEURAD_SCHED=""
for _grp in camera_opt cnn fields hashgrids trajectory_opt; do NEURAD_SCHED="$NEURAD_SCHED --optimizers.$_grp.scheduler.max-steps 30000"; done; unset _grp
export NEURAD_EXTRA="--pipeline.model.use-temporal-appearance False$NEURAD_SCHED"
export NEURAD_DP_EXTRA="--add-missing-points True $FIX_DP"
# EmerNeRF: jobs/emernerf (configs/nvidia_av_fog_safecrop.yaml, 25000 iterations, own venv + venv_neurad for tcnn).
export EMER_CONFIG=configs/nvidia_av_fog_safecrop.yaml
# V1.3 adapter
export V13UNI=$REPRO/code/neurad-studio-v13-uni
export V13MV=$REPRO/code/neurad-studio-v13-mv
export V13_QUEUE=$BW/jobs/final/v13_queue.txt
export LANES_MD=/users/yitiwang/projects/foggyfields-v1/reports/v13_overnight/final_ablation_2026-11-23/RESULTS_LANES.md
# UniSim (plugin code/_src/unisim @ 86dcdee, as in fixbench/config.sh): paper config, no ray drop, U5b/U6/U6b fixes pinned.
# The plugin is registered in venv_neurad by jobs/final/unisim_install.sh (editable, --no-deps) before the first UniSim lane.
# 2026-11-25 FAIR BUDGET (user: same steps for every method): every method trains 30 000 steps. UniSim's paper schedule
# (stages 2500+7500+10001, schedulers 20001) is stretched x1.5 exactly as NeuRAD's was stretched to 30k.
UNISIM_SCHED="--pipeline.steps-per-stage 3750 11250 15001"
for _grp in trajectory_opt discriminator cnn fields camera_opt; do UNISIM_SCHED="$UNISIM_SCHED --optimizers.$_grp.scheduler.max-steps 30000"; done; unset _grp
export UNISIM_EXTRA="--pipeline.model.lidar-decoder-fp32 True --pipeline.model.lidar-decoder-layernorm True --pipeline.model.sampler.occupancy-point-dilation True $UNISIM_SCHED"
export UNISIM_DP_EXTRA="--use-range-image-raydrop False $FIX_DP"
export UNISIM_ITERS=30000
# EmerNeRF: 30 000 steps (was its default 25 000); its LR schedule follows optim.num_iters. node_final passes ITERS=$EMER_ITERS.
export EMER_ITERS=30000
