# bench/ — snapshot of the Clariden baseline bench harness

`ncf_crop_ablation_jobs/` mirrors `/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/` (the live copy on scratch, which
is subject to the purge policy). It is the harness that trains and scores the SOTA baselines (SplatAD, NeuRAD, UniSim,
EmerNeRF) and the FoggyField V1.3 lanes on NFF-02/08/10 under ONE common evaluator (safe_crop RGB, 693 images; returns-only
LiDAR CD / CD² / depth median on 99 held-out sweeps).

Final-ablation entry points (`ncf_crop_ablation_jobs/final/`):
- `run_final.sbatch` + `node_final.sh` + `config_final.sh`: one node, 4 GPU chains from a plan file, self-chaining (CHAIN=1),
  owner lock per plan, lane locks per run dir (`../lanelock.sh`), all lanes resumable.
- `supervisor.py`: login-node supervisor (nohup). Rewrites the plans with the remaining lanes, keeps a debug chunk queued,
  submits a preemptable fallback when nothing is queued, excludes lanes after 2 real failures, and publishes
  `build_table.py` / `final_table.py` / `montage.py` / `copy_figs.sh` / `publish_metrics.sh` outputs to
  `/users/yitiwang/FoggyfieldV1/RGB-Branch-Metrics/`.
- `emer_fix_env.sh`: self-repair of the EmerNeRF venv after scratch purges (2026-11-25: omegaconf/antlr4 unpacked directly,
  nerfacc .py sources restored next to the surviving csrc.so).

Refresh the snapshot: `rsync -a --exclude __pycache__ --exclude 'core*' --exclude '*.bak*' --exclude '*.log' --exclude '.plan_*' <live jobs>/ bench/ncf_crop_ablation_jobs/`.
