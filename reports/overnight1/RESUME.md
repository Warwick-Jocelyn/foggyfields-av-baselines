# RESUME — how any session picks this project up

Project: FoggyFields (explicit world-space fog medium; NeuRAD backbone for physics evidence).
This workspace: /capstor/scratch/cscs/yitiwang/foggyfields  (renamed from foggyfields-baseline 2026-10-26).

READ FIRST, in order:
1. /users/yitiwang/store-shared/yitiwang/FoggyFields_Claude_Experiment_Plan_v2.md  (authoritative plan + my addendum at the end)
2. /users/yitiwang/store-shared/yitiwang/FoggyFields_Overnight_Task1.md            (tonight's scope + status addendum)
3. reports/overnight1/REPORT.md                                                     (what was done, with evidence)
4. Claude memory: foggyfields-overnight1.md (+ ncf-nff-crop-ablation-project.md)

STATE (2026-10-26 ~03:00):
- Tasks 0-3 done, reported, pushed to github Warwick-Jocelyn/foggyfields-av-baselines (d32b221):
  src/foggyfields/ (MediumIntegrator + transport, 8/8 tests), reports/overnight1/.
- Task 4 B1-lite: slurm job 3750288 (debug) fits (beta0, A, g) on frozen NeuRAD ckpts,
  NFF-15 (primary fog) + NFF-13 (spray contrast). Outputs -> b1lite/<clip>/results.json
  + logs/b1lite_<clip>.log. If the job died: resubmit `sbatch -p debug -t 01:00:00 jobs/b1lite.sbatch`
  (idempotent; paths inside reference THIS workspace).
AFTER the job: parse b1lite/*/results.json -> append Task 4 block to reports/overnight1/REPORT.md
  (template: plan §10.3; gate = does attenuation-only change anything + beta within x2 of anchor
  0.0252 [NFF-15]; range tail is structurally unchanged - say so) -> write the preliminary report
  to /users/yitiwang/store-shared/yitiwang/FoggyFields_Overnight1_Report.md -> copy b1lite results
  + qual PNGs into the repo reports/overnight1/B1lite/ -> commit + push.
STANDING RULES: user authorised full autonomy (no permission questions); small-scale GPU only
  (debug partition); never rerun finished baselines; never touch slurm job 3734009; deliverables
  as editable .md files in store-shared.
NEXT MILESTONES (plan §6): P1 analytic plane-scene tests + per-sample RGB head (attach
  neurad_field.py:141-144); P2 synthetic fog (Hahner + LISA) on NCF-02; CVPR go/no-go 11-09.
