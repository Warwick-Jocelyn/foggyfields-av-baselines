# Storage & durability strategy (CSCS Alps / Clariden)

> **Why this exists:** on 2026-07 the whole project was silently deleted from
> `/iopsstor/scratch`. Root cause below. The rule now: **scratch is disposable;
> every irreplaceable byte has a durable home.**

## The purge rules (verified against CSCS docs, 2026-07)

Both scratch tiers delete files on **last-access time (atime)** — reading OR writing
resets the clock. No backups on either.

| Filesystem | Deleted after no access for | Backed up? | Use for |
|---|---|---|---|
| `/iopsstor/scratch/cscs/$USER` (fast NVMe) | **14 days** | ❌ | transient training I/O only |
| `/capstor/scratch/cscs/$USER` | **30 days** | ❌ | medium-term staging |
| `/capstor/store/cscs/swissai/a0195` | **never** (no cleanup) | ✅ daily tape, 3 copies; kept 3 mo after project ends | durable on-cluster home |
| `$HOME` `/users/$USER` | never | ✅ 7-day snapshots in `$HOME/.snapshot` | small configs/dotfiles |

`.uenv-images` is the only scratch dir exempt from cleanup.
Docs: https://docs.cscs.ch/storage/filesystems/

**What bit us:** the project lived on `/iopsstor/scratch` (14-day window). Code went
unread >14 days → purged, including all `.git` objects → no on-cluster git recovery.

## The 3-tier source-of-truth model

1. **Off-cluster truth (primary):**
   - Code → GitHub `Warwick-Jocelyn/foggyfields-av-baselines` (this repo).
   - Dataset → HuggingFace `JocelynW/NFF` (dataset, gated — needs your HF token).
   - Checkpoints → HuggingFace `<user>/nvidia-av-fog-baseline-checkpoints` (see `scripts/upload_hf.sh`).
2. **On-cluster durable (`/capstor/store/.../swissai/a0195/yitiwang/`):** mirror of the
   code repo, `configs_backup/`, `containers/` (14G images), `results/`, and — the gap that
   caused data loss — **final checkpoints** (`checkpoints_final/`).
3. **Working tier (`/iopsstor/scratch/.../projects/foggyfields`):** transient only.
   Fully rebuildable from tiers 1–2 by `scripts/infra/bootstrap.sh` in minutes (+ dataset download time).

## Where things are (2026-07 recovery snapshot)

| Asset | Status | Source of truth |
|---|---|---|
| Code / harness / jobs | ✅ recovered | GitHub `Warwick-Jocelyn/foggyfields-av-baselines` |
| Upstreams (neurad-studio, splatad, EmerNeRF) | re-clonable | pins in `BASE_COMMITS.txt` |
| Dataset (NFF clips) | ✅ safe | HF `JocelynW/NFF` (24,855 clip files) |
| Results / metrics / videos | ✅ safe | `/capstor/store/.../foggyfields/results/` |
| Containers (14G) | ✅ safe | `/capstor/store/.../containers/` |
| **Trained checkpoints** | ⚠️ **lost** (never mirrored) | must retrain, then `sync_to_store.sh` keeps them |

## Operating rules

- **After every run that produces something you'd hate to lose:** `bash scripts/infra/sync_to_store.sh`.
- **Add it as a daily cron** (belt-and-suspenders): see the header of `sync_to_store.sh`.
- **Never** point `--output-dir` at a path you don't also sync to store.
- To rebuild a purged scratch from nothing: `bash scripts/infra/bootstrap.sh <workdir>`.
