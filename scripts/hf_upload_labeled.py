"""Upload the 15 best checkpoints to a private HF model repo with CLEAR
method/clip labelling, plus a MANIFEST.md mapping every file -> method, clip,
dataset, PSNR. Uploads directly from source paths (no local 18GB copy).
"""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO = "JocelynW/foggyfields-av-baseline-checkpoints"
OUT = Path("/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output")
PAN = Path("/networkhome/WMGDS/wang3_y/ETH/Code/outputs")
api = HfApi(token=os.environ["HF_TOKEN"])

# (dataset, method, clip_id, clip_desc, psnr, ckpt_path, note)
NS = lambda p: (p, p.parent.parent / "config.yml", "config.yml")          # nerfstudio: config one level above nerfstudio_models/
EM = lambda p: (p, p.parent / "config.yaml", "config.yaml")               # emernerf: config.yaml beside ckpt

ENTRIES = [
 # ---- NVIDIA_AV_Fog ----
 ("nvidia_av_fog","splatad","002","rural / medium-fog / ~113km-h","21.64",
   NS(OUT/"splatad/002/unnamed/splatad/2026-05-29_230151/nerfstudio_models/step-000029999.ckpt"),""),
 ("nvidia_av_fog","splatad","003","residential / light-fog","24.60",
   NS(OUT/"splatad/003/unnamed/splatad/2026-05-29_230151/nerfstudio_models/step-000029999.ckpt"),""),
 ("nvidia_av_fog","splatad","004","highway / heavy-fog","23.94",
   NS(OUT/"splatad/004/unnamed/splatad/2026-05-29_230151/nerfstudio_models/step-000029999.ckpt"),""),
 ("nvidia_av_fog","neurad","002","rural / medium-fog / ~113km-h","27.63",
   NS(OUT/"neurad/002/unnamed/neurad-paper/2026-05-29_002702/nerfstudio_models/step-000029999.ckpt"),""),
 ("nvidia_av_fog","neurad","003","residential / light-fog","29.10",
   NS(OUT/"neurad/003/unnamed/neurad-paper/2026-05-29_002701/nerfstudio_models/step-000029999.ckpt"),""),
 ("nvidia_av_fog","neurad","004","highway / heavy-fog","29.90",
   NS(OUT/"neurad/004/unnamed/neurad-paper/2026-05-29_002702/nerfstudio_models/step-000029999.ckpt"),""),
 ("nvidia_av_fog","emernerf","002","rural / medium-fog / ~113km-h","31.36",
   EM(OUT/"emernerf/002/foggyfields_emernerf/nvidia_fog_002_7cam/checkpoint_25000.pth"),"PRE-split-fix (stale) - re-train on new machine"),
 ("nvidia_av_fog","emernerf","003","residential / light-fog","34.11",
   EM(OUT/"emernerf/003/foggyfields_emernerf/nvidia_fog_003_7cam/checkpoint_25000.pth"),"PRE-split-fix (stale) - re-train on new machine"),
 ("nvidia_av_fog","emernerf","004","highway / heavy-fog","34.66",
   EM(OUT/"emernerf/004/foggyfields_emernerf/nvidia_fog_004_7cam/checkpoint_25000.pth"),"PRE-split-fix (stale) - re-train on new machine"),
 # ---- PandaSet (clear-weather reference) ----
 ("pandaset","splatad","011","clear weather","27.52",
   NS(PAN/"paper_011_split05_paperfaithful/unnamed/splatad/2026-05-18_113557/nerfstudio_models/step-000030000.ckpt"),""),
 ("pandaset","splatad","078","clear weather","31.65",
   NS(PAN/"paper_078_split05_paperfaithful/unnamed/splatad/2026-05-18_124006/nerfstudio_models/step-000030000.ckpt"),""),
 ("pandaset","neurad","011","clear weather","26.46",
   NS(PAN/"neurad_paper_011_split05/unnamed/neurad-paper/2026-05-18_174055/nerfstudio_models/step-000029999.ckpt"),""),
 ("pandaset","neurad","078","clear weather","30.04",
   NS(PAN/"neurad_paper_078_split05/unnamed/neurad-paper/2026-05-18_174055/nerfstudio_models/step-000029999.ckpt"),""),
 ("pandaset","emernerf","011","clear weather","27.55",
   EM(PAN/"emernerf_011_split05/foggyfields_emernerf/scene_011_6cam_dyn_flow_rgb/checkpoint_30000.pth"),""),
 ("pandaset","emernerf","078","clear weather","31.26",
   EM(PAN/"emernerf_078_split05/foggyfields_emernerf/scene_078_6cam_dyn_flow_rgb/checkpoint_30000.pth"),""),
]

# pre-flight: verify every source file exists
missing = []
for ds,m,cid,desc,psnr,(ck,cfg,cfgname),note in ENTRIES:
    if not ck.exists(): missing.append(str(ck))
    if not cfg.exists(): missing.append(str(cfg))
if missing:
    print("ABORT — missing source files:"); [print("  ", x) for x in missing]; sys.exit(1)
print(f"all {len(ENTRIES)} checkpoints + configs found.")

create_repo(REPO, repo_type="model", private=True, exist_ok=True, token=os.environ["HF_TOKEN"])
print(f"repo ready (private): https://huggingface.co/{REPO}")

# ---- build MANIFEST.md + README.md ----
rows = ["| Dataset | Method | Clip | Scene | PSNR↑ | Path in repo | Note |",
        "|---|---|---|---|---|---|---|"]
def repo_dir(ds,m,cid,desc):
    safe = desc.replace(" ","").replace("/","-")
    return f"{ds}/{m}/clip{cid}_{safe}" if ds=="nvidia_av_fog" else f"{ds}/{m}/clip{cid}"
for ds,m,cid,desc,psnr,(ck,cfg,cfgname),note in ENTRIES:
    rd = repo_dir(ds,m,cid,desc)
    rows.append(f"| {ds} | **{m}** | **{cid}** | {desc} | {psnr} | `{rd}/{ck.name}` | {note} |")
manifest = ("# Checkpoint manifest — which file is which method × clip\n\n"
 "Protocol: nvs_50_50 (50/50 held-out novel view). PSNR = held-out RGB.\n"
 "Each nerfstudio `.ckpt` needs its sibling `config.yml`; each EmerNeRF `.pth` needs `config.yaml`.\n\n"
 + "\n".join(rows) + "\n\n"
 "## How to load\n"
 "- SplatAD/NeuRAD: `ns-render dataset --load-config <dir>/config.yml ...` or `ns-eval --load-config <dir>/config.yml`\n"
 "- EmerNeRF: load the `.pth` with the sibling `config.yaml` via `train_emernerf.py` (resume/eval)\n\n"
 "## Clips\n"
 "- NVIDIA_AV_Fog **002** rural, medium fog, ~113 km/h · **003** residential, light fog · **004** highway, heavy fog\n"
 "- PandaSet **011 / 078** clear-weather reference runs\n\n"
 "⚠️ EmerNeRF NVIDIA checkpoints are PRE-split-fix (held out the complementary frames); re-train with the fixed loader.\n")
(Path("/tmp/MANIFEST.md")).write_text(manifest)
api.upload_file(path_or_fileobj="/tmp/MANIFEST.md", path_in_repo="README.md", repo_id=REPO, repo_type="model")
print("uploaded README.md/MANIFEST")

# ---- upload each ckpt + config into its labelled dir ----
for i,(ds,m,cid,desc,psnr,(ck,cfg,cfgname),note) in enumerate(ENTRIES,1):
    rd = repo_dir(ds,m,cid,desc)
    print(f"[{i}/{len(ENTRIES)}] {rd}  ({ck.stat().st_size/1e9:.2f} GB) ...", flush=True)
    api.upload_file(path_or_fileobj=str(ck), path_in_repo=f"{rd}/{ck.name}", repo_id=REPO, repo_type="model")
    api.upload_file(path_or_fileobj=str(cfg), path_in_repo=f"{rd}/{cfgname}", repo_id=REPO, repo_type="model")
    print(f"    done {rd}", flush=True)

print("ALL DONE:", f"https://huggingface.co/{REPO}")
