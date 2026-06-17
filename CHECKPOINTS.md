# CHECKPOINTS — HuggingFace manifest

Large artifacts live on HuggingFace, **not** in git. Two repos:

| HF repo | type | size | contents |
|---|---|---|---|
| `<user>/nvidia-av-fog-processed` | dataset (**gated**) | ~4.4 GB | processed clips 002/003/004 (`processed_pinhole/` only) + dataset card |
| `<user>/nvidia-av-fog-baseline-checkpoints` | model | ~18 GB | the 15 best checkpoints below + sibling `config.yml` + showcase videos |

> **Dataset license:** the processed clips derive from **NVIDIA PhysicalAI-AV / NCore**.
> Verify NVIDIA's redistribution terms before uploading. If derivatives may not be
> redistributed, upload **only** the dataset card + conversion script and have users
> regenerate from upstream. Gate the dataset repo (access-request) regardless. The `raw/`
> camera images are NVIDIA's source — never upload them; ship only `processed_pinhole/`.

## Model repo layout
```
nvidia-av-fog-baseline-checkpoints/
├── README.md                          # model card (this table + how to load)
├── nvidia/
│   ├── splatad/{002,003,004}/{step-000029999.ckpt, config.yml}
│   ├── neurad/{002,003,004}/{step-000029999.ckpt, config.yml}
│   └── emernerf/{002,003,004}/{checkpoint_25000.pth, config.yaml}   # ⚠ pre-split-fix, see RESULTS †
├── pandaset/
│   ├── splatad/{011,078}/{step-000030000.ckpt, config.yml}
│   ├── neurad/{011,078}/{step-000029999.ckpt, config.yml}
│   └── emernerf/{011,078}/{checkpoint_30000.pth, config.yaml}
└── videos/                            # a few 7-cam panorama_rgb.mp4 showcase renders
```

## Checkpoint sizes (the keep-set)
| group | files | each | subtotal |
|---|---|---|---|
| NVIDIA SplatAD | 3 | ~1.58 GB | 4.7 GB |
| NVIDIA NeuRAD | 3 | ~1.41 GB | 4.2 GB |
| NVIDIA EmerNeRF (stale) | 3 | ~0.70 GB | 2.1 GB |
| PandaSet SplatAD | 2 | ~1.65 GB | 3.3 GB |
| PandaSet NeuRAD | 2 | ~1.41 GB | 2.8 GB |
| PandaSet EmerNeRF | 2 | ~0.70 GB | 1.4 GB |
| **total (15 ckpts + configs + videos)** | | | **~18.5 GB** |

Exclude the stale NVIDIA EmerNeRF set on first upload → ~16.4 GB.

## How to load
- **SplatAD / NeuRAD (nerfstudio):** the `.ckpt` is useless without its sibling `config.yml`.
  ```bash
  ns-render dataset --load-config <dir>/config.yml --output-path out --rendered-output-names rgb --pose-source val
  ns-eval --load-config <dir>/config.yml --output-path eval.json
  ```
- **EmerNeRF:** load `checkpoint_*.pth` with the matching `config.yaml` via `train_emernerf.py`
  in resume/eval mode.

## Staging + upload
Run `scripts/stage_checkpoints.sh` (gathers the exact files above into an upload tree with
existence checks) then `scripts/upload_hf.sh` (after `hf auth login`). Both print what they
do before doing it.

The exact source paths on the origin box are listed in `RESULTS.md` §4.
