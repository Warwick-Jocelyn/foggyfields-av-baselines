"""Rewrite a published nerfstudio config.yml so its checkpoint can be evaluated on this machine.
Sets output_dir/data paths, disables wandb, and symlinks the .ckpt where eval_setup will look for it."""
import argparse, shutil, yaml
from pathlib import Path
from nerfstudio.engine.trainer import TrainerConfig  # noqa: needed so yaml can rebuild the objects
ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, required=True, help="dir with config.yml + step-*.ckpt from HF")
ap.add_argument("--dst", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
a = ap.parse_args()
cfg = yaml.load((a.src / "config.yml").read_text(), Loader=yaml.Loader)
assert isinstance(cfg, TrainerConfig)
cfg.output_dir = a.dst.resolve(); cfg.data = None; cfg.vis = "tensorboard"
cfg.load_dir = None; cfg.load_checkpoint = None
cfg.pipeline.datamanager.data = None
cfg.pipeline.datamanager.dataparser.data = a.data.resolve()
ckpt_dir = cfg.get_checkpoint_dir(); ckpt_dir.mkdir(parents=True, exist_ok=True)
for ck in a.src.glob("step-*.ckpt"):
    dst = ckpt_dir / ck.name
    if not dst.exists(): dst.symlink_to(ck.resolve())
out = a.dst / "config.yml"; out.write_text(yaml.dump(cfg))
print("wrote", out, "| ckpt dir", ckpt_dir, "| sequence", cfg.pipeline.datamanager.dataparser.sequence,
      "| splits", cfg.pipeline.datamanager.dataparser.splits_json)
