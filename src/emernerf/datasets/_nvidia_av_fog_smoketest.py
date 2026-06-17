"""Smoke-test launcher for NVIDIAAVFogDataset.

Stubs out `datasets` package __init__.py to avoid pulling in nuscenes →
radiance_fields → tcnn (which require CUDA). We register a synthetic
`datasets` module BEFORE any real import resolves.
"""
import os, sys, types, importlib

REPO = "/networkhome/WMGDS/wang3_y/ETH/Code/EmerNeRF"
sys.path.insert(0, REPO)

# Pre-create datasets and datasets.base namespace packages with empty __init__
def _ns_pkg(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__file__ = os.path.join(path, "__init__.py")
    sys.modules[name] = mod
    return mod

_ns_pkg("datasets", os.path.join(REPO, "datasets"))
_ns_pkg("datasets.base", os.path.join(REPO, "datasets", "base"))

# Now import each base submodule explicitly so datasets.base.X resolves.
for sub in ("lidar_source", "pixel_source", "scene_dataset", "split_wrapper"):
    importlib.import_module(f"datasets.base.{sub}")

# Now safely import the actual file
spec = importlib.util.spec_from_file_location(
    "datasets.nvidia_av_fog",
    os.path.join(REPO, "datasets", "nvidia_av_fog.py"),
)
nv = importlib.util.module_from_spec(spec)
sys.modules["datasets.nvidia_av_fog"] = nv
spec.loader.exec_module(nv)

# --- Build minimal config and run ---
from omegaconf import OmegaConf
import logging
logging.basicConfig(level=logging.INFO)

cfg = OmegaConf.create({
    "data_root": "/networkhome/WMGDS/wang3_y/ETH/Dataset/NVIDIA_AV_Fog",
    "dataset": "nvidia_av_fog",
    "scene_idx": "002",
    "start_timestep": 0,
    "end_timestep": 4,
    "ray_batch_size": 4096,
    "preload_device": "cpu",
    "pixel_source": {
        "load_size": [960, 1280],
        "downscale": 1,
        "num_cams": 7,
        "test_image_stride": 2,
        "load_rgb": True,
        "load_sky_mask": False,
        "load_dynamic_mask": False,
        "load_features": False,
        "skip_feature_extraction": True,
        "target_feature_dim": 64,
        "feature_model_type": "dinov2_vitb14",
        "feature_extraction_stride": 7,
        "feature_extraction_size": [644, 966],
        "delete_features_after_run": False,
        "sampler": {"buffer_downscale": 16, "buffer_ratio": 0.0},
    },
    "lidar_source": {
        "load_lidar": True,
        "only_use_top_lidar": False,
        "truncated_max_range": 80,
        "truncated_min_range": -2,
        "lidar_downsample_factor": 4,
        "lidar_percentile": 0.02,
    },
})

ds = nv.NVIDIAAVFogDataset(data_cfg=cfg)
print("====== SMOKE TEST RESULTS ======")
print(f"num_cams                  = {ds.num_cams}")
print(f"num_img_timesteps         = {ds.num_img_timesteps}")
print(f"num_train_timesteps       = {ds.num_train_timesteps}")
print(f"num_test_timesteps        = {ds.num_test_timesteps}")
print(f"pixel_source.num_imgs     = {ds.pixel_source.num_imgs}")
print(f"pixel_source.images shape = {tuple(ds.pixel_source.images.shape)}")
print(f"sample image[0] shape     = {tuple(ds.pixel_source.images[0].shape)}")
print(f"intrinsics[0]             =\n{ds.pixel_source.intrinsics[0]}")
print(f"cam_to_worlds[0]          =\n{ds.pixel_source.cam_to_worlds[0]}")
print(f"AABB                      = {ds.aabb.tolist()}")
if ds.lidar_source is not None:
    print(f"lidar.origins shape       = {tuple(ds.lidar_source.origins.shape)}")
    print(f"lidar.directions shape    = {tuple(ds.lidar_source.directions.shape)}")
    print(f"lidar.ranges shape        = {tuple(ds.lidar_source.ranges.shape)}")
    print(f"lidar.num_timesteps       = {ds.lidar_source.num_timesteps}")
    pts = ds.lidar_source.origins[:1] + ds.lidar_source.directions[:1] * ds.lidar_source.ranges[:1]
    print(f"sample world pt           = {pts.numpy().tolist()}")
print("================================")
