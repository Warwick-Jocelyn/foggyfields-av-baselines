"""FoggyFields package. Torch-dependent modules load lazily so that numpy-only consumers
(fogsim_adapter on the login node) can import without torch."""
_TORCH_EXPORTS = {
    'HomogeneousFog': 'medium', 'MediumIntegrator': 'medium', 'MediumLUT': 'medium',
    'expected_depth_overlay': 'transport', 'ordered_airlight': 'transport',
    'attenuate_surface_echo': 'transport',
    'FogReturnMixture': 'mixture', 'MixtureConfig': 'mixture',
}


def __getattr__(name):
    if name in _TORCH_EXPORTS:
        import importlib
        mod = importlib.import_module(f'.{_TORCH_EXPORTS[name]}', __name__)
        return getattr(mod, name)
    raise AttributeError(name)
