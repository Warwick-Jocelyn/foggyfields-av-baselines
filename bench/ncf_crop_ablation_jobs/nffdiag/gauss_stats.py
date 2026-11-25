#!/usr/bin/env python3
"""Gaussian population stats of SplatAD checkpoints (CPU): count, opacity and scale quantiles.
    python gauss_stats.py label=path.ckpt ...
"""
import sys, torch
def q(x, ps=(0.1, 0.5, 0.9, 0.99)):
    x = x.flatten().float()
    if x.numel() > 2_000_000:
        x = x[torch.randint(0, x.numel(), (2_000_000,))]
    return " ".join(f"{v:.3g}" for v in torch.quantile(x, torch.tensor(ps)).tolist())
print("label | N | opacity q10/50/90/99 | frac op<0.005 | frac op>0.5 | max-scale(m) q10/50/90/99 | frac maxscale>1m | >5m | mean scale")
for arg in sys.argv[1:]:
    lab, p = arg.split("=", 1)
    sd = torch.load(p, map_location="cpu", weights_only=False)["pipeline"]
    k = lambda n: next(v for kk, v in sd.items() if kk.endswith("gauss_params." + n) or kk.endswith("." + n) and "gauss" in kk)
    op = torch.sigmoid(k("opacities")).flatten()
    sc = torch.exp(k("scales"))
    ms = sc.max(-1).values
    print(f"{lab} | {op.numel()/1e6:.2f}M | {q(op)} | {(op<0.005).float().mean():.3f} | {(op>0.5).float().mean():.3f} | "
          f"{q(ms)} | {(ms>1).float().mean():.3f} | {(ms>5).float().mean():.3f} | {sc.mean():.3g}", flush=True)
    del sd
