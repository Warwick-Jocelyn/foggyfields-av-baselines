import torch, glob, math
FF = '/capstor/scratch/cscs/yitiwang/foggyfields'
for v in ['V2b', 'V2f', 'V2t-tv0', 'V2t', 'V2m', 'N13med']:
    cks = glob.glob(f'{FF}/p3/{v}/**/step-000029999.ckpt', recursive=True)
    if not cks: print(v, 'ckpt not found'); continue
    sd = torch.load(cks[0], map_location='cpu')['pipeline']
    gk = [k for k in sd if 'gain' in k.lower()]
    for k in gk:
        val = float(sd[k].float().mean())
        print(v, k.split('.')[-1], 'raw=', round(val, 4), '=> gain=', round(math.exp(val), 3))
    if not gk: print(v, 'NO gain key in ckpt')
