#!/usr/bin/env python3
"""Minimal dependency-free TensorBoard event reader (scalars written by torch.utils.tensorboard: Value.simple_value).

  read_scalars(paths) -> {tag: [(step, wall_time, value), ...]} (sorted by step; for a step logged by several resumed
  runs the most recent wall_time wins).
"""
import struct
from pathlib import Path


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s; s += 7
        if not c & 0x80:
            return r, i


def _fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v = b[i:i + 8]; i += 8
        elif wt == 2:
            ln, i = _varint(b, i); v = b[i:i + ln]; i += ln
        elif wt == 5:
            v = b[i:i + 4]; i += 4
        else:
            return
        yield f, wt, v


def _records(path):
    with open(path, "rb") as fh:
        while True:
            h = fh.read(12)
            if len(h) < 12:
                return
            ln = struct.unpack("<Q", h[:8])[0]
            d = fh.read(ln); fh.read(4)
            if len(d) < ln:
                return
            yield d


def read_scalars(paths):
    best = {}
    for p in paths:
        for rec in _records(p):
            wall, step, vals = 0.0, 0, []
            for f, wt, v in _fields(rec):
                if f == 1 and wt == 1:
                    wall = struct.unpack("<d", v)[0]
                elif f == 2 and wt == 0:
                    step = v
                elif f == 5 and wt == 2:                     # Summary
                    for f2, wt2, v2 in _fields(v):
                        if f2 == 1 and wt2 == 2:             # Value
                            tag, val = None, None
                            for f3, wt3, v3 in _fields(v2):
                                if f3 == 1 and wt3 == 2:
                                    tag = v3.decode("utf-8", "replace")
                                elif f3 == 2 and wt3 == 5:
                                    val = struct.unpack("<f", v3)[0]
                            if tag is not None and val is not None:
                                vals.append((tag, val))
            for tag, val in vals:
                k = (tag, step)
                if k not in best or wall >= best[k][0]:
                    best[k] = (wall, val)
    out = {}
    for (tag, step), (wall, val) in best.items():
        out.setdefault(tag, []).append((step, wall, val))
    for t in out:
        out[t].sort()
    return out


if __name__ == "__main__":
    import sys
    s = read_scalars(sorted(Path(sys.argv[1]).rglob("events.out.tfevents*")))
    for t, v in sorted(s.items()):
        print(f"{t}: n={len(v)} steps {v[0][0]}..{v[-1][0]} last={v[-1][2]:.5g}")
