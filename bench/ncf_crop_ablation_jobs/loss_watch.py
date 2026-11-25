#!/usr/bin/env python3
"""Loss-convergence watch for the crop/mask ablation.

Reads the TensorBoard event files the runs write and reports, per run, whether the loss is
still going down, has flattened, has stalled, or has gone non-finite. No tensorboard/torch
needed: TFRecord is [u64 len][u32 crc][payload][u32 crc], and the Event protobuf is walked
with a minimal varint reader -- enough to pull `step`, `wall_time` and scalar summaries.

Usage: loss_watch.py <outputs_root> [--tag-substr loss] [--json]
Prints one line per run; exit 1 if anything looks wrong (non-finite / stalled / rising).
"""
import os, sys, struct, json, math, time
from collections import defaultdict

def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, i
        s += 7

def _fields(b):
    """Yield (field_number, wire_type, value_or_bytes) for one protobuf message."""
    i, n = 0, len(b)
    while i < n:
        try:
            k, i = _varint(b, i)
        except IndexError:
            return
        fn, wt = k >> 3, k & 7
        if wt == 0:
            v, i = _varint(b, i); yield fn, wt, v
        elif wt == 1:
            if i + 8 > n: return
            yield fn, wt, b[i:i+8]; i += 8
        elif wt == 2:
            ln, i = _varint(b, i)
            if i + ln > n: return
            yield fn, wt, b[i:i+ln]; i += ln
        elif wt == 5:
            if i + 4 > n: return
            yield fn, wt, b[i:i+4]; i += 4
        else:
            return

def read_events(path, want):
    """-> {tag: [(step, value), ...]}  for tags containing any string in `want`."""
    out = defaultdict(list)
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return out
    i, n = 0, len(data)
    while i + 12 <= n:
        (ln,) = struct.unpack_from('<Q', data, i)
        body = i + 12
        if body + ln + 4 > n:
            break
        rec = data[body:body + ln]
        i = body + ln + 4
        step = None; vals = []
        for fn, wt, v in _fields(rec):
            if fn == 2 and wt == 0:          # Event.step
                step = v
            elif fn == 5 and wt == 2:        # Event.summary
                for sfn, swt, sv in _fields(v):
                    if sfn != 1 or swt != 2: continue   # Summary.value
                    tag = None; sval = None
                    for vfn, vwt, vv in _fields(sv):
                        if vfn == 1 and vwt == 2:
                            tag = vv.decode('utf-8', 'replace')
                        elif vfn == 2 and vwt == 5:
                            sval = struct.unpack('<f', vv)[0]
                    if tag is not None and sval is not None:
                        vals.append((tag, sval))
        if step is None:
            continue
        for tag, sval in vals:
            if any(w in tag.lower() for w in want):
                out[tag].append((step, sval))
    return out

def verdict(series):
    """series = [(step, value)] -> (status, detail)

    Median-based, not mean-based. SplatAD's MCMC densification/relocation produces transient
    loss spikes two orders of magnitude above the running level; a single one of those in the
    tail window is enough to make a mean-based test call a healthily converging run "RISING".
    The spike COUNT is reported separately, because a run that is genuinely unstable spikes
    repeatedly rather than once.
    """
    if len(series) < 8:
        return "starting", f"{len(series)} points"
    series = sorted(series)
    steps = [s for s, _ in series]
    vals = [v for _, v in series]
    if any(not math.isfinite(v) for v in vals):
        bad = next(s for (s, v) in series if not math.isfinite(v))
        return "NON-FINITE", f"first at step {bad}"
    h = max(4, len(vals) // 4)
    head, tail = sorted(vals[:h]), sorted(vals[-h:])
    f = head[len(head) // 2]
    l = tail[len(tail) // 2]
    rel = (f - l) / f if f else 0.0
    spikes = sum(1 for v in vals[-h:] if l and v > 4 * l)
    p95 = tail[min(len(tail) - 1, int(0.95 * len(tail)))]
    sp = f", {spikes} spikes>4x, p95 {p95:.3f}" if spikes else ""
    where = f"step {steps[-1]}: med {f:.4f} -> {l:.4f}"
    if rel < -0.05:
        return "RISING", f"{where} (UP {-rel:.1%}{sp})"
    if spikes > max(4, 0.05 * h):   # a RATE: one spike in 750 is MCMC, one in 20 is not
        return "UNSTABLE", f"{where} (down {rel:.1%} but {spikes}/{h} tail spikes>4x, p95 {p95:.3f})"
    if rel > 0.02:
        return "converging", f"{where} (down {rel:.1%}{sp})"
    iqr = (tail[int(0.75 * len(tail))] - tail[int(0.25 * len(tail))]) / l if l else 0.0
    if iqr < 0.15:
        return "flat", f"{where} (tail IQR {iqr:.1%}{sp})"
    return "noisy-flat", f"{where} ({rel:+.1%}, tail IQR {iqr:.1%}{sp})"


TARGET_ITERS = int(os.environ.get("SPLATAD_ITERS", "30000"))


def main():
    root = sys.argv[1]
    want = ["loss"]
    if "--tag-substr" in sys.argv:
        want = [sys.argv[sys.argv.index("--tag-substr") + 1].lower()]
    as_json = "--json" in sys.argv
    runs = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.startswith("events.out.tfevents"):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, root).split(os.sep)
            name = "/".join(rel[:3]) if len(rel) >= 3 else "/".join(rel[:-1])
            runs.setdefault(name, []).append(p)
    bad = 0
    report = {}
    for name in sorted(runs):
        merged = defaultdict(list)
        newest = 0.0
        for p in runs[name]:
            newest = max(newest, os.path.getmtime(p))
            for tag, pts in read_events(p, want).items():
                merged[tag].extend(pts)
        if not merged:
            print(f"  {name:28s} no scalar matching {want} yet")
            continue
        tag = min(merged, key=lambda t: (("train" not in t.lower()), len(t)))
        st, detail = verdict(merged[tag])
        age = time.time() - newest
        last_step = max(s for s, _ in merged[tag]) if merged[tag] else 0
        # A finished run stops writing events; that is completion, not a stall.
        finished = last_step >= TARGET_ITERS - 50
        stale = (age > 900) and not finished
        if st in ("NON-FINITE", "RISING", "UNSTABLE") or stale:
            bad += 1
        flag = ("  <<< STALLED (%.0f min since last write)" % (age / 60)) if stale else ("  [training complete]" if finished else "")
        print(f"  {name:28s} {st:11s} {tag:22s} {detail}{flag}")
        report[name] = {"status": st, "tag": tag, "detail": detail, "stale_s": int(age),
                        "n_points": len(merged[tag])}
    if as_json:
        print(json.dumps(report, indent=2))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
