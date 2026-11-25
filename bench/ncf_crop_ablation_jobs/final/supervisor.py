#!/usr/bin/env python3
"""Final-ablation SOTA baselines: hands-off supervisor (login node, nohup; no GPU, no Claude session needed).

Every CYCLE seconds:
  1. lane status: a lane (kind, clip) is DONE when its RGB metric JSON + LiDAR JSON exist in results_final/std/.
     A lane that failed (rc not in 0/75 busy/143 wall-clock, or env/plugin setup failure) >= MAX_FAIL times since this
     supervisor's first start is EXCLUDED and reported in STATUS.md as NEEDS ATTENTION (never retried blindly).
  2. rewrites every dynamic plan file (atomic mv; running jobs keep their snapshot) with the remaining lanes: each GPU
     chain starts on a different lane and then walks the rest (lane locks skip BUSY lanes, finished lanes skip fast).
  3. keeps one debug chunk queued (only when my debug jobs < 2 -> alternation with the other session) and, if no
     preemptable/normal finalbase job exists at all, submits one 3 h self-chaining preemptable chunk (<= 1 per 2 h).
  4. publishes: build_table.py -> RESULTS_BASELINES.md, publish_metrics.sh, copy_figs.sh, montage.py, final_table.py,
     STATUS.md in /users/yitiwang/FoggyfieldV1/RGB-Branch-Metrics.
  5. when nothing remains: cancels pending finalbase_* jobs, final publish, exits.
Restart (idempotent):  nohup setsid python3 -I $BW/jobs/final/supervisor.py >> $BW/logs_final/supervisor.out 2>&1 &
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

BW = Path("/capstor/scratch/cscs/yitiwang/ncf-crop-ablation")
JF = BW / "jobs/final"
OUT = BW / "outputs_final"
RES = BW / "results_final"
LOGS = BW / "logs_final"
DST = Path("/users/yitiwang/FoggyfieldV1/RGB-Branch-Metrics")
REP = Path("/users/yitiwang/projects/foggyfields-v1/reports/v13_overnight/final_ablation_2026-11-23/RESULTS_BASELINES.md")
PLANMD = Path("/users/yitiwang/projects/foggyfields-v1/reports/v13_overnight/final_ablation_2026-11-23/PLAN.md")
REG = Path("/users/yitiwang/projects/foggyfields-v1/runs/registry_allocations.txt")
PY = "/users/yitiwang/miniforge3/bin/python3"
STATE = LOGS / "supervisor_state.json"
PIDF = LOGS / "supervisor.pid"
CYCLE, PUBLISH_EVERY, MAX_FAIL = 300, 900, 2
DEADLINE = time.time() + 4 * 86400
CLIPS = ["NFF-02", "NFF-08", "NFF-10"]
BUDGET = 30000          # 2026-11-25 user: every method trains the SAME number of steps (fair budget) -> 30k for all
OV = Path("/capstor/scratch/cscs/yitiwang/foggyfields-v1/v13_overnight")
# 2026-11-25 18:15 user: "forget the BRDF variants for now, baseline performance only" -> no V1.3 training/scoring/viz lanes.
# The 30k V1.3 recipe (neurad-fog-v13-30k, jobs/final/v13train.sh) is kept; re-enable with the two lanes below.
V13_LANES = {}   # was {"F": "V1.3-BRDF+LiDAR_F", "Fc": "V1.3-Colour+LiDAR_Fc"}
DYN_PLANS = ["A", "B", "C", "D2", "D3", "N1", "N2", "N3", "N4", "N5", "P"]
# 3D 360-deg orbit videos (jobs/final/viz3d_orbit.py, lane kind viz3d) on one clip, enabled by the flag file after the smoke test
VIZ_CLIP = "NFF-02"
VIZ_DIR = Path("/users/yitiwang/FoggyfieldV1/RGB_branch_3D_viz")
VIZ_FLAG = JF / ".viz3d_enabled"
VIZ_ORDER = ["SplatAD", "NeuRAD", "UniSim", "EmerNeRF"] + list(V13_LANES.values())


def _json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def sota_done(k, c, v="std"):
    """RGB metrics at the 30k budget AND a LiDAR JSON written after them."""
    mp, lp = RES / v / "metrics" / f"metrics_{k}__safe__{c}.json", RES / v / "lidar" / f"lidar_{k}_safe__{c}.json"
    m = _json(mp)
    return bool(m) and int(m.get("iters") or 0) >= BUDGET and lp.is_file() and lp.stat().st_mtime >= mp.stat().st_mtime - 60


def v13_dir(c, L):
    return OV / f"20261123_{c}_v13_final30k" / "train" / f"V13-{L}"


def v13_cfg(c, L):
    cs = sorted(v13_dir(c, L).glob("*/*/*/config.yml"))
    return str(cs[0]) if cs else None


def v13_trained(c, L):
    return (v13_dir(c, L) / "TRAIN_DONE").is_file()


def v13_scored(c, L):
    lb = f"V13-{L}30k"
    return (RES / "v13/metrics" / f"metrics_v13__{lb}__{c}.json").is_file() and (RES / "v13/lidar" / f"lidar_v13__{lb}__{c}.json").is_file()


def all_lanes():
    """Every unit of work, in priority order (longest first). key = failure-count / status key, row = plan text after the GPU."""
    L = []
    for c in CLIPS:
        L.append(dict(key=f"emernerf:{c}", row=f"emernerf {c}", done=sota_done("emernerf", c), ready=True, what=f"EmerNeRF {c} (30k)"))
    for c in CLIPS:
        for l in V13_LANES:
            L.append(dict(key=f"v13train:{l}:{c}", row=f"v13train {c} {l}", done=v13_trained(c, l), ready=True,
                          what=f"V1.3 {l} {c} train 30k"))
    for c in CLIPS:
        L.append(dict(key=f"unisim:{c}", row=f"unisim {c}", done=sota_done("unisim", c), ready=True, what=f"UniSim {c} (30k)"))
    for c in ("NFF-08", "NFF-10", "NFF-02"):
        L.append(dict(key=f"neurad:{c}", row=f"neurad {c}", done=sota_done("neurad", c), ready=True, what=f"NeuRAD {c} (30k)"))
    # 2026-11-25 user: add NFF-05 (night / dark weather) for the SOTA comparison (V1.3 on NFF-05 needs its per-clip inputs first)
    for k, v, nm in (("emernerf", "std", "EmerNeRF"), ("splatad", "allfix", "SplatAD allfix"), ("unisim", "std", "UniSim"), ("neurad", "std", "NeuRAD")):
        L.append(dict(key=f"{k}:NFF-05", row=f"{k} NFF-05", done=sota_done(k, "NFF-05", v), ready=True, what=f"{nm} NFF-05 night (30k)"))
    for m, v, c in [("splatad", "allfix", c) for c in CLIPS] + [("neurad", "std", "NFF-02")]:   # scored before dumps were kept
        L.append(dict(key=f"lidardump:{m}:{c}", row=f"lidardump {c} {m} {v}", done=(OUT / v / "_lidar_dump" / f"{m}_safe__{c}").is_dir(),
                      ready=True, score=True, what=f"{m} {c} LiDAR re-dump (extended metrics)"))
    for c in CLIPS:
        for l in V13_LANES:
            cfg = v13_cfg(c, l)
            L.append(dict(key=f"v13:V13-{l}30k:{c}", row=f"v13 V13-{l}30k {c} {cfg}", done=v13_scored(c, l),
                          ready=bool(v13_trained(c, l) and cfg), score=True, what=f"V1.3 {l} {c} score (common evaluator)"))
    return L


def viz_lanes():
    out = []

    def td_cfg(p):   # TRAIN_DONE -> its config.yml, only when trained at the budget
        try:
            t = p.read_text().splitlines()
            if any(f"iters={BUDGET}" in x for x in t) and t[0].strip().endswith("config.yml") and Path(t[0].strip()).is_file():
                return t[0].strip()
        except Exception:
            return None
    for lb, p in (("SplatAD", OUT / "allfix/splatad/safe" / VIZ_CLIP / "TRAIN_DONE"), ("NeuRAD", OUT / "std/neurad/safe" / VIZ_CLIP / "TRAIN_DONE"),
                  ("UniSim", OUT / "std/unisim/safe" / VIZ_CLIP / "TRAIN_DONE")):
        cfg = td_cfg(p)
        if cfg:
            out.append((lb, cfg))
    if sota_done("emernerf", VIZ_CLIP):
        out.append(("EmerNeRF", str(OUT / "std/emernerf/safe" / VIZ_CLIP / "train/run")))
    for l, lb in V13_LANES.items():
        if v13_trained(VIZ_CLIP, l) and v13_cfg(VIZ_CLIP, l):
            out.append((lb, v13_cfg(VIZ_CLIP, l)))
    return out


def viz_done(lb):
    """meta.json + mp4 from a checkpoint at the budget; a stale (smaller-step) render is removed so it is redone."""
    d = VIZ_DIR / VIZ_CLIP / lb
    m = _json(d / "meta.json")
    if m is None or not any(d.glob("orbit360_*.mp4")):
        return False
    if int(m.get("step") or 0) < BUDGET - 2:
        for f in list(d.glob("*.mp4")) + list(d.glob("*.jpg")) + [d / "meta.json"]:
            f.unlink(missing_ok=True)
        return False
    return True


def log(msg):
    print(f"[{time.strftime('%F %T')}] [supervisor] {msg}", flush=True)


def sh(cmd, timeout=600):
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return 99, str(e)


def state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(s):
    tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(s, indent=1)); tmp.replace(STATE)


LANE_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] \[\S+ job (\d+)\] lane (\S+) (\S+) ?(.*?)\s*gpu \d rc=(\d+)")
ENV_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] \[\S+ job (\d+)\] emernerf env repair failed rc=(\d+).*__(NFF-\d\d)\.log")
UNI_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] \[\S+ job (\d+)\] unisim plugin install failed rc=(\d+)")


def failures(since):
    """-> {lane key: number of real failures since `since`} (rc 0 ok, 75 busy, 143 wall-clock are not failures)."""
    f = {}

    def add(ts, keys):
        if time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S")) >= since:
            for k in keys:
                f[k] = f.get(k, 0) + 1
    for p in LOGS.glob("run_finalbase_*.log"):
        try:
            lines = p.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for ln in lines:
            m = LANE_RE.match(ln)
            if m:
                ts, _, k, a1, rest, rc = m.groups()
                if int(rc) in (0, 75, 143):
                    continue
                r1 = rest.split()[0] if rest.split() else ""
                key = {"viz3d": f"viz3d:{r1}:{a1}", "v13train": f"v13train:{r1}:{a1}", "v13": f"v13:{a1}:{r1}",
                       "lidardump": f"lidardump:{r1}:{a1}"}.get(k, f"{k}:{a1}")
                add(ts, [key]); continue
            m = ENV_RE.match(ln)
            if m:
                add(m.group(1), [f"emernerf:{m.group(4)}"]); continue
            m = UNI_RE.match(ln)
            if m:
                add(m.group(1), [f"unisim:{c}" for c in CLIPS])
    return f


def plan_text(rem, viz=(), score=()):
    if not rem and not viz and not score:
        return ""
    rows = []
    for g in range(4):
        rows += [f"{g} viz3d {VIZ_CLIP} {lb} {cfg}" for j, (lb, cfg) in enumerate(viz) if j % 4 == g]
        rows += [f"{g} {r}" for j, r in enumerate(score) if j % 4 == g]
        if not rem:
            continue
        order = [rem[g % len(rem)]] + rem[4:] + [rem[j] for j in range(min(4, len(rem))) if j != g % len(rem)]
        seen = []
        for r in order:
            if r not in seen:
                seen.append(r)
        rows += [f"{g} {r}" for r in seen]
    return "\n".join(rows) + "\n"


def owner_alive(plan):
    hb = OUT / f".owner_plan_{plan}" / "heartbeat"
    return hb.is_file() and time.time() - hb.stat().st_mtime < 300


def write_plans(rem, viz=(), score=()):
    txt = plan_text(rem, viz, score)
    for p in DYN_PLANS:
        f = JF / f"plan_{p}.txt"
        if f.is_file() and f.read_text() == txt:
            continue
        tmp = JF / f".plan_{p}.txt.new"; tmp.write_text(txt); tmp.replace(f)
        if not owner_alive(p):  # stale completion markers of an old plan content must not stop the chain
            for m in list(OUT.glob(f".chain_ok_plan_{p}_gpu*")) + [OUT / f".plan_done_plan_{p}"]:
                m.unlink(missing_ok=True)
        log(f"plan_{p} rewritten: {txt.count(chr(10))} rows")


def queue():
    rc, out = sh(["squeue", "-u", "yitiwang", "-h", "-o", "%i|%P|%j|%T"])
    if rc != 0:
        return None
    return [l.split("|") for l in out.strip().splitlines() if l.count("|") == 3]


def submit(acct, part, tl, name, plan, chain):
    exp = f"ALL,PLAN={JF}/plan_{plan}.txt" + (",CHAIN=1" if chain else "")
    rc, out = sh(["sbatch", "--parsable", "-A", acct, "-p", part, "-t", tl, f"--job-name={name}", f"--export={exp}",
                  str(JF / "run_final.sbatch")])
    jid = out.strip().splitlines()[-1] if out.strip() else ""
    if rc == 0 and jid.split(";")[0].isdigit():
        jid = jid.split(";")[0]
        with open(REG, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} job {jid} {acct}/{part} 1 node <= {tl}: "
                     f"plan_{plan} chunk (supervisor.py, dynamic remaining-lane plan{', CHAIN=1' if chain else ''}) [final-ablation]\n")
        with open(PLANMD, "a") as fh:
            fh.write(f"- {time.strftime('%H:%M')} supervisor: {part} chunk {jid} = plan_{plan} (remaining lanes).\n")
        log(f"submitted {name} {acct}/{part} {tl} -> {jid}")
        return jid
    log(f"sbatch {name} failed: {out.strip()[:300]}")
    return None


def lidar_ext():
    """extended LiDAR metrics (jobs/final/lidar_extended.py, CPU) for every per-sweep dump without an up-to-date result."""
    jobs = []
    for v in ("allfix", "opreg5e4", "std"):
        for d in (OUT / v / "_lidar_dump").glob("*_safe__NFF-*"):
            m, c = d.name.split("_safe__")
            jobs.append((d, RES / v / "lidar_ext" / f"lidar_ext__{m}__{c}.json", m, c))
    for d in (OUT / "std/emernerf/safe").glob("NFF-*/lidar_dump"):
        jobs.append((d, RES / "std/lidar_ext" / f"lidar_ext__emernerf__{d.parent.name}.json", "emernerf", d.parent.name))
    for d in (OUT / "v13").glob("*__NFF-*/lidar_dump"):
        lb, c = d.parent.name.split("__")
        jobs.append((d, RES / "v13/lidar_ext" / f"lidar_ext__{lb}__{c}.json", lb, c))
    for d, o, lb, c in jobs:
        fs = list(d.glob("frame_*.npz"))
        if len(fs) < 99 or (o.is_file() and o.stat().st_mtime >= max(f.stat().st_mtime for f in fs)):
            continue
        rc, out = sh(["env", "OPENBLAS_NUM_THREADS=4", PY, "-I", str(JF / "lidar_extended.py"), "--dump", str(d), "--out", str(o),
                      "--label", lb, "--clip", c], timeout=1800)
        log(f"lidar_ext {lb} {c} rc={rc} {out.strip()[-160:]}")


def publish(rem, excl, fails, q, final=False):
    rem = list(rem)
    lidar_ext()
    sh(["env", "OPENBLAS_NUM_THREADS=4", PY, "-I", str(JF / "convergence_check.py"), "--out-md", str(DST / "baselines/CONVERGENCE.md"),
        "--out-json", str(DST / "baselines/convergence.json")], timeout=1800)
    viz_index()
    sh([PY, "-I", str(JF / "build_table.py"), "--write", str(REP)])
    sh(["bash", str(JF / "publish_metrics.sh")])
    sh(["bash", str(JF / "copy_figs.sh")])
    for clip in ("NFF-02", "NFF-08", "NFF-10"):
        sh([PY, "-I", str(JF / "montage.py"), "--clip", clip, "--frame", "0099", "--out", str(DST / "figures")])
    st = "**COMPLETE.**" if final else f"**IN PROGRESS** — {len(rem)} work unit(s) still running/queued; see `STATUS.md`."
    sh([PY, "-I", str(JF / "final_table.py"), "--out", str(DST / "FINAL_TABLE.md"), "--status", st])
    mine = [r for r in (q or []) if r[2].startswith("finalbase")]
    L = [f"# STATUS — fair-budget comparison, every method {BUDGET} steps (auto-updated by supervisor.py, {time.strftime('%Y-%m-%d %H:%M')})", "",
         "| work unit | state |", "|---|---|", "| SplatAD allfix + opreg5e4 x3 clips (30k) | DONE |"]
    for l in all_lanes():
        n = fails.get(l["key"], 0)
        st = "DONE" if l["done"] else (f"**EXCLUDED — failed {n}x, needs attention**" if l["key"] in excl else
                                       ("remaining" if l["ready"] else "waits for its training") + (f" ({n} failure(s))" if n else ""))
        L.append(f"| {l['what']} | {st} |")
    for lb, _ in [(x, None) for x in VIZ_ORDER]:
        L.append(f"| 360-degree video {VIZ_CLIP} {lb} | {'DONE' if viz_done(lb) else 'queued (after its 30k training)'} |")
    L += ["", "Slurm jobs of this pipeline:", ""] + [f"- {r[0]} {r[1]} {r[2]} {r[3]}" for r in mine] + [
        "", "Results: `FINAL_TABLE.md` (compact, with training time), `baselines/CONVERGENCE.md` (validation),",
        "`baselines/RESULTS_BASELINES.md` (full + wall time), figures `figures/` (GT vs methods, held-out frame 99) and",
        "`/users/yitiwang/FoggyfieldV1/RGB-LiDAR-Viz/baselines/<clip>/<method>/` (renders + LiDAR BEV).",
        f"Supervisor log: `{LOGS}/supervisor.log`; restart: `nohup setsid python3 -I {JF}/supervisor.py >> {LOGS}/supervisor.out 2>&1 &`."]
    (DST / "STATUS.md").write_text("\n".join(L) + "\n")


def viz_index():
    L = ["# RGB_branch_3D_viz — 360-degree orbit videos of every reconstructed scene (auto-updated by supervisor.py)", "",
         f"Clip {VIZ_CLIP}, camera front_wide_120fov, middle held-out timestep (time frozen, actors frozen). Frame 0 = the REAL",
         "held-out camera (compare with `frame0_vs_gt_*.jpg`: render vs GT, PSNR printed); then the camera orbits 360 deg around a",
         "point 8 m ahead, rising to +4 m at the far side. Left = model RGB, right = model expected depth (log 1-80 m, red = near, blue = far or empty).",
         "What to look for: road/kerbs/poles stay straight and in place while the view rotates (correct 3D); floaters, grey",
         "fog-blobs in free space, or a scene that only looks right at frame 0 = overfitted / wrong geometry.",
         "Renderer: `/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/final/viz3d_orbit.py` (lane kind viz3d in node_final.sh).", "",
         "| method | video | frame-0 PSNR vs GT (0.5x) | median depth range over orbit (m) | checkpoint step |", "|---|---|---|---|---|"]
    for lb in VIZ_ORDER:
        d = VIZ_DIR / VIZ_CLIP / lb
        try:
            m = json.loads((d / "meta.json").read_text())
            vids = " ".join(f"[{p.name}]({VIZ_CLIP}/{lb}/{p.name})" for p in sorted(d.glob("orbit360_*.mp4")))
            dr = m.get("depth_median_m_min_max")
            L.append(f"| {lb} | {vids} | {m['frame0_psnr_vs_gt_db']:.2f} dB | {dr[0]:.1f}-{dr[1]:.1f} | {m['step']} |" if dr else
                     f"| {lb} | {vids} | {m['frame0_psnr_vs_gt_db']:.2f} dB | - | {m['step']} |")
        except Exception:
            L.append(f"| {lb} | queued (rendered automatically once the method has finished its {BUDGET}-step training) | | | |")
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    (VIZ_DIR / "README.md").write_text("\n".join(L) + "\n")


def main():
    if PIDF.is_file():
        try:
            os.kill(int(PIDF.read_text().strip()), 0)
            log("another supervisor is alive -> exit"); return
        except Exception:
            pass
    PIDF.write_text(str(os.getpid()))
    s = state()
    s.setdefault("since", time.time()); s.setdefault("last_debug", 0); s.setdefault("last_fallback", 0); s.setdefault("last_pub", 0)
    save_state(s)
    log(f"start pid {os.getpid()} (failures counted since {time.strftime('%F %T', time.localtime(s['since']))})")
    while time.time() < DEADLINE:
        fails = failures(s["since"])
        lanes = all_lanes()
        excl = {l["key"] for l in lanes if not l["done"] and fails.get(l["key"], 0) >= MAX_FAIL}
        act = [l for l in lanes if not l["done"] and l["key"] not in excl and l["ready"]]
        rem = [l["row"] for l in act if not l.get("score")]
        score = [l["row"] for l in act if l.get("score")]
        viz = [v for v in viz_lanes() if VIZ_FLAG.is_file() and not viz_done(v[0])
               and fails.get(f"viz3d:{v[0]}:{VIZ_CLIP}", 0) < MAX_FAIL]
        write_plans(rem, viz, score)
        q = queue()
        if q is not None:
            fb = [r for r in q if r[2].startswith("finalbase")]
            if not rem and not viz and not score:
                for r in fb:
                    if r[3] == "PENDING":
                        sh(["scancel", r[0]]); log(f"all lanes finished -> cancelled pending {r[0]} {r[2]}")
                if not any(r[3] == "RUNNING" for r in fb):
                    publish(rem, excl, fails, queue(), final=True)
                    rc, out = sh(["bash", str(JF / "publish_github.sh"), "final"], timeout=1200)
                    log(f"GitHub final publish rc={rc}: {out.strip()[-400:]}")
                    with open(PLANMD, "a") as fh:
                        fh.write(f"- {time.strftime('%m-%d %H:%M')} supervisor: ALL DONE; GitHub final publish rc={rc}; excluded {sorted(excl)}.\n")
                    log(f"ALL DONE (excluded: {sorted(excl)})"); PIDF.unlink(missing_ok=True); return
            else:
                dbg_mine = [r for r in q if r[1] == "debug"]
                dbg_fb = [r for r in dbg_mine if r[2].startswith("finalbase_D")]
                if not dbg_fb and len(dbg_mine) < 2 and time.time() - s["last_debug"] > 900:
                    if submit("a0195", "debug", "01:30:00", "finalbase_D2", "D2", False):
                        s["last_debug"] = time.time()
                big = [r for r in fb if r[1] != "debug"]
                if not big and time.time() - s["last_fallback"] > 7200:
                    acct = "a0284" if s.get("fallback_acct") != "a0284" else "a0195"
                    if submit(acct, "preemptable", "03:00:00", "finalbase_P", "P", True):
                        s["last_fallback"] = time.time(); s["fallback_acct"] = acct
        if time.time() - s["last_pub"] > PUBLISH_EVERY:
            publish(rem, excl, fails, q); s["last_pub"] = time.time()
            log(f"published; remaining {rem + score}; viz {[v[0] for v in viz]}; excluded {sorted(excl)}")
        save_state(s)
        time.sleep(CYCLE)
    log("deadline"); PIDF.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
