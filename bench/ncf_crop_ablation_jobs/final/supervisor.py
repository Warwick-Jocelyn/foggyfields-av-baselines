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
# priority order: short finishing work first, then the long untested/long lanes so each GPU starts on a different one
LANES = [("neurad", "NFF-02"), ("emernerf", "NFF-02"), ("emernerf", "NFF-08"), ("unisim", "NFF-02"),
         ("emernerf", "NFF-10"), ("neurad", "NFF-08"), ("unisim", "NFF-08"), ("unisim", "NFF-10"), ("neurad", "NFF-10")]
DYN_PLANS = ["A", "B", "C", "D2", "D3", "N1", "N2", "N3", "N4", "N5", "P"]


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


def done(kind, clip):
    return (RES / "std/metrics" / f"metrics_{kind}__safe__{clip}.json").is_file() and \
           (RES / "std/lidar" / f"lidar_{kind}_safe__{clip}.json").is_file()


LANE_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] \[\S+ job (\d+)\] lane (\S+) (NFF-\d\d) .*rc=(\d+)")
ENV_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] \[\S+ job (\d+)\] (emernerf) env repair failed rc=(\d+).*__(NFF-\d\d)\.log")
UNI_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] \[\S+ job (\d+)\] unisim plugin install failed rc=(\d+)")


def failures(since):
    f = {}
    for p in LOGS.glob("run_finalbase_*.log"):
        try:
            lines = p.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for ln in lines:
            m = LANE_RE.match(ln)
            if m:
                ts, _, k, c, rc = m.groups(); rc = int(rc)
                if rc in (0, 75, 143):
                    continue
            else:
                m = ENV_RE.match(ln)
                if m:
                    ts, _, k, rc, c = m.groups()
                else:
                    m = UNI_RE.match(ln)
                    if not m:
                        continue
                    ts, _, rc = m.groups(); k, c = "unisim", "*"
            if time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S")) < since:
                continue
            for cc in (["NFF-02", "NFF-08", "NFF-10"] if c == "*" else [c]):
                f[(k, cc)] = f.get((k, cc), 0) + 1
    return f


def plan_text(rem):
    if not rem:
        return ""
    rows = []
    for g in range(4):
        order = [rem[g % len(rem)]] + rem[4:] + [rem[j] for j in range(min(4, len(rem))) if j != g % len(rem)]
        seen = []
        for l in order:
            if l not in seen:
                seen.append(l)
        rows += [f"{g} {k} {c}" for k, c in seen]
    return "\n".join(rows) + "\n"


def owner_alive(plan):
    hb = OUT / f".owner_plan_{plan}" / "heartbeat"
    return hb.is_file() and time.time() - hb.stat().st_mtime < 300


def write_plans(rem):
    txt = plan_text(rem)
    for p in DYN_PLANS:
        f = JF / f"plan_{p}.txt"
        if f.is_file() and f.read_text() == txt:
            continue
        tmp = JF / f".plan_{p}.txt.new"; tmp.write_text(txt); tmp.replace(f)
        if not owner_alive(p):  # stale completion markers of an old plan content must not stop the chain
            for m in list(OUT.glob(f".chain_ok_plan_{p}_gpu*")) + [OUT / f".plan_done_plan_{p}"]:
                m.unlink(missing_ok=True)
        log(f"plan_{p} rewritten: {len(rem)} lanes")


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


def publish(rem, excl, fails, q, final=False):
    sh([PY, "-I", str(JF / "build_table.py"), "--write", str(REP)])
    sh(["bash", str(JF / "publish_metrics.sh")])
    sh(["bash", str(JF / "copy_figs.sh")])
    for clip in ("NFF-02", "NFF-08", "NFF-10"):
        sh([PY, "-I", str(JF / "montage.py"), "--clip", clip, "--frame", "0099", "--out", str(DST / "figures")])
    st = "**COMPLETE.**" if final else f"**IN PROGRESS** — {len(rem)} SOTA lane(s) still running/queued; see `STATUS.md`."
    sh([PY, "-I", str(JF / "final_table.py"), "--out", str(DST / "FINAL_TABLE.md"), "--status", st])
    mine = [r for r in (q or []) if r[2].startswith("finalbase")]
    L = [f"# STATUS — SOTA baselines (auto-updated by supervisor.py, {time.strftime('%Y-%m-%d %H:%M')})", "",
         "| method | clip | state |", "|---|---|---|"]
    for k, c in LANES:
        s = "DONE (RGB + LiDAR scored)" if done(k, c) else (
            f"**EXCLUDED — failed {fails.get((k, c), 0)}x, needs attention** (log `{LOGS}/std/lanes/{k}__safe__{c}.log`)"
            if (k, c) in excl else f"remaining ({fails.get((k, c), 0)} failure(s) so far)")
        L.append(f"| {k} | {c} | {s} |")
    L += ["", "Slurm jobs of this pipeline:", ""] + [f"- {r[0]} {r[1]} {r[2]} {r[3]}" for r in mine] + [
        "", "SplatAD (allfix + opreg5e4) and the V1.3 lanes are complete. Results: `FINAL_TABLE.md` (compact),",
        "`baselines/RESULTS_BASELINES.md` (full + wall time), figures `figures/` (GT vs methods, held-out frame 99) and",
        "`/users/yitiwang/FoggyfieldV1/RGB-LiDAR-Viz/baselines/<clip>/<method>/` (renders + LiDAR BEV).",
        f"Supervisor log: `{LOGS}/supervisor.log`; restart: `nohup setsid python3 -I {JF}/supervisor.py >> {LOGS}/supervisor.out 2>&1 &`."]
    (DST / "STATUS.md").write_text("\n".join(L) + "\n")


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
        excl = {l for l in LANES if not done(*l) and fails.get(l, 0) >= MAX_FAIL}
        rem = [l for l in LANES if not done(*l) and l not in excl]
        write_plans(rem)
        q = queue()
        if q is not None:
            fb = [r for r in q if r[2].startswith("finalbase")]
            if not rem:
                for r in fb:
                    if r[3] == "PENDING":
                        sh(["scancel", r[0]]); log(f"all lanes finished -> cancelled pending {r[0]} {r[2]}")
                if not any(r[3] == "RUNNING" for r in fb):
                    publish(rem, excl, fails, queue(), final=True)
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
            log(f"published; remaining {rem}; excluded {sorted(excl)}")
        save_state(s)
        time.sleep(CYCLE)
    log("deadline"); PIDF.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
