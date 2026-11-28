"""run_queue.py with a fake trainer and a fake nvidia-smi (CPU only, a few seconds)."""
import os
import stat
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_queue  # noqa: E402

FAKE = """#!{py}
import os, sys, time
out = sys.argv[1]; fail_once = sys.argv[2] == "fail_once"
os.makedirs(out, exist_ok=True)
flag = os.path.join(out, "attempted")
first = not os.path.exists(flag)
open(flag, "w").close()
time.sleep(0.3)
if fail_once and first:
    sys.exit(3)
open(os.path.join(out, "DONE"), "w").close()
"""


def setup(tmp_path, rows):
    fake = tmp_path / "fake_trainer.py"
    fake.write_text(FAKE.format(py=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    man = tmp_path / "m.csv"
    man.write_text("run_id,exp,priority,dataset,partition,method,seed,rounds,overrides\n" + "\n".join(rows) + "\n")
    tpl = f"{sys.executable} {fake} {{out}} {{overrides}}"
    return man, tpl


def queue(tmp_path, man, tpl, *extra, env=None):
    cmd = [sys.executable, os.path.join(ROOT, "run_queue.py"), "--manifest", str(man), "--out-root",
           str(tmp_path / "res"), "--cmd-template", tpl, "--settle", "0", "--poll", "0.1", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})},
                          timeout=120)


def test_skip_on_restart_and_retry_once(tmp_path):
    man, tpl = setup(tmp_path, ["a,E,1,debug,iid,ehsfp,0,3,oracle=1", "b,E,1,debug,iid,ehsfp,0,3,oracle=0"])
    # {overrides} is only a marker the fake trainer reads: use the first run to fail once
    tpl = tpl.replace("{overrides}", "ok")
    (tmp_path / "res" / "a").mkdir(parents=True)
    r = queue(tmp_path, man, tpl.replace(" ok", " fail_once"), "--jobs", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    log = (tmp_path / "res" / "queue_status.log").read_text()
    assert log.count("FAIL") == 2 and log.count("retry") == 2 and log.count("DONE ") == 2, log  # both fail once
    assert (tmp_path / "res" / "a" / "stdout_try1.log").exists() and (tmp_path / "res" / "a" / "stdout_try2.log").exists()
    # restart: everything finished -> nothing to do, no new START lines
    before = log.count("START")
    r2 = queue(tmp_path, man, tpl)
    assert "0 runs to do" in r2.stdout
    assert (tmp_path / "res" / "queue_status.log").read_text().count("START") == before


def test_gives_up_after_one_retry(tmp_path):
    man, tpl = setup(tmp_path, ["c,E,1,debug,iid,ehsfp,0,3,oracle=1"])
    always = tpl.replace("{overrides}", "x").replace(sys.executable + " ", sys.executable + " -c 'import sys; sys.exit(2)' ", 1)
    r = queue(tmp_path, man, always)
    log = (tmp_path / "res" / "queue_status.log").read_text()
    assert r.returncode == 1 and "giving up" in log and log.count("START") == 2


def test_min_free_gb_waits(tmp_path):
    man, tpl = setup(tmp_path, ["d,E,1,debug,iid,ehsfp,0,3,oracle=1"])
    tpl = tpl.replace("{overrides}", "ok")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    mem = tmp_path / "free_mb"
    mem.write_text("1024")  # 1 GB free < 4 GB required
    smi = bindir / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\ncat {mem}\n")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    env = {"PATH": f"{bindir}:{os.environ['PATH']}"}
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "run_queue.py"), "--manifest", str(man), "--out-root",
                          str(tmp_path / "res"), "--cmd-template", tpl, "--settle", "0", "--poll", "0.2",
                          "--min-free-gb", "4"], cwd=ROOT, env={**os.environ, **env},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2.5)
    assert p.poll() is None and not (tmp_path / "res" / "d" / "DONE").exists()
    log = tmp_path / "res" / "queue_status.log"
    assert not log.exists() or "START" not in log.read_text()  # still waiting
    mem.write_text("8192")
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, out
    assert (tmp_path / "res" / "d" / "DONE").exists()


def test_gates_dry_run_maps_flags():
    r = subprocess.run([sys.executable, "run_queue.py", "--manifest", "gates.csv", "--dry-run"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cmds = [l for l in r.stdout.splitlines() if "main.py" in l]
    assert len(cmds) == 5
    c1c2 = [c for c in cmds if "G1_C1C2" in c][0]
    for frag in ("--max-delay 5", "--oracle 1", "--drift-comp rate", "--alpha-mode optimal",
                 "configs/revision/hsfp_setup_cifar100.yaml", "--seed 1", "--rounds 60",
                 "--out results/G1_C1C2_tau5_s1"):
        assert frag in c1c2, (frag, c1c2)
