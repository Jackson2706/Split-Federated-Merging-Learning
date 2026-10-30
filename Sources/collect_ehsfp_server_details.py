#!/usr/bin/env python3
"""Recover evidence for E-HSFP's 200-round experiment description.

Run with the experiment's Python environment, from the server repository:
    python collect_ehsfp_server_details.py --repo . --results /path/to/200-round/results
If results are inside the repository, --results can be omitted.

Output: ehsfp_server_details.json. Upload it with the 200-round launch command.
This inspects existing files and the CURRENT environment. It never starts
training, installs packages, loads checkpoints, or sends network requests.
Current environment/source are not evidence of historical settings unless
the author confirms that the same environment/source produced the results.
Only the output JSON is created; original experiment files are not modified.
"""

import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


SECRET = re.compile(r"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret|authorization|private[_-]?key)")
URL_AUTH = re.compile(r"(https?://)[^\s/@:]+:[^\s/@]+@", re.I)
RELEVANT = re.compile(
    r"round|epoch|seed|dataset|partition|split|backbone|resnet|vgg|unet|cut[_ -]?layer|"
    r"feature[_ -]?dim|client|edge|cloud|local|batch|sample|shuffle|drop_last|"
    r"temperature|optimizer|learning|scheduler|warmup|warm_up|weight_decay|"
    r"memory|replay|prototype|reliab|covariance|cholesky|low.rank|lambda|beta|gamma|"
    r"kappa|chi|epsilon|dropout|stale|timeout|cold.start|latency|partial|coverage|"
    r"empty|invalid|skip|iou|dice|f1|accuracy|elapsed|duration|wall.time|git.commit|"
    r"torch|cuda|cudnn|python|version|finished|completed|started|t1|t2|I_c",
    re.I,
)
SKIP = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".cache",
        "checkpoints", "weights", "wandb", "data", "datasets", "images", "figures"}
LIMITS = {"records": 100, "logs": 25, "source": 35}
MAX_BYTES = 512 * 1024


def clean_text(text):
    lines = []
    for line in str(text).splitlines():
        if SECRET.search(line):
            lines.append("[REDACTED: credential-like field]")
        else:
            lines.append(URL_AUTH.sub(r"\1[REDACTED]@", line))
    return "\n".join(lines)


def clean_object(value, key="", depth=0):
    if SECRET.search(key):
        return "[REDACTED]"
    if depth > 14:
        return "[nested content omitted]"
    if isinstance(value, dict):
        return {str(k): clean_object(v, str(k), depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 200:
            return {"item_count": len(value), "first_20": [clean_object(v, key, depth + 1) for v in value[:20]],
                    "truncated": True}
        return [clean_object(v, key, depth + 1) for v in value]
    if isinstance(value, str):
        return clean_text(value)[:12000]
    return value


def command(args, cwd=None, timeout=12):
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, check=False)
        return {"exit_code": p.returncode, "stdout": clean_text(p.stdout)[:24000],
                "stderr": clean_text(p.stderr)[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": type(exc).__name__}


def small_file(path):
    try:
        return Path(path).read_text(errors="replace").strip()[:12000]
    except OSError:
        return None


def environment():
    cpu = small_file("/proc/cpuinfo") or ""
    model = re.search(r"^model name\s*:\s*(.+)$", cpu, re.M)
    packages = {}
    for name in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "PyYAML", "pandas", "flwr"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    quota_files = ["/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max",
                   "/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                   "/sys/fs/cgroup/cpu/cpu.cfs_period_us", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]
    torch_probe = (
        "import json,torch;print(json.dumps({'torch':torch.__version__,"
        "'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),"
        "'cuda_available':torch.cuda.is_available()}))"
    )
    return {
        "scope": "Inspection-time environment; confirm historical correspondence before citing in paper.",
        "python": sys.version, "os_release": small_file("/etc/os-release"),
        "kernel": platform.release(), "architecture": platform.machine(),
        "cpu_model": model.group(1) if model else platform.processor(),
        "visible_logical_cpus": os.cpu_count(),
        "process_cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "host_memory_info": small_file("/proc/meminfo"),
        "cgroup_limits": {p: small_file(p) for p in quota_files},
        "packages": packages,
        "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "cuda_toolkit": command(["nvcc", "--version"]),
        "torch_runtime_probe": command([sys.executable, "-I", "-c", torch_probe], timeout=25),
    }


def kind(path, repo):
    name = path.name.lower()
    parts = {x.lower() for x in path.parts}
    suffix = path.suffix.lower()
    if suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".csv"}:
        if (any(x in name for x in ("config", "metadata", "manifest", "hparam", "summary", "metrics", "result", "setting"))
                or parts.intersection({"config", "configs", "configuration"})):
            return "records"
    if suffix in {".log", ".out"} or (suffix == ".txt" and any(x in name for x in ("log", "train", "run"))):
        return "logs"
    if suffix in {".py", ".sh"}:
        if "ehsfp" in parts:
            return "source"
        if "h-sfp" in parts and name in {
            "hierarchy.py", "main.py", "model.py", "models.py", "losses.py", "dataset.py", "datasets.py", "utils.py"
        }:
            return "source"
        if name in {"journal_run.sh", "run_server.sh", "run_experiments.sh"}:
            return "source"
    return None


def snippets(text, limit):
    lines = text.splitlines()
    indices = set()
    for i, line in enumerate(lines):
        if RELEVANT.search(line):
            indices.update(range(max(0, i - 1), min(len(lines), i + 2)))
    selected = sorted(indices)
    return {"matching_lines": [f"{i + 1}: {clean_text(lines[i])}" for i in selected[:limit]],
            "truncated": len(selected) > limit}


def inspect_file(path, category, role):
    stat = path.stat()
    out = {"path": str(path), "scope": role, "kind": category, "size_bytes": stat.st_size,
           "file_mtime_utc": datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat()}
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
        truncated = len(raw) > MAX_BYTES
        if truncated:
            stream.seek(max(0, stat.st_size - 96 * 1024))
            tail = stream.read(96 * 1024)
        else:
            tail = b""
    raw = raw[:MAX_BYTES]
    text = raw.decode("utf-8", errors="replace")
    out["content_truncated"] = truncated
    if not truncated:
        out["sha256"] = hashlib.sha256(raw).hexdigest()
    if category == "logs":
        out.update(snippets(text, 110))
        if tail:
            out["tail_evidence"] = snippets(tail.decode("utf-8", errors="replace"), 60)
    elif path.suffix.lower() == ".json" and not truncated:
        try:
            out["record"] = clean_object(json.loads(text))
        except ValueError:
            out["parse_error"] = "Not valid JSON; relevant lines retained."
            out.update(snippets(text, 160))
    elif path.suffix.lower() == ".csv" and not truncated:
        reader = csv.DictReader(io.StringIO(text))
        out["columns"] = reader.fieldnames
        rows = list(reader)
        out["row_count"] = len(rows)
        out["rows"] = [clean_object(row) for row in rows[:200]]
        out["rows_truncated"] = len(rows) > 200
    elif category == "source":
        out["source_status"] = "Current source, not proven producing source."
        if "ehsfp" in {p.lower() for p in path.parts} and len(raw) <= 96 * 1024 and not truncated:
            out["source_text"] = clean_text(text)
        else:
            out.update(snippets(text, 240))
    else:
        # YAML/TOML are recorded as text, never executed or imported.
        out["text"] = clean_text(text)[:48000]
        out["text_truncated"] = len(text) > 48000 or truncated
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("ehsfp_server_details.json"))
    parser.add_argument("--no-environment", action="store_true", help="Inspect saved records only, e.g. on a MacBook.")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir():
        parser.error("--repo must be an existing directory")
    roots = [(p.resolve(), "author-selected results directory") for p in args.results]
    roots.append((repo, "repository file; campaign correspondence unverified"))
    if any(not p.is_dir() for p, _ in roots):
        parser.error("Every --results path must be an existing directory")
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; choose another --output path to preserve it")
    report = {
        "collected_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "Recover existing evidence for 200-round experiments; do not substitute 60-round development records.",
        "limitations": [
            "Current software and source may differ from the historical training environment.",
            "Configuration defaults do not establish resolved run settings or prove that a loss was executed.",
            "Missing evidence means not found within the bounded scan, not that an event or implementation did not exist.",
            "Coverage/invalid-batch rates and runtime overhead cannot be reconstructed unless logged or otherwise recorded.",
        ],
        "repository": {"root": str(repo), "head": command(["git", "rev-parse", "HEAD"], repo),
                       "branch": command(["git", "branch", "--show-current"], repo),
                       "status": command(["git", "status", "--porcelain=v1", "--untracked-files=normal"], repo)},
        "current_environment": None if args.no_environment else environment(),
        "files": [], "warnings": [],
    }
    counts = {k: 0 for k in LIMITS}
    seen = set()
    examined = 0
    for root, role in roots:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP and not d.startswith("."))
            for name in sorted(files):
                path = Path(directory) / name
                if path.is_symlink() or path.resolve() == output or path.resolve() in seen:
                    continue
                seen.add(path.resolve())
                examined += 1
                if examined > 50000:
                    break
                category = kind(path, repo)
                if category is None:
                    continue
                if counts[category] >= LIMITS[category]:
                    report["warnings"].append(f"{category} scan cap reached; omitted {path}")
                    continue
                try:
                    report["files"].append(inspect_file(path, category, role))
                    counts[category] += 1
                except (OSError, UnicodeError, csv.Error, ValueError) as exc:
                    report["warnings"].append(f"Could not inspect {path}: {type(exc).__name__}")
            if examined > 50000:
                report["warnings"].append("Overall 50,000-file scan cap reached; point --results at the exact run directory.")
                break
        if examined > 50000:
            break
    report["scan_summary"] = {"files_examined": examined, "included": counts, "limits": LIMITS}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Saved {output}")
    print(f"Included {counts['records']} records, {counts['logs']} logs, {counts['source']} source files.")
    print("Review the JSON before sharing; supply the 200-round launch command and identify the matching result files.")


if __name__ == "__main__":
    main()
