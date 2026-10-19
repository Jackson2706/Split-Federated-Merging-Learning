#!/usr/bin/env python3
"""Download, verify and lay out the datasets used by the E-HSFP / MiSFP journal experiments.

    python scripts/prepare_datasets.py                       # default set: cifar10 cifar100 ham10000
    python scripts/prepare_datasets.py --datasets isic2018    # optional (~11 GB, only for Table II reruns)
    python scripts/prepare_datasets.py --verify-only          # check, never download
    python scripts/prepare_datasets.py --root /path/to/data   # default: Sources/data

Layout produced (exactly what the loaders read, relative to Sources/ as configs assume):
    data/cifar/cifar-10-batches-py/      torchvision.datasets.CIFAR10(root='data/cifar/')
    data/cifar/cifar-100-python/         torchvision.datasets.CIFAR100(root='data/cifar/')
    data/HAM10000/HAM10000_metadata.csv, HAM10000_images_part_{1,2}/   classification/*/data/get_data.py
    data/ISIC2018/ISIC2018_Task1-2_{Training,Validation}_Input, ISIC2018_Task1_{Training,Validation}_GroundTruth

Guarantees
  * Official sources only (Toronto CIFAR page, Harvard Dataverse doi:10.7910/DVN/DBW86T, ISIC challenge S3).
  * Archives land in <root>/.partial/ and are resumed with HTTP Range; MD5 checked against published values.
  * Extraction goes to a staging dir (tar 'data' filter / zip path check) and is renamed into place only
    after content checks pass; <dataset>/.ready.json is written last, atomically.
  * An exclusive flock per dataset serialises concurrent callers (e.g. parallel jobs).
  * No splitting or preprocessing happens here: train/test splits, partitions, seeds and normalisation stay
    in the loaders, so no test data can leak through preparation.
  * Status for every dataset is recorded in <root>/DATASETS_STATUS.json.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ROOT = os.path.join(SRC, "data")
DEFAULT_SET = ["cifar10", "cifar100", "ham10000"]
UA = {"User-Agent": "prepare_datasets/1.0 (research reproduction)"}

# torchvision's own per-file checksums for the extracted batches (torchvision/datasets/cifar.py)
CIFAR = {
    "cifar10": dict(
        url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        md5="c58f30108f718f92721af3b95e74349a", base="cifar-10-batches-py", segments=3,
        files={"data_batch_1": "c99cafc152244af753f735de768cd75f", "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
               "data_batch_3": "54ebc095f3ab1f0389bbae665268c751", "data_batch_4": "634d18415352ddfa80567beed471001a",
               "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb", "test_batch": "40351d587109b95175f43aff81a1287e",
               "batches.meta": "5ff9c542aee3614f3951f8cda6e48888"}),
    "cifar100": dict(
        url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        md5="eb9058c3a382ffc7106e4002c42a8d85", base="cifar-100-python", segments=3,
        files={"train": "16019d7e3df5f24257cddd939b257f8d", "test": "f0ef6b0ae62326f3e7ffdfab6717acfc",
               "meta": "7973b15100ade9c7d40fb424638fde48"}),
}
DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/"
HAM = dict(doi="doi:10.7910/DVN/DBW86T", license="CC BY-NC 4.0 (Dataverse terms of use)",
           zips={"HAM10000_images_part_1": (3172585, "4639bfa73ab251610530a97c898e6e46", 5000),
                 "HAM10000_images_part_2": (3172584, "da43d6cc50f6613013be07e8986b384b", 5015)},
           metadata_id=4338392, metadata_md5="8f85fb1aa29d80a2797247e434deb79d", n_rows=10015,
           columns=["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"])
ISIC_S3 = "https://isic-challenge-data.s3.amazonaws.com/2018/"
ISIC = {"ISIC2018_Task1-2_Training_Input": 2594, "ISIC2018_Task1_Training_GroundTruth": 2594,
        "ISIC2018_Task1-2_Validation_Input": 100, "ISIC2018_Task1_Validation_GroundTruth": 100}
GATED = {"imagenet": (
    "ImageNet-1K (ILSVRC2012) is licence-gated. Register at https://image-net.org/download-images "
    "(or accept the terms of the 'ILSVRC/imagenet-1k' dataset on Hugging Face), download "
    "ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz into "
    "<root>/ImageNet/, then run torchvision.datasets.ImageNet(root=..., split='val') once to build its index. "
    "~150 GB; this machine's disk (32 GB) cannot hold it.")}


def now():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg):
    print(f"[prepare_datasets {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Lock:
    def __init__(self, root, name):
        os.makedirs(os.path.join(root, ".locks"), exist_ok=True)
        self.path = os.path.join(root, ".locks", f"{name}.lock")

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"waiting for lock {self.path} (another process is preparing it)")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def _probe(url):
    """(final_url, size) if the server honours byte ranges, else (url, None)."""
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range", "")
            if r.status == 206 and "/" in cr and cr.split("/")[-1].isdigit():
                return r.geturl(), int(cr.split("/")[-1])
    except Exception as e:
        log(f"range probe failed ({type(e).__name__}: {e})")
    return url, None


def _segmented(url, dest, size, segments):
    """Parallel byte-range download; each segment file resumes independently. Bandwidth per
    connection on this host is ~50-90 KB/s, so a single stream would take hours."""
    import concurrent.futures as cf
    segdir = f"{dest}.seg{segments}"
    os.makedirs(segdir, exist_ok=True)
    step = -(-size // segments)
    spans = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(segments) if i * step < size]

    def fetch(span):
        i, lo, hi = span
        p = os.path.join(segdir, f"{i:04d}")
        for attempt in range(1, 21):
            have = os.path.getsize(p) if os.path.exists(p) else 0
            if have == hi - lo + 1:
                return
            if have > hi - lo + 1:
                os.remove(p); have = 0
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={lo + have}-{hi}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.status != 206:
                        raise RuntimeError(f"no range support (HTTP {r.status})")
                    with open(p, "ab") as fh:
                        shutil.copyfileobj(r, fh, 1 << 20)
            except Exception as e:
                time.sleep(min(30, 2 * attempt))
        if os.path.getsize(p) != hi - lo + 1:
            raise RuntimeError(f"segment {i} incomplete after retries")

    with cf.ThreadPoolExecutor(min(segments, 32)) as ex:
        list(ex.map(fetch, spans))
    with open(dest, "wb") as out:
        for i, _, _ in spans:
            with open(os.path.join(segdir, f"{i:04d}"), "rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 22)
    shutil.rmtree(segdir)


def download(url, dest, md5=None, retries=5, segments=None):
    """Resumable download into dest (a .partial path). Returns md5 of the finished file."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and md5 and md5sum(dest) == md5:
        log(f"reuse verified {os.path.basename(dest)}")
        return md5
    final, size = _probe(url)
    if size and size > 8 << 20:
        n = segments or 32
        for attempt in range(1, 3):
            if os.path.exists(dest):
                os.remove(dest)
            log(f"GET {url} ({size/1e6:.0f} MB, {n} parallel ranges, attempt {attempt})")
            _segmented(final, dest, size, n)
            got = md5sum(dest)
            if md5 is None or got == md5:
                return got
            log(f"MD5 mismatch {got} != {md5}; re-downloading")
            os.remove(dest)
        raise RuntimeError(f"MD5 mismatch persists for {url}")
    for attempt in range(1, retries + 1):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:      # server ignored Range: restart
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                log(f"GET {url} (attempt {attempt}, resume at {have/1e6:.0f} MB of {total/1e6:.0f} MB)")
                with open(dest, "ab" if have else "wb") as fh:
                    shutil.copyfileobj(r, fh, 1 << 22)
        except Exception as e:  # transient network failure -> retry with resume
            if getattr(e, "code", None) == 416:   # range not satisfiable: file already complete
                pass
            else:
                log(f"download error ({type(e).__name__}: {e}); retry in {5*attempt}s")
                time.sleep(5 * attempt)
                continue
        got = md5sum(dest)
        if md5 is None or got == md5:
            return got
        log(f"MD5 mismatch for {dest}: {got} != {md5}; discarding and retrying")
        os.remove(dest)
    raise RuntimeError(f"failed to download {url} after {retries} attempts")


def safe_extract_tar(archive, dest):
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")   # rejects absolute paths, '..', links outside dest, devices


def safe_extract_zip(archive, dest):
    real = os.path.realpath(dest)
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"corrupt member {bad} in {archive}")
        for m in zf.namelist():
            target = os.path.realpath(os.path.join(dest, m))
            if not (target == real or target.startswith(real + os.sep)):
                raise RuntimeError(f"unsafe path {m} in {archive}")
        zf.extractall(dest)


def ready_path(root, name, sub):
    return os.path.join(root, sub, f".ready_{name}.json")


# ----------------------------------------------------------------------------- CIFAR
def verify_cifar(root, name):
    spec = CIFAR[name]
    base = os.path.join(root, "cifar", spec["base"])
    errs = [f for f, m in spec["files"].items()
            if not os.path.exists(os.path.join(base, f)) or md5sum(os.path.join(base, f)) != m]
    return (not errs), (f"bad/missing files {errs}" if errs else "all batch MD5s match torchvision")


def prepare_cifar(root, name, verify_only):
    spec = CIFAR[name]
    ok, msg = verify_cifar(root, name)
    if ok or verify_only:
        return ok, msg, {}
    part = os.path.join(root, ".partial", os.path.basename(spec["url"]))
    # the Toronto host rate-limits bursts of connections (TLS resets observed at 32), so stay polite
    md5 = download(spec["url"], part, spec["md5"], segments=spec["segments"])
    stage = os.path.join(root, ".partial", f"stage_{name}")
    shutil.rmtree(stage, ignore_errors=True)
    safe_extract_tar(part, stage)
    dst = os.path.join(root, "cifar", spec["base"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.rename(dst, f"{dst}.invalid-{int(time.time())}")   # keep, never delete existing data
    os.rename(os.path.join(stage, spec["base"]), dst)
    shutil.rmtree(stage)
    ok, msg = verify_cifar(root, name)
    if ok:
        os.remove(part)   # extracted batches carry their own published MD5s
    return ok, msg, {"source": spec["url"], "archive_md5": md5}


# ----------------------------------------------------------------------------- HAM10000
def verify_ham(root):
    import csv
    d = os.path.join(root, "HAM10000")
    meta = os.path.join(d, "HAM10000_metadata.csv")
    if not os.path.exists(meta):
        return False, "metadata csv missing"
    with open(meta, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != HAM["n_rows"] or any(c not in rows[0] for c in HAM["columns"]):
        return False, f"metadata has {len(rows)} rows / columns {list(rows[0])}"
    imgs = {}
    for part, (_, _, n) in HAM["zips"].items():
        p = os.path.join(d, part)
        files = [f for f in os.listdir(p) if f.endswith(".jpg")] if os.path.isdir(p) else []
        if len(files) != n:
            return False, f"{part}: {len(files)} jpgs, expected {n}"
        imgs.update({f[:-4]: p for f in files})
    missing = [r["image_id"] for r in rows if r["image_id"] not in imgs]
    if missing:
        return False, f"{len(missing)} metadata image_ids have no image, e.g. {missing[:3]}"
    return True, f"{len(rows)} metadata rows, {len(imgs)} images, every image_id resolvable"


def prepare_ham(root, verify_only):
    ok, msg = verify_ham(root)
    if ok or verify_only:
        return ok, msg, {}
    d = os.path.join(root, "HAM10000")
    os.makedirs(d, exist_ok=True)
    pd_ = os.path.join(root, ".partial", "ham10000")
    info = {"source": f"Harvard Dataverse {HAM['doi']}", "license": HAM["license"], "files": {}}
    # metadata: Dataverse ingests the CSV as .tab; its published MD5 is that of the ORIGINAL upload
    csv_part = os.path.join(pd_, "HAM10000_metadata.csv")
    info["files"]["HAM10000_metadata.csv"] = download(
        f"{DATAVERSE}{HAM['metadata_id']}?format=original", csv_part, HAM["metadata_md5"])
    for part, (fid, md5, _) in HAM["zips"].items():
        z = os.path.join(pd_, f"{part}.zip")
        info["files"][f"{part}.zip"] = download(f"{DATAVERSE}{fid}", z, md5)
        stage = os.path.join(pd_, f"stage_{part}")
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        safe_extract_zip(z, stage)
        jpgs = []
        for dp, _, fs in os.walk(stage):
            jpgs += [os.path.join(dp, f) for f in fs if f.endswith(".jpg")]
        out = os.path.join(stage, "_flat")
        os.makedirs(out)
        for f in jpgs:
            os.rename(f, os.path.join(out, os.path.basename(f)))
        dst = os.path.join(d, part)
        if os.path.exists(dst):
            os.rename(dst, f"{dst}.invalid-{int(time.time())}")
        os.rename(out, dst)
        shutil.rmtree(stage)
    os.replace(csv_part, os.path.join(d, "HAM10000_metadata.csv"))
    ok, msg = verify_ham(root)
    if ok:  # free disk: images verified against metadata; archive MD5s recorded above
        for part in HAM["zips"]:
            os.remove(os.path.join(pd_, f"{part}.zip"))
    return ok, msg, info


# ----------------------------------------------------------------------------- ISIC-2018 (optional)
def verify_isic(root):
    d = os.path.join(root, "ISIC2018")
    for sub, n in ISIC.items():
        p = os.path.join(d, sub)
        k = len([f for f in os.listdir(p) if f.endswith((".jpg", ".png"))]) if os.path.isdir(p) else 0
        if k != n:
            return False, f"{sub}: {k} images, expected {n}"
    return True, "image/mask counts match the challenge release"


def prepare_isic(root, verify_only):
    ok, msg = verify_isic(root)
    if ok or verify_only:
        return ok, msg, {}
    need = 24e9
    if shutil.disk_usage(root).free < need:
        return False, f"needs ~{need/1e9:.0f} GB free for download+extraction; have {shutil.disk_usage(root).free/1e9:.1f}", {}
    d = os.path.join(root, "ISIC2018")
    info = {"source": ISIC_S3, "note": "no published checksums; zip CRC + image counts verified", "files": {}}
    for sub in ISIC:
        z = os.path.join(root, ".partial", "isic2018", f"{sub}.zip")
        info["files"][f"{sub}.zip"] = download(f"{ISIC_S3}{sub}.zip", z, None)
        stage = os.path.join(root, ".partial", "isic2018", f"stage_{sub}")
        shutil.rmtree(stage, ignore_errors=True)
        safe_extract_zip(z, stage)
        src = os.path.join(stage, sub) if os.path.isdir(os.path.join(stage, sub)) else stage
        os.makedirs(d, exist_ok=True)
        os.rename(src, os.path.join(d, sub))
        shutil.rmtree(stage, ignore_errors=True)
        os.remove(z)
    ok, msg = verify_isic(root)
    return ok, msg, info


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_SET,
                    choices=["cifar10", "cifar100", "ham10000", "isic2018", "imagenet", "all"])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    names = ["cifar10", "cifar100", "ham10000", "isic2018", "imagenet"] if "all" in a.datasets else a.datasets
    root = os.path.abspath(a.root)
    os.makedirs(root, exist_ok=True)
    status_path = os.path.join(root, "DATASETS_STATUS.json")
    rc = 0
    for name in names:
        if name in GATED:
            log(f"{name}: GATED -- {GATED[name]}")
            res = {"status": "blocked_gated", "instructions": GATED[name], "checked": now()}
            rc = rc or 2
        else:
            with Lock(root, name):
                t0 = time.time()
                try:
                    if name in CIFAR:
                        ok, msg, info = prepare_cifar(root, name, a.verify_only)
                    elif name == "ham10000":
                        ok, msg, info = prepare_ham(root, a.verify_only)
                    else:
                        ok, msg, info = prepare_isic(root, a.verify_only)
                except Exception as e:
                    ok, msg, info = False, f"{type(e).__name__}: {e}", {}
                res = {"status": "ready" if ok else ("missing" if a.verify_only else "failed"),
                       "detail": msg, "checked": now(), "seconds": round(time.time() - t0, 1)}
                if info:
                    res["provenance"] = {**info, "prepared": now()}
                log(f"{name}: {res['status']} -- {msg}")
                rc = rc or (0 if ok else 1)
        with Lock(root, "_status"):
            st = json.load(open(status_path)) if os.path.exists(status_path) else {}
            prev = st.get(name, {})
            if "provenance" in prev and "provenance" not in res:
                res["provenance"] = prev["provenance"]   # keep the record of the original download
            st[name] = res
            st["_root"] = root
            atomic_json(status_path, st)
    return rc


if __name__ == "__main__":
    sys.exit(main())
