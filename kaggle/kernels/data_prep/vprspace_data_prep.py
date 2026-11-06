# VPR-Space data preparation (CPU + Internet). Downloads a subset of the release's split-tar parts
# in parallel (resumable), verifies each part against the sha256 published on the GitHub release page
# and leaves the raw parts in /kaggle/working so GPU sessions (Internet OFF) can attach this
# notebook's output via kernel_sources and read members directly (no extraction needed).
import concurrent.futures as cf, hashlib, json, os, time, urllib.request

PARTS = __PARTS__  # filled in per notebook by the launcher
BASE = "https://github.com/rfechner/VPR-Space/releases/download/v0.1.0"
OUT = "/kaggle/working"
EXPECTED = {
    "vpr-space.tar.0000": "f9e90c863069bb1af5046dc5acc26827f520fe2a457b0da0e1d20f5b680787ed",
    "vpr-space.tar.0001": "1e33deb2a177ec17e4816c99d0c49765358155d171f674751514ace5020727b8",
    "vpr-space.tar.0002": "cc5022513245fb804e3c3aac73886bb35df5e6cc87a123744a3768b0b61f4832",
    "vpr-space.tar.0003": "46fde1629ebb29a206a647a76afe8f582fd23082366a6f5a5ecc354516c6b248",
    "vpr-space.tar.0004": "edfa6c23526306b586650b229f598706726461122c1295ae63da4a93dae89db6",
    "vpr-space.tar.0005": "0272619058b8b6dbb47b63b44d613ce178550e1f77aa2cc7492618d681f6f74b",
    "vpr-space.tar.0006": "fb381c4afee3c061bca298afe0385b9606beb19ce070104e0c21576fe5298ce6",
    "vpr-space.tar.0007": "8ef7f9f32626397a57ddb9ff4370eefccb8f109a9f6300c5517a4bee54b93cee",
    "vpr-space.tar.0008": "42ab056b3755fa9fa92378ddd1c14b21e44c339b5f8283f0be7cf06055d98afc",
    "vpr-space.tar.0009": "9e122c1cef958a59e4ff460e80011c1a34a0a082982079d5725034fdb8f5dc9a",
    "vpr-space.tar.0010": "c2b2a8ccc0db57c8e5081ab27104851fe2ecbef1d9552d03c63227fd3c896862",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def download(name):
    tmp = os.path.join(OUT, name + ".partial")
    for attempt in range(10):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        try:
            hdr = {"User-Agent": "vprspace-repro/1.0"}
            if have:
                hdr["Range"] = f"bytes={have}-"
            with urllib.request.urlopen(urllib.request.Request(f"{BASE}/{name}", headers=hdr), timeout=120) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                with open(tmp, mode) as f:
                    while True:
                        b = r.read(16 << 20)
                        if not b:
                            break
                        f.write(b)
            return tmp
        except Exception as e:
            log(f"{name}: attempt {attempt + 1} failed at {have} bytes: {e!r}")
            time.sleep(min(2 ** attempt, 60))
    raise IOError(f"download failed: {name}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name):
    t0 = time.time()
    for verify_round in range(2):
        tmp = download(name)
        digest = sha256(tmp)
        if digest == EXPECTED[name]:
            dest = os.path.join(OUT, name)
            os.replace(tmp, dest)
            size = os.path.getsize(dest)
            log(f"{name}: OK {size} bytes in {time.time() - t0:.0f}s ({size / max(time.time() - t0, 1e-9) / 2**20:.0f} MiB/s)")
            return {"part": name, "bytes": size, "sha256": digest, "ok": True}
        log(f"{name}: sha256 MISMATCH {digest}; re-downloading")
        os.remove(tmp)
    return {"part": name, "bytes": None, "sha256": digest, "ok": False}


if __name__ == "__main__":
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=len(PARTS)) as ex:
        results = list(ex.map(fetch, PARTS))
    summary = {"parts": results, "all_ok": all(r["ok"] for r in results), "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, f"parts_{PARTS[0][-4:]}_{PARTS[-1][-4:]}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("done", json.dumps(summary))
    if not summary["all_ok"]:
        raise SystemExit("some parts failed verification")
