# VPR-Space release inspection (CPU only).
# Streams the 11 split-tar parts of rfechner/VPR-Space v0.1.0, verifies each part's sha256
# against the values shown on the GitHub release page, writes a per-file manifest
# (path, size, sha256, PNG header info), extracts only small non-image files
# (GT matrices, poses, metadata) and recomputes the paper's Table IV statistics.
import hashlib, io, json, os, platform, shutil, struct, subprocess, sys, tarfile, time, urllib.request

BASE = os.environ.get("VPRS_BASE", "https://github.com/rfechner/VPR-Space/releases/download/v0.1.0")
OUT = os.environ.get("VPRS_OUT", "/kaggle/working")
SCRATCH = os.environ.get("VPRS_SCRATCH", "/tmp/vprs_meta")
EXPECTED = {  # sha256 per part as displayed on the release page (expanded_assets/v0.1.0)
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
PARTS = json.loads(os.environ["VPRS_PARTS"]) if "VPRS_PARTS" in os.environ else sorted(EXPECTED)
SMALL_EXT = (".npz", ".npy", ".json", ".csv", ".txt", ".yaml", ".yml", ".md", ".py", ".toml", ".cfg", ".tsv", ".xml")
SMALL_MAX_FILE, SMALL_MAX_TOTAL = 200 << 20, 4 << 30

# The 16 setups exactly as in the authors' dataset.py, and the paper's Table IV
# (#DB, #Q, % queries w/ match, % of pos. matches) in the same order.
SETUPS = [("MadMax-A0", "MadMax-A2"), ("MadMax-A2", "MadMax-A4"), ("MadMax-C0", "MadMax-C1"),
          ("MadMax-E0", "MadMax-E1"), ("MadMax-F0", "MadMax-F3"), ("MadMax-G0", "MadMax-G2"),
          ("S3LI-Etna-crater", "S3LI-Etna-crater_inout"), ("S3LI-Etna-traverse_1", "S3LI-Etna-loops"),
          ("LunarLoc-9", "LunarLoc-10"), ("LunarLoc-15", "LunarLoc-17"),
          ("SPICE-A-dawn-off-spad@5.0us", "SPICE-A-night-on-spad@5.0us"),
          ("SPICE-A-dawn-off@2500.0us", "SPICE-A-night-on@2500.0us"),
          ("SPICE-A-dawn-off-spad@5.0us", "SPICE-A-noon-off-spad@5.0us"),
          ("SPICE-A-dawn-off@2500.0us", "SPICE-A-noon-off@2500.0us"),
          ("SPICE-A-noon-off-spad@5.0us", "SPICE-A-night-on-spad@5.0us"),
          ("SPICE-A-noon-off@2500.0us", "SPICE-A-night-on@2500.0us")]
TABLE_IV = {"MadMax-A0--MadMax-A2": (171, 180, 92.8, 8.55), "MadMax-A2--MadMax-A4": (180, 412, 20.9, 0.74),
            "MadMax-C0--MadMax-C1": (502, 463, 78.6, 1.89), "MadMax-E0--MadMax-E1": (517, 507, 43.2, 0.62),
            "MadMax-F0--MadMax-F3": (117, 229, 89.1, 5.14), "MadMax-G0--MadMax-G2": (258, 290, 51.7, 1.75),
            "S3LI-Etna-crater--S3LI-Etna-crater_inout": (1129, 1683, 18.2, 0.15),
            "S3LI-Etna-traverse_1--S3LI-Etna-loops": (485, 780, 36.8, 0.40),
            "LunarLoc-9--LunarLoc-10": (119, 183, 71.0, 3.19), "LunarLoc-15--LunarLoc-17": (118, 54, 27.8, 0.97),
            "SPICE-A-dawn-off-spad@5.0us--SPICE-A-night-on-spad@5.0us": (98, 97, 76.3, 1.25),
            "SPICE-A-dawn-off@2500.0us--SPICE-A-night-on@2500.0us": (99, 97, 78.4, 2.55),
            "SPICE-A-dawn-off-spad@5.0us--SPICE-A-noon-off-spad@5.0us": (98, 97, 100.0, 6.00),
            "SPICE-A-dawn-off@2500.0us--SPICE-A-noon-off@2500.0us": (99, 99, 96.0, 5.93),
            "SPICE-A-noon-off-spad@5.0us--SPICE-A-night-on-spad@5.0us": (97, 97, 72.2, 1.74),
            "SPICE-A-noon-off@2500.0us--SPICE-A-night-on@2500.0us": (99, 97, 78.4, 2.72)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def open_url(url, offset=0, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vprspace-repro/1.0",
                                                      **({"Range": f"bytes={offset}-"} if offset else {})})
            r = urllib.request.urlopen(req, timeout=120)
            if offset and r.status != 206:
                raise IOError(f"server ignored Range (status {r.status})")
            return r
        except Exception as e:
            log(f"open {url} @ {offset} failed ({e!r}); retry {t + 1}/{tries}")
            time.sleep(2 ** t)
    raise IOError(f"cannot open {url}")


class Parts(io.RawIOBase):
    """Concatenation of all parts as one read-only stream; hashes each part on the fly; resumes on errors."""

    def __init__(self, names):
        self.names, self.i, self.r, self.results = names, -1, None, []
        self.t0 = time.time()
        self._advance()

    def _advance(self):
        if self.i >= 0:
            self.r.close()
            name = self.names[self.i]
            d = self.h.hexdigest()
            self.results.append({"part": name, "bytes": self.n, "sha256": d, "expected": EXPECTED.get(name),
                                 "ok": d == EXPECTED.get(name)})
            log(f"part {name}: {self.n} bytes sha256 {'OK' if d == EXPECTED.get(name) else 'MISMATCH ' + d}"
                f" ({self.n / max(time.time() - self.tp, 1e-9) / 2**20:.0f} MiB/s)")
        self.i += 1
        if self.i < len(self.names):
            self.h, self.n, self.tp = hashlib.sha256(), 0, time.time()
            self.r = open_url(f"{BASE}/{self.names[self.i]}")
        else:
            self.r = None

    def readable(self):
        return True

    def readinto(self, b):
        while self.r is not None:
            try:
                k = self.r.readinto(b)
            except Exception as e:
                log(f"read error in {self.names[self.i]} at {self.n}: {e!r}; resuming")
                self.r.close()
                self.r = open_url(f"{BASE}/{self.names[self.i]}", offset=self.n)
                continue
            if k:
                self.h.update(memoryview(b)[:k])
                self.n += k
                return k
            self._advance()
        return 0


def png_info(head):
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        w, h = struct.unpack(">II", head[16:24])
        return w, h, head[24], head[25]  # width, height, bit depth, color type
    return None


def env_info():
    info = {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        info["mem_total_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except Exception:
        pass
    for mod in ("torch", "torchvision", "numpy", "PIL", "cv2", "xformers", "timm", "transformers",
                "pytorch_lightning", "lightning", "faiss", "sklearn", "huggingface_hub", "safetensors", "einops"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "present")
        except Exception as e:
            info[mod] = None
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else None
    except Exception:
        pass
    for cmd in (["df", "-h", "/kaggle/working", "/tmp", "/kaggle"], ["nvidia-smi"]):
        try:
            info[" ".join(cmd)] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception as e:
            info[" ".join(cmd)] = repr(e)
    return info


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    env = env_info()
    log("env:", json.dumps({k: v for k, v in env.items() if "df" not in k and "nvidia" not in k}))
    stream = Parts(PARTS)
    reader = io.BufferedReader(stream, buffer_size=16 << 20)
    manifest, saved_total, n_members, t0 = [], 0, 0, time.time()
    with tarfile.open(fileobj=reader, mode="r|") as tf:
        for m in tf:
            n_members += 1
            rec = {"path": m.name, "type": "dir" if m.isdir() else ("file" if m.isfile() else "other"),
                   "size": m.size, "mtime": m.mtime}
            if m.isfile():
                f = tf.extractfile(m)
                h, head, size = hashlib.sha256(), b"", 0
                save = (m.name.lower().endswith(SMALL_EXT) and m.size <= SMALL_MAX_FILE
                        and saved_total + m.size <= SMALL_MAX_TOTAL)
                out = None
                if save:
                    dest = os.path.join(SCRATCH, m.name.lstrip("./"))
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    out = open(dest, "wb")
                while True:
                    chunk = f.read(8 << 20)
                    if not chunk:
                        break
                    if len(head) < 64:
                        head += chunk[: 64 - len(head)]
                    h.update(chunk)
                    size += len(chunk)
                    if out:
                        out.write(chunk)
                if out:
                    out.close()
                    saved_total += size
                rec.update(sha256=h.hexdigest(), saved=bool(save))
                pi = png_info(head)
                if pi:
                    rec.update(width=pi[0], height=pi[1], bit_depth=pi[2], color_type=pi[3])
                else:
                    rec["magic"] = head[:8].hex()
            manifest.append(rec)
            if n_members % 2000 == 0:
                log(f"{n_members} members, part {stream.i}, {sum(r['size'] for r in manifest) / 2**30:.2f} GiB")
    # drain anything after the tar end-of-archive marker so every part is fully hashed
    while reader.read(16 << 20):
        pass
    log(f"stream done: {n_members} members in {time.time() - t0:.0f}s")

    with open(os.path.join(OUT, "manifest.jsonl"), "w") as fo:
        for r in manifest:
            fo.write(json.dumps(r) + "\n")
    summary = {"parts": stream.results, "all_parts_ok": all(r["ok"] for r in stream.results),
               "n_members": n_members, "n_files": sum(r["type"] == "file" for r in manifest),
               "total_file_bytes": sum(r["size"] for r in manifest if r["type"] == "file"), "env": env}
    # directory summary (top two levels) and image statistics per trajectory
    dirs = {}
    for r in manifest:
        if r["type"] != "file":
            continue
        parts = r["path"].lstrip("./").split("/")
        key = "/".join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) > 1 else "<root>")
        d = dirs.setdefault(key, {"files": 0, "bytes": 0, "ext": {}, "png_shapes": {}})
        d["files"] += 1
        d["bytes"] += r["size"]
        ext = os.path.splitext(r["path"])[1].lower()
        d["ext"][ext] = d["ext"].get(ext, 0) + 1
        if "width" in r:
            s = f"{r['width']}x{r['height']} bd{r['bit_depth']} ct{r['color_type']}"
            d["png_shapes"][s] = d["png_shapes"].get(s, 0) + 1
    summary["dirs"] = dirs
    summary["root_entries"] = sorted({r["path"].lstrip("./").split("/")[0] for r in manifest})

    # Table IV recomputation from the released GT matrices
    try:
        import numpy as np
        root = None
        for cand in [SCRATCH] + [os.path.join(SCRATCH, x) for x in os.listdir(SCRATCH)]:
            if os.path.isdir(os.path.join(cand, "vpr_gt")):
                root = cand
                break
        summary["gt_root"] = root
        rows = {}
        if root:
            gtdirs = sorted(os.listdir(os.path.join(root, "vpr_gt")))
            summary["gt_variants"] = gtdirs
            n_img, traj_stats = {}, {}
            for r in manifest:
                p = r["path"].lstrip("./").split("/")
                if r["type"] == "file" and "images" in p and r["path"].endswith(".png"):
                    traj = p[p.index("images") + 1]
                    n_img[traj] = n_img.get(traj, 0) + 1
                    ts = traj_stats.setdefault(traj, {"n": 0, "bytes": 0, "shapes": {}, "first": r["path"], "last": r["path"]})
                    ts["n"] += 1
                    ts["bytes"] += r["size"]
                    s = f"{r.get('width')}x{r.get('height')} bd{r.get('bit_depth')} ct{r.get('color_type')}"
                    ts["shapes"][s] = ts["shapes"].get(s, 0) + 1
                    ts["first"], ts["last"] = min(ts["first"], r["path"]), max(ts["last"], r["path"])
            summary["n_images_per_trajectory"] = n_img
            summary["trajectory_stats"] = traj_stats
            for db, q in SETUPS:
                sid = f"{db}--{q}"
                row = {"n_db_images": n_img.get(db), "n_q_images": n_img.get(q), "paper": TABLE_IV[sid]}
                for gv in gtdirs:
                    fp = os.path.join(root, "vpr_gt", gv, f"{sid}.npz")
                    if not os.path.exists(fp):
                        row[gv] = "missing"
                        continue
                    with np.load(fp) as g:
                        keys = list(g.keys())
                        hard, soft = g["gt_hard"].astype(bool), g["gt_soft"].astype(bool)
                        row[gv] = {"keys": keys, "dtype": str(g["gt_hard"].dtype), "shape_hard": list(hard.shape),
                                   "shape_soft": list(soft.shape), "hard_pos": int(hard.sum()),
                                   "soft_pos": int(soft.sum()), "soft_superset_of_hard": bool((soft | hard).sum() == soft.sum()),
                                   "pct_q_with_hard_match": round(100 * float((hard.sum(0) > 0).mean()), 2),
                                   "pct_q_with_any_match": round(100 * float(((hard | soft).sum(0) > 0).mean()), 2),
                                   "pct_pos_hard": round(100 * float(hard.mean()), 3),
                                   "pct_pos_union": round(100 * float((hard | soft).mean()), 3)}
                for traj in (db, q):
                    fp = os.path.join(root, "poses", f"{traj}.npy")
                    if os.path.exists(fp):
                        row[f"poses_{traj}"] = list(np.load(fp, mmap_mode="r").shape)
                rows[sid] = row
        summary["table_iv_check"] = rows
    except Exception as e:
        summary["table_iv_error"] = repr(e)
    with open(os.path.join(OUT, "summary.json"), "w") as fo:
        json.dump(summary, fo, indent=1)
    # pack the small extracted files into a single archive (keeps the output file count low)
    shutil.make_archive(os.path.join(OUT, "vprspace_small_files"), "gztar", SCRATCH)
    log("done; all parts ok:", summary["all_parts_ok"])


if __name__ == "__main__":
    main()
