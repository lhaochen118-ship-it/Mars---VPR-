# Plain (symlink-free) copies of the Hugging Face weight files used offline. Kaggle drops symlinks when saving
# outputs, which breaks the regular HF cache (snapshots/* are symlinks). Each file is downloaded by URL at a pinned
# revision and verified against the sha256 (LFS oid) reported by the Hub API.
import hashlib, json, os, shutil, time, urllib.request

W = "/kaggle/working/hf_plain"
FILES = [("gberton/MegaLoc", "a0f34722c4297ff787e022433799250180860af7", ["model.safetensors", "config.json"]),
         ("timm/vit_small_patch16_dinov3.lvd1689m", "main", ["model.safetensors", "config.json"]),
         ("timm/vit_base_patch16_dinov3.lvd1689m", "main", ["model.safetensors", "config.json"]),
         ("timm/vit_large_patch16_dinov3.lvd1689m", "main", ["model.safetensors", "config.json"])]


def get(url, dest=None):
    for t in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vprspace-repro/1.0"}), timeout=180) as r:
                if dest is None:
                    return r.read()
                with open(dest + ".tmp", "wb") as f:
                    shutil.copyfileobj(r, f, 16 << 20)
                os.replace(dest + ".tmp", dest)
                return dest
        except Exception as e:
            print("retry", url, repr(e), flush=True)
            time.sleep(2 ** t)
    raise IOError(url)


manifest = []
for repo, rev, names in FILES:
    tree = {f["path"]: f for f in json.loads(get(f"https://huggingface.co/api/models/{repo}/tree/{rev}"))}
    d = os.path.join(W, repo.replace("/", "--"))
    os.makedirs(d, exist_ok=True)
    for n in names:
        if n not in tree:
            continue
        dest = get(f"https://huggingface.co/{repo}/resolve/{rev}/{n}", os.path.join(d, n))
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for c in iter(lambda: f.read(64 << 20), b""):
                h.update(c)
        expected = (tree[n].get("lfs") or {}).get("oid")
        ok = expected is None or h.hexdigest() == expected
        manifest.append({"repo": repo, "revision": rev, "file": n, "bytes": os.path.getsize(dest), "sha256": h.hexdigest(),
                         "expected_lfs_sha256": expected, "ok": ok})
        print(repo, n, os.path.getsize(dest), "sha256 ok" if ok else "MISMATCH", flush=True)
        assert ok, (repo, n)
json.dump(manifest, open(os.path.join(W, "hf_plain_manifest.json"), "w"), indent=1)
