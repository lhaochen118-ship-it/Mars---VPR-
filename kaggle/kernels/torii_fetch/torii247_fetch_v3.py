# Try the ORIGINAL host for Torii's 24/7 code (dense VLAD description) and keep only the small, relevant files
# (MATLAB code, vocabulary / PCA .mat). CPU kernel, Internet ON.
import io, json, os, time, urllib.request, zipfile

OUT = "/kaggle/working/torii247"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (academic reproduction of VPR-Space; DenseVLAD vocabulary lookup)"}
urls = ["http://www.ok.ctrl.titech.ac.jp/~torii/project/247/download/247code.zip",
        "https://www.ok.ctrl.titech.ac.jp/~torii/project/247/download/247code.zip",
        "http://www.ok.sc.e.titech.ac.jp/~torii/project/247/download/247code.zip"]
rep = []
for u in urls:
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=120)
        size = int(r.headers.get("Content-Length") or 0)
        print("OK", u, r.status, size, flush=True)
        path = "/tmp/247code.zip"
        with open(path, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        z = zipfile.ZipFile(path)
        names = [zi.filename for zi in z.infolist()]
        json.dump([[zi.filename, zi.file_size] for zi in z.infolist()], open(f"{OUT}/zip_listing.json", "w"), indent=0)
        for zi in z.infolist():
            low = zi.filename.lower()
            if low.endswith((".m", ".txt", ".c", ".cpp", ".h")) or (low.endswith(".mat") and zi.file_size < 2_000_000_000):
                z.extract(zi, OUT)
        rep.append({"url": u, "status": "downloaded", "bytes": os.path.getsize(path), "files": len(names)})
        break
    except Exception as e:
        print("FAIL", u, repr(e)[:300], f"{time.time() - t:.0f}s", flush=True)
        rep.append({"url": u, "status": repr(e)[:300]})
json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
