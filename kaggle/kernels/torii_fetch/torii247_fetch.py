# Fetch Torii et al.'s original DenseVLAD (24/7 place recognition) code, vocabulary and PCA from the Internet
# Archive (the original host returns HTTP 503). CPU kernel, Internet ON. Saves everything found to /kaggle/working.
import io, json, os, re, time, urllib.parse, urllib.request, zipfile

OUT = "/kaggle/working/torii247"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (academic reproduction of VPR-Space; DenseVLAD vocabulary lookup)"}


def get(url, tries=5):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
        except Exception as e:
            print("  retry", i, url, repr(e)[:200], flush=True)
            time.sleep(10 * (i + 1))
    return None


rows = []
for prefix in ["ok.ctrl.titech.ac.jp/~torii/", "www.ok.ctrl.titech.ac.jp/~torii/", "ok.sc.e.titech.ac.jp/~torii/",
               "www.ok.sc.e.titech.ac.jp/~torii/", "www.di.ens.fr/willow/research/netvlad/", "www.di.ens.fr/~josef/"]:
    q = ("http://web.archive.org/cdx/search/cdx?url=" + urllib.parse.quote(prefix) +
         "&matchType=prefix&output=json&filter=statuscode:200&collapse=urlkey&limit=2000")
    b = get(q)
    if not b:
        continue
    try:
        data = json.loads(b or b"[]")
    except Exception:
        print("cdx parse failed", prefix, b[:200])
        continue
    print(f"CDX {prefix}: {max(len(data) - 1, 0)} urls", flush=True)
    rows += [dict(zip(data[0], r)) for r in data[1:]]
json.dump(rows, open(f"{OUT}/cdx_rows.json", "w"), indent=0)
for r in rows:
    print(r["timestamp"], r["mimetype"], r["length"], r["original"])

want = re.compile(r"(247|densevlad|dnscnt|vlad|rdsift|pcaproj|code)", re.I)
ext = re.compile(r"\.(zip|tar\.gz|tgz|tar|mat|m|txt|html?)$", re.I)
got = []
for r in rows:
    u = r["original"]
    path = urllib.parse.urlparse(u).path
    if not (want.search(u) and (ext.search(path) or path.endswith("/"))):
        continue
    try:
        if int(r["length"]) > 1_500_000_000:
            print("skip large", u, r["length"]); continue
    except ValueError:
        pass
    b = get(f"http://web.archive.org/web/{r['timestamp']}id_/{u}")
    if b is None:
        continue
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip("/")) or "index"
    if path.endswith("/"):
        name += "_index.html"
    open(f"{OUT}/{name}", "wb").write(b)
    got.append((name, len(b)))
    print("saved", name, len(b), flush=True)
    if name.lower().endswith(".zip"):
        try:
            z = zipfile.ZipFile(io.BytesIO(b))
            for zi in z.infolist():
                print("   zip:", zi.filename, zi.file_size)
            d = f"{OUT}/unzip_{name[:-4]}"
            z.extractall(d)
        except Exception as e:
            print("  bad zip", repr(e))
json.dump(got, open(f"{OUT}/downloaded.json", "w"), indent=1)
print("DONE", len(got), "files")
