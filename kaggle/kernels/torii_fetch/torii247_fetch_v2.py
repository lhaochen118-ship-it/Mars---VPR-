# Targeted version: read the archived 24/7 project page, list its links, then download (raw, id_) only the
# archived code/model files it links to. Short timeouts, progress printed. CPU kernel, Internet ON.
import json, os, re, time, urllib.parse, urllib.request, zipfile

OUT = "/kaggle/working/torii247"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (academic reproduction of VPR-Space; DenseVLAD vocabulary lookup)"}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def get(url, timeout=60, tries=3):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except Exception as e:
            log("  fail", i, url[:150], repr(e)[:120])
            time.sleep(3)
    return None


def snapshot(url):
    b = get("http://archive.org/wayback/available?url=" + urllib.parse.quote(url, safe=""), 30)
    try:
        s = json.loads(b)["archived_snapshots"].get("closest")
        return s["timestamp"] if s and s.get("available") else None
    except Exception:
        return None


pages = ["www.ok.ctrl.titech.ac.jp/~torii/project/247/", "www.ok.ctrl.titech.ac.jp/~torii/project/247/download/",
         "www.ok.ctrl.titech.ac.jp/~torii/project/repttile/", "www.ok.ctrl.titech.ac.jp/~torii/"]
links = set()
for p in pages:
    ts = snapshot(p)
    log("page", p, "snapshot", ts)
    if not ts:
        continue
    html = get(f"http://web.archive.org/web/{ts}id_/http://{p}", 90)
    if not html:
        continue
    open(f"{OUT}/" + re.sub(r"[^A-Za-z0-9]+", "_", p) + ".html", "wb").write(html)
    for h in re.findall(rb'href\s*=\s*["\']([^"\']+)', html, re.I):
        u = urllib.parse.urljoin("http://" + p, h.decode("latin-1"))
        u = re.sub(r"^https?://web\.archive\.org/web/\d+[a-z_]*/", "", u)
        links.add(u)
links = sorted(links)
log(len(links), "links")
for u in links:
    print("   LINK", u)
json.dump(links, open(f"{OUT}/links.json", "w"), indent=0)
want = [u for u in links if re.search(r"\.(zip|tar\.gz|tgz|tar|mat|m|gz)$", urllib.parse.urlparse(u).path, re.I)]
log("candidate files:", want)
got = []
for u in want:
    ts = snapshot(u.split("://", 1)[-1])
    log("file", u, "snapshot", ts)
    if not ts:
        continue
    b = get(f"http://web.archive.org/web/{ts}id_/{u}", 900, 2)
    if not b:
        continue
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", urllib.parse.urlparse(u).path.strip("/"))
    open(f"{OUT}/{name}", "wb").write(b)
    got.append([u, ts, len(b)])
    log("saved", name, len(b))
    if name.lower().endswith(".zip"):
        try:
            z = zipfile.ZipFile(f"{OUT}/{name}")
            for zi in z.infolist():
                print("   zip:", zi.filename, zi.file_size)
            z.extractall(f"{OUT}/unzip_{name[:-4]}")
        except Exception as e:
            log("bad zip", repr(e))
json.dump(got, open(f"{OUT}/downloaded.json", "w"), indent=1)
log("DONE", len(got), "files")
